################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = get_X(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = get_sigma(particle_field, i)[]
#Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.VectorPotential) where R = SVector{3,R}(0.0,0.0,0.0) # If this breaks AD: replace with 'zeros(3,R)'
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.ScalarPotential) where R = zero(R)
Base.getindex(particle_field::ParticleField, i, ::fmm.Strength) = get_Gamma(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Velocity) = get_U(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.VelocityGradient) = reshape(get_J(particle_field, i), (3, 3))
Base.getindex(particle_field::ParticleField, i, ::fmm.Body) = get_particle(particle_field, i)

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Body) = get_particle(particle_field, i) .= val

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.ScalarPotential) = nothing
#Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VectorPotential) = nothing
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Velocity) = set_U(particle_field, i, val)
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VelocityGradient) = set_J(particle_field, i, vec(val))

fmm.get_n_bodies(particle_field::ParticleField) = get_np(particle_field)
Base.length(particle_field::ParticleField) = get_np(particle_field) # currently called internally by the version of the FMM I'm using. this will need to be changed to work with ImplicitAD, which probably just means getting the latest FMM version. that's on hold because there are a bunch of other breaking changes I'll need to deal with to get correct derivative again.

Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where TF = TF

fmm.buffer_element(system::ParticleField) = zeros(eltype(system.particles), size(system.particles, 1))

fmm.body_to_multipole!(system::ParticleField, args...) = fmm.body_to_multipole!(fmm.Point{fmm.Vortex}, system, args...)

@inline function vorticity_direct(target_system::ParticleField, target_index, source_system, source_index)
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target, fmm.POSITION]
        Wx = zero(eltype(target_system))
        Wy = zero(eltype(target_system))
        Wz = zero(eltype(target_system))
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = get_Gamma(source_system, i_source)
            source_x, source_y, source_z = get_X(source_system, i_source)
            sigma = get_sigma(source_system, i_source)[]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz # sqrt hahs an undefined derivative at r=0, so AD gets NaNs introduced without this check.
            if r2 > 0
                r = sqrt(r2)
                zeta = source_system.zeta(r/sigma)/(sigma*sigma*sigma)
                Wx += zeta * gamma_x
                Wy += zeta * gamma_y
                Wz += zeta * gamma_z
            end
        end
        get_vorticity(target_system, j_target) .+= Wx, Wy, Wz
    end
end

@inline function vorticity_direct(target_system, target_index, source_system, source_index)
    return nothing
end

@inline function Estr_direct(target_system::ParticleField, j_target, source_particle, r, zeta, transposed)
    Estr_direct(target_system[j_target, fmm.BODY], source_particle, r, zeta, transposed)
end

@inline function Estr_direct(target_system, j_target, source_particle, r, zeta, transposed)
    return nothing
end

# GPU kernel for Reals that uses atomic reduction (incompatible with ForwardDiff.Duals but faster)
# Uses all available GPUs and single stream
# Each leaf is loaded on to a gpu and the kernel is launched. Results are copied back after
# all available gpus are launched asynchronously.
function fmm.nearfield_device!(
        target_systems::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any, 1},
        target_tree::fmm.Tree,
        derivatives_switches::fmm.DerivativesSwitch{PS,VS,GS},
        source_systems::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any, 1},
        source_tree::fmm.Tree,
        direct_list) where {PS,VS,GS}

    # Sort the direct_list by targets
    # This is to avoid race conditions when parallelized with multiple gpus
    sorted_direct_list = fmm.sort_by_target(direct_list, target_tree.branches)

    # The gpu kernel requires the source indices to be expanded to a single array
    target_sources = combine_source_indices(sorted_direct_list, source_tree.branches)
    leaf_count = length(target_sources)

    # Check if direct interaction is required for the entire domain
    fully_direct = (leaf_count == 8) && is_fully_direct(target_sources)

    ngpus = length(CUDA.devices())

    # Sets precision for computations on GPU
    T = Float64

    # Dummy initialization so that UJ_d is defined in all lower scopes
    UJ_d_list = Vector{CuArray{T, 2}}(undef, ngpus)

    if fully_direct && ngpus == 1
        ns = get_np(source_systems)

        # Copy source particles from CPU to GPU
        s_d = CuArray{T}(view(source_systems.particles, 1:7, 1:ns))

        # Pad target array to nearest multiple of 32 (warp size)
        # for efficient p, q launch config
        t_padding = 0
        nt = ns
        if mod(nt, 32) != 0
            t_padding = 32*cld(nt, 32) - nt
        end
        t_size = nt + t_padding

        # Copy target particles from CPU to GPU
        t_d = s_d
        UJ_d = CUDA.zeros(T, 12, nt)

        # Get p, q for optimal GPU kernel launch configuration
        # p is no. of targets in a block
        # q is no. of columns per block
        p, q = get_launch_config(t_size; max_threads_per_block=512)

        # Compute no. of threads, no. of blocks and shared memory
        threads = p*q
        blocks = cld(t_size, p)
        shmem = sizeof(T) * 7 * p

        # Check if GPU shared memory is sufficient
        dev = CUDA.device()
        check_shared_memory(dev, shmem)

        # Compute interactions using GPU
        kernel = source_systems.kernel.g_dgdr
        @cuda threads=threads blocks=blocks shmem=shmem gpu_atomic_direct!(UJ_d, s_d, t_d, Int32(p), Int32(q), kernel)

        view(target_systems.particles, 10:12, 1:nt) .= Array(view(UJ_d, 1:3, :))
        view(target_systems.particles, 16:24, 1:nt) .= Array(view(UJ_d, 4:12, :))

        # Clear GPU array to avoid GC pressure
        CUDA.unsafe_free!(UJ_d)
    else
        ileaf = 1
        while ileaf <= leaf_count
            leaf_remaining = leaf_count-ileaf+1

            ileaf_gpu = ileaf
            # Copy data to GPU and launch kernel
            for igpu in min(ngpus, leaf_remaining):-1:1

                # Set gpu
                dev = CUDA.device!(igpu-1)

                # Compute number of sources
                source_indices = expand_source_indices(target_sources[ileaf_gpu], source_tree.branches)
                ns = length(source_indices)


                # Copy source particles from CPU to GPU
                s_d = CuArray{T}(view(source_systems.particles, 1:7, source_indices))

                # Pad target array to nearest multiple of 32 (warp size)
                # for efficient p, q launch config
                t_padding = 0
                target_index_range = target_tree.branches[target_sources[ileaf_gpu][1]].bodies_index
                nt = length(target_index_range)
                if mod(nt, 32) != 0
                    t_padding = 32*cld(nt, 32) - nt
                end

                # Copy target particles from CPU to GPU
                t_d = CuArray{T}(view(target_systems.particles, 1:7, target_index_range))
                UJ_d = CUDA.zeros(T, 12, nt)
                t_size = nt + t_padding

                # Get p, q for optimal GPU kernel launch configuration
                # p is no. of targets in a block
                # q is no. of columns per block
                p, q = get_launch_config(t_size; max_threads_per_block=512)

                # Compute no. of threads, no. of blocks and shared memory
                threads::Int32 = p*q
                blocks::Int32 = cld(t_size, p)
                shmem = sizeof(T) * 7 * p

                # Check if GPU shared memory is sufficient
                check_shared_memory(dev, shmem)

                # Compute interactions using GPU
                kernel = source_systems.kernel.g_dgdr
                @cuda threads=threads blocks=blocks shmem=shmem gpu_atomic_direct!(UJ_d, s_d, t_d, Int32(p), Int32(q), kernel)

                UJ_d_list[igpu] = UJ_d

                ileaf_gpu += 1
            end

            ileaf_gpu = ileaf
            for igpu in min(ngpus, leaf_remaining):-1:1
                # Set gpu
                CUDA.device!(igpu-1)

                target_index_range = target_tree.branches[target_sources[ileaf_gpu][1]].bodies_index

                # Copy results back from GPU to CPU
                UJ_d = UJ_d_list[igpu]
                view(target_systems.particles, 10:12, target_index_range) .= Array(view(UJ_d, 1:3, :))
                view(target_systems.particles, 16:24, target_index_range) .= Array(view(UJ_d, 4:12, :))

                # Clear GPU array to avoid GC pressure
                CUDA.unsafe_free!(UJ_d)

                ileaf_gpu += 1
            end

            ileaf = ileaf_gpu
        end
    end

    # SFS contribution
    if source_systems.toggle_sfs
        r = zero(eltype(source_systems))
        for (target_index, source_index) in zip(target_indices, source_indices)
            for j_target in target_index
                for source_particle in eachcol(view(source_systems.particles, :, source_index))
                    # include self-induced contribution to SFS
                    Estr_direct(target_systems, j_target, source_particle, r, source_systems.kernel.zeta, source_systems.transposed)
                end
            end
        end
    end
    return nothing
end

# GPU kernel for Reals that uses atomic reduction (incompatible with ForwardDiff.Duals but faster)
# Uses all available GPUs and mulitple streams per gpu
# Each leaf is loaded on to a gpu and the kernel is launched. Results are copied back after
# all available gpus are launched asynchronously.
function fmm.nearfield_device!(
        target_systems::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any, 2},
        target_tree::fmm.Tree,
        derivatives_switches::fmm.DerivativesSwitch{PS,VS,GS},
        source_systems::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any, 2},
        source_tree::fmm.Tree,
        direct_list) where {PS,VS,GS}

    # Sort the direct_list by targets
    # This is to avoid race conditions when parallelized with multiple gpus
    sorted_direct_list = fmm.sort_by_target(direct_list, target_tree.branches)

    # The gpu kernel requires the source indices to be expanded to a single array
    target_sources = combine_source_indices(sorted_direct_list, source_tree.branches)
    leaf_count = length(target_sources)

    ngpus = length(CUDA.devices())
    nstreams_per_gpu = 2
    nstreams = nstreams_per_gpu * ngpus
    streams = Vector{CuStream}(undef, nstreams)

    # Sets precision for computations on GPU
    T = Float64

    # Dummy initialization so that UJ_d is defined in all lower scopes
    UJ_d_list = Vector{CuArray{T, 2}}(undef, nstreams)

    ileaf = 1
    while ileaf <= leaf_count
        leaf_remaining = leaf_count-ileaf+1

        ileaf_stream = ileaf
        igpu = 1
        # Copy data to GPU and launch kernel
        for istream in min(nstreams, leaf_remaining):-1:1

            # Set gpu and stream
            dev = CUDA.device!(igpu-1)
            streams[istream] = CuStream()

            # Compute number of sources
            source_indices = expand_source_indices(target_sources[ileaf_stream], source_tree.branches)
            ns = length(source_indices)

            # Copy source particles from CPU to GPU
            s_d = CuArray{T}(view(source_systems.particles, 1:7, source_indices))

            # Pad target array to nearest multiple of 32 (warp size)
            # for efficient p, q launch config
            t_padding = 0
            target_index_range = target_tree.branches[target_sources[ileaf_stream][1]].bodies_index
            nt = length(target_index_range)
            if mod(nt, 32) != 0
                t_padding = 32*cld(nt, 32) - nt
            end

            # Copy target particles from CPU to GPU
            t_d = CuArray{T}(view(target_systems.particles, 1:7, target_index_range))
            t_size = nt + t_padding

            # Initialize output array
            UJ_d = CUDA.zeros(T, 12, nt)

            # Get p, q for optimal GPU kernel launch configuration
            # p is no. of targets in a block
            # q is no. of columns per block
            p, q = get_launch_config(t_size; max_threads_per_block=512)

            # Compute no. of threads, no. of blocks and shared memory
            threads::Int32 = p*q
            blocks::Int32 = cld(t_size, p)
            shmem = sizeof(T) * 7 * p

            # Check if GPU shared memory is sufficient
            check_shared_memory(dev, shmem)

            # Compute interactions using GPU
            kernel = source_systems.kernel.g_dgdr
            @cuda threads=threads blocks=blocks stream=streams[istream] shmem=shmem gpu_atomic_direct!(UJ_d, s_d, t_d, Int32(p), Int32(q), kernel)

            UJ_d_list[istream] = UJ_d

            ileaf_stream += 1
            igpu = (igpu % ngpus) + 1  # Cycle igpu over 1:ngpus
        end

        ileaf_stream = ileaf
        istream = 1
        igpu = 1
        for istream in min(nstreams, leaf_remaining):-1:1
            # Set gpu and stream
            CUDA.device!(igpu-1)
            stream!(streams[istream]) do

                target_index_range = target_tree.branches[target_sources[ileaf_stream][1]].bodies_index

                # Copy results back from GPU to CPU
                UJ_d = UJ_d_list[istream]
                view(target_systems.particles, 10:12, target_index_range) .= Array(view(UJ_d, 1:3, :))
                view(target_systems.particles, 16:24, target_index_range) .= Array(view(Uj_d, 4:12, :))

                # Clear GPU array to avoid GC pressure
                CUDA.unsafe_free!(UJ_d)
            end

            ileaf_stream += 1
            igpu = (igpu % ngpus) + 1  # Cycle igpu over 1:ngpus
        end

        ileaf = ileaf_stream
    end

    # SFS contribution
    if source_systems.toggle_sfs
        r = zero(eltype(source_systems))
        for (target_index, source_index) in zip(target_indices, source_indices)
            for j_target in target_index
                for source_particle in eachcol(view(source_systems.particles, :, source_index))
                    # include self-induced contribution to SFS
                    Estr_direct(target_systems, j_target, source_particle, r, source_systems.kernel.zeta, source_systems.transposed)
                end
            end
        end
    end
    return nothing
end

# GPU kernel for ForwardDiff.Duals that uses parallel reduction
# function fmm.direct!(
#         target_system::ParticleField{TFT,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true},
#         target_index,
#         derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS},
#         source_system::ParticleField{TFS,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true},
#         source_index) where {TFT,TFS,PS,VS,GS}
#
#     if source_system.toggle_rbf
#         vorticity_direct(target_system, target_index, source_system, source_index)
#     else
#         # Sets precision for computations on GPU
#         # This is currently not being used for compatibility with Duals while Broadcasting
#         T = Float64
#
#         # Copy data from CPU to GPU
#         s_d = CuArray{TFS}(view(source_system.particles, 1:7, source_index))
#         t_d = CuArray{TFT}(view(target_system.particles, 1:24, target_index))
#
#         # Get p, q for optimal GPU kernel launch configuration
#         # p is no. of targets in a block
#         # q is no. of columns per block
#         p, q = get_launch_config(length(target_index); max_threads_per_block=512)
#
#         # Compute no. of threads, no. of blocks and shared memory
#         threads::Int32 = p*q
#         blocks::Int32 = cld(length(target_index), p)
#         shmem = sizeof(TFT) * 12 * p # XYZ + Γ123 + σ = 7 variables but (12*p) to handle UJ summation for each target
#
#         # Check if GPU shared memory is sufficient
#         dev = CUDA.device()
#         dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
#         if shmem > dev_shmem
#             error("Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU.
#                   Try using more GPUs or reduce Chunk size if using ForwardDiff.")
#         end
#
#         # Compute interactions using GPU
#         kernel = source_system.kernel.g_dgdr
#         @cuda threads=threads blocks=blocks shmem=shmem gpu_reduction_direct!(s_d, t_d, q, kernel)
#
#         # Copy back data from GPU to CPU
#         view(target_system.particles, 10:12, target_index) .= Array(t_d[10:12, :])
#         view(target_system.particles, 16:24, target_index) .= Array(t_d[16:24, :])
#
#         # SFS contribution
#         r = zero(eltype(source_system))
#         for j_target in target_index
#             for source_particle in eachcol(view(source_system.particles, :, source_index))
#                 # include self-induced contribution to SFS
#                 if source_system.toggle_sfs
#                     Estr_direct(target_system, j_target, source_particle, r, source_system.kernel.zeta, source_system.transposed)
#                 end
#             end
#         end
#     end
#
#
#     return nothing
# end

# CPU kernel
function fmm.direct!(
        target_system::ParticleField, target_index,
        derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS},
        source_system::ParticleField, source_index) where {PS,VS,GS}

    if source_system.toggle_rbf
        for target_index in target_indices
            vorticity_direct(target_system, target_index, source_system, source_index)
        end
    else

        Threads.@threads for j_target in target_index
            r = zero(eltype(source_system))  # Moved inside loop for thread-safety
            target_x, target_y, target_z = target_system[j_target, fmm.POSITION]

            for source_particle in eachcol(view(source_system.particles, :, source_index))
                gamma_x, gamma_y, gamma_z = get_Gamma(source_particle)
                source_x, source_y, source_z = get_X(source_particle)
                sigma = get_sigma(source_particle)[]
                dx = target_x - source_x
                dy = target_y - source_y
                dz = target_z - source_z
                r2 = dx*dx + dy*dy + dz*dz
                if !iszero(r2)
                    r = sqrt(r2)
                    # Regularizing function and deriv
                    g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)

                    # K × Γp
                    crss1 = -const4 / r^3 * ( dy*gamma_z - dz*gamma_y )
                    crss2 = -const4 / r^3 * ( dz*gamma_x - dx*gamma_z )
                    crss3 = -const4 / r^3 * ( dx*gamma_y - dy*gamma_x )

                    # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                    Ux = g_sgm * crss1
                    Uy = g_sgm * crss2
                    Uz = g_sgm * crss3
                    # get_U(target_particle) .+= Ux, Uy, Uz
                    Ux0, Uy0, Uz0 = target_system[j_target, fmm.VELOCITY]
                    target_system[j_target, fmm.VELOCITY] = Ux+Ux0, Uy+Uy0, Uz+Uz0

                    # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                    # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                    aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r^2
                    # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                    # Adds the Kronecker delta term
                    aux2 = -const4 * g_sgm / r^3
                    # j=1
                    du1x1 = aux * crss1 * dx
                    du2x1 = aux * crss2 * dx - aux2 * gamma_z
                    du3x1 = aux * crss3 * dx + aux2 * gamma_y
                    # j=2
                    du1x2 = aux * crss1 * dy + aux2 * gamma_z
                    du2x2 = aux * crss2 * dy
                    du3x2 = aux * crss3 * dy - aux2 * gamma_x
                    # j=3
                    du1x3 = aux * crss1 * dz - aux2 * gamma_y
                    du2x3 = aux * crss2 * dz + aux2 * gamma_x
                    du3x3 = aux * crss3 * dz

                    du1x10, du2x10, du3x10, du1x20, du2x20, du3x20, du1x30, du2x30, du3x30 = target_system[j_target, fmm.VELOCITY_GRADIENT]
                    target_system[j_target, fmm.VELOCITY_GRADIENT] = SMatrix{3,3}(
                                                                                  du1x10 + du1x1,
                                                                                  du2x10 + du2x1,
                                                                                  du3x10 + du3x1,
                                                                                  du1x20 + du1x2,
                                                                                  du2x20 + du2x2,
                                                                                  du3x20 + du3x2,
                                                                                  du1x30 + du1x3,
                                                                                  du2x30 + du2x3,
                                                                                  du3x30 + du3x3
                                                                                 )
                end

                # include self-induced contribution to SFS
                if source_system.toggle_sfs
                    Estr_direct(target_system, j_target, source_particle, r, source_system.kernel.zeta, source_system.transposed)
                end
            end
        end
    end
    return nothing
end
