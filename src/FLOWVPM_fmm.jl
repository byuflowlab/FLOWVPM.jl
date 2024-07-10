################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = get_X(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = get_sigma(particle_field, i)[]
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.VectorPotential) where R = SVector{3,R}(0.0,0.0,0.0) # If this breaks AD: replace with 'zeros(3,R)'
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.ScalarPotential) where R = zero(R)
Base.getindex(particle_field::ParticleField, i, ::fmm.Strength) = get_Gamma(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Velocity) = get_U(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.VelocityGradient) = reshape(get_J(particle_field, i), (3, 3))
Base.getindex(particle_field::ParticleField, i, ::fmm.Body) = get_particle(particle_field, i)

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Body) = get_particle(particle_field, i) .= val

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.ScalarPotential) = nothing
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VectorPotential) = nothing
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Velocity) = set_U(particle_field, i, val)
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VelocityGradient) = set_J(particle_field, i, vec(val))

fmm.get_n_bodies(particle_field::ParticleField) = get_np(particle_field)
Base.length(particle_field::ParticleField) = get_np(particle_field) # currently called internally by the version of the FMM I'm using. this will need to be changed to work with ImplicitAD, which probably just means getting the latest FMM version. that's on hold because there are a bunch of other breaking changes I'll need to deal with to get correct derivative again.

Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where TF = TF

fmm.buffer_element(system::ParticleField) = zeros(eltype(system.particles), size(system.particles, 1))

fmm.B2M!(system::ParticleField, args...) = fmm.B2M!_vortexpoint(system, args...)

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
# Uses 1 GPU
function fmm.direct!(
        target_system::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,1},
        target_indices,
        derivatives_switch::fmm.DerivativesSwitch{PS,VPS,VS,GS},
        source_system::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,1},
        source_index) where {PS,VPS,VS,GS}

    if source_system.toggle_rbf
        for target_index in target_indices
            vorticity_direct(target_system, target_index, source_system, source_index)
        end
    else
        # Compute number of targets
        nt = 0
        for target_index in target_indices
            nt += length(target_index)
        end

        # Sets precision for computations on GPU
        # This is currently not being used for compatibility with Duals while Broadcasting
        T = Float64

        # Copy source particles from CPU to GPU
        s_d = CuArray{T}(view(source_system.particles, 1:7, source_index))

        # Copy target particles from CPU to GPU
        t_d = CuArray{T}(undef, 24, nt)
        istart = 1
        iend = 0
        for target_index in target_indices
            iend += length(target_index)
            copyto!(view(t_d, 1:24, istart:iend), target_system.particles[1:24, target_index])
            istart = iend + 1
        end

        # Get p, q for optimal GPU kernel launch configuration
        # p is no. of targets in a block
        # q is no. of columns per block
        p, q = get_launch_config(nt; T=T)

        # Compute no. of threads, no. of blocks and shared memory
        threads::Int32 = p*q
        blocks::Int32 = cld(nt, p)
        shmem = sizeof(T) * 7 * p

        # Check if GPU shared memory is sufficient
        dev = CUDA.device()
        dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        if shmem > dev_shmem
            error("Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU.
                  Try reducing ncrit, using more GPUs or reduce Chunk size if using ForwardDiff.")
        end

        # Compute interactions using GPU
        kernel = source_system.kernel.g_dgdr
        @cuda threads=threads blocks=blocks shmem=shmem gpu_atomic_direct!(s_d, t_d, q, kernel)

        # Copy back from GPU to CPU
        istart = 1
        iend = 0
        for target_index in target_indices
            iend += length(target_index)
            target_system.particles[10:12, target_index] .= Array(t_d[10:12, istart:iend])
            target_system.particles[16:24, target_index] .= Array(t_d[16:24, istart:iend])
            istart = iend + 1
        end

        # SFS contribution
        r = zero(eltype(source_system))
        for target_index in target_indices
            for j_target in target_index
                for source_particle in eachcol(view(source_system.particles, :, source_index))
                    # include self-induced contribution to SFS
                    if source_system.toggle_sfs
                        Estr_direct(target_system, j_target, source_particle, r, source_system.kernel.zeta, source_system.transposed)
                    end
                end
            end
        end
    end
    return nothing
end

# GPU kernel for Reals that uses atomic reduction (incompatible with ForwardDiff.Duals but faster)
# Uses 2 GPUs
function fmm.direct!(
        target_system::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,2},
        target_indices,
        derivatives_switch::fmm.DerivativesSwitch{PS,VPS,VS,GS},
        source_system::ParticleField{<:Real,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,2},
        source_index) where {PS,VPS,VS,GS}

    if source_system.toggle_rbf
        for target_index in target_indices
            vorticity_direct(target_system, target_index, source_system, source_index)
        end
    else
        # Get CUDA devices
        devs = CUDA.devices()

        # Compute no. of target indices to split into 2
        n_indices = length(target_indices)
        n_indices_mid = cld(n_indices, 2)

        # Sets precision for computations on GPU
        # This is currently not being used for compatibility with Duals while Broadcasting
        T = Float64

        # Raise warnings and errors
        if Threads.nthreads() < 2
            @warn "Launch atleast 2 CPU threads for better efficieny"
        end
        if length(devs) < 2
            @error "You set useGPU=2, but only $(length(devs)) GPU/s were available"
        end

        @sync begin
            Threads.@spawn begin
                # Device 1
                dev = device!(0)

                # Compute number of targets
                nt1 = 0
                for target_index in target_indices[1:n_indices_mid]
                    nt1 += length(target_index)
                end

                # Copy source particles from CPU to GPU
                s_d1 = CuArray{T}(view(source_system.particles, 1:7, source_index))

                # Copy target particles from CPU to GPU
                t_d1 = CuArray{T}(undef, 24, nt1)
                istart = 1
                iend = 0
                for target_index in target_indices[1:n_indices_mid]
                    iend += length(target_index)
                    copyto!(view(t_d1, 1:24, istart:iend), target_system.particles[1:24, target_index])
                    istart = iend + 1
                end

                # Get p, q for optimal GPU kernel launch configuration
                # p is no. of targets in a block
                # q is no. of columns per block
                p, q = get_launch_config(nt1; T=T)

                # Compute no. of threads, no. of blocks and shared memory
                threads::Int32 = p*q
                blocks::Int32 = cld(nt1, p)
                shmem = sizeof(T) * 7 * p

                # Check if GPU shared memory is sufficient
                dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
                if shmem > dev_shmem
                    error("Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU.
                          Try reducing ncrit, using more GPUs or reduce Chunk size if using ForwardDiff.")
                end

                # Compute interactions using GPU
                kernel = source_system.kernel.g_dgdr
                @cuda threads=threads blocks=blocks shmem=shmem gpu_atomic_direct!(s_d1, t_d1, q, kernel)

                # Copy back from GPU to CPU
                istart = 1
                iend = 0
                for target_index in target_indices[1:n_indices_mid]
                    iend += length(target_index)
                    target_system.particles[10:12, target_index] .= Array(t_d1[10:12, istart:iend])
                    target_system.particles[16:24, target_index] .= Array(t_d1[16:24, istart:iend])
                    istart = iend + 1
                end

            end

            Threads.@spawn begin
                # Device 2
                dev = device!(1)

                # Compute number of targets
                nt2 = 0
                for target_index in target_indices[n_indices_mid+1:end]
                    nt2 += length(target_index)
                end

                # Copy source particles from CPU to GPU
                s_d2 = CuArray{T}(view(source_system.particles, 1:7, source_index))

                # Copy target particles from CPU to GPU
                t_d2 = CuArray{T}(undef, 24, nt2)
                istart = 1
                iend = 0
                for target_index in target_indices[n_indices_mid+1:end]
                    iend += length(target_index)
                    copyto!(view(t_d2, 1:24, istart:iend), target_system.particles[1:24, target_index])
                    istart = iend + 1
                end

                # Get p, q for optimal GPU kernel launch configuration
                # p is no. of targets in a block
                # q is no. of columns per block
                p, q = get_launch_config(nt2; T=T)

                # Compute no. of threads, no. of blocks and shared memory
                threads::Int32 = p*q
                blocks::Int32 = cld(nt2, p)
                shmem = sizeof(T) * 7 * p

                # Check if GPU shared memory is sufficient
                dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
                if shmem > dev_shmem
                    error("Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU.
                          Try reducing ncrit, using more GPUs or reduce Chunk size if using ForwardDiff.")
                end

                # Compute interactions using GPU
                kernel = source_system.kernel.g_dgdr
                @cuda threads=threads blocks=blocks shmem=shmem gpu_atomic_direct!(s_d2, t_d2, q, kernel)

                # Copy back from GPU to CPU
                istart = 1
                iend = 0
                for target_index in target_indices[n_indices_mid+1:end]
                    iend += length(target_index)
                    target_system.particles[10:12, target_index] .= Array(t_d2[10:12, istart:iend])
                    target_system.particles[16:24, target_index] .= Array(t_d2[16:24, istart:iend])
                    istart = iend + 1
                end
            end
        end

        # SFS contribution
        r = zero(eltype(source_system))
        for target_index in target_indices
            for j_target in target_index
                for source_particle in eachcol(view(source_system.particles, :, source_index))
                    # include self-induced contribution to SFS
                    if source_system.toggle_sfs
                        Estr_direct(target_system, j_target, source_particle, r, source_system.kernel.zeta, source_system.transposed)
                    end
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
#         derivatives_switch::fmm.DerivativesSwitch{PS,VPS,VS,GS},
#         source_system::ParticleField{TFS,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true},
#         source_index) where {TFT,TFS,PS,VPS,VS,GS}
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
#         p, q = get_launch_config(length(target_index); T=T)
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
        target_system::ParticleField{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,0},
        target_indices,
        derivatives_switch::fmm.DerivativesSwitch{PS,VPS,VS,GS},
        source_system::ParticleField{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,0},
        source_index) where {PS,VPS,VS,GS}

    if source_system.toggle_rbf
        for target_index in target_indices
            vorticity_direct(target_system, target_index, source_system, source_index)
        end
    else
        r = zero(eltype(source_system))

        for target_index in target_indices
            for j_target in target_index
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
    end
    return nothing
end
