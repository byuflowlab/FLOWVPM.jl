# Contains utilities for handling gpu kernel
function check_launch(n, p, q, max_threads_per_block=0; throw_error=false)
    if p > n; throw_error && error("p must be less than or equal to n"); return false; end
    if p*q >= max_threads_per_block; throw_error && error("p*q must be less than $max_threads_per_block"); return false; end
    if q > p; throw_error && error("q must be less than or equal to p"); return false; end
    if n % p != 0; throw_error && error("n must be divisible by p"); return false; end
    if p % q != 0; throw_error && error("p must be divisible by q"); return false; end

    return true
end

function check_shared_memory(dev, shmem_required, throw_error=true)
    dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    if shmem > dev_shmem
        msg = "Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU. Try reducing ncrit, using more GPUs or reduce Chunk size if using ForwardDiff."
        if throw_error
            error(msg)
        else
            @warn msg
        end
    end
    return
end

function get_launch_config(nt; p_max=0, q_max=0, max_threads_per_block=256)
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? p_max : q_max

    divs_n = sort(divisors(nt))
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_n)
        if div <= p_max
            p = div
            ip = i
        else
            break
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    if nt <= 1<<13
        isgood = true
        for i in 1:ip
            for j in 1:ip
                isgood = check_launch(nt, divs_n[i], divs_n[j], max_threads_per_block)
                if isgood && (divs_n[i] <= p_max)
                    # Check if this is the max achievable ij value
                    # in the p, q choice matrix
                    obj_val = i_weight*i+j_weight*j
                    if (obj_val >= max_ij) && (divs_n[j] <= q_max)
                        max_ij = obj_val
                        p = divs_n[i]
                        q = divs_n[j]
                    end
                end
            end
        end
    end

    return p, q
end

const eps2 = 1e-6
const const4 = 0.25/pi
@inline function gpu_interaction(tx, ty, tz, s, j, kernel)
    T = eltype(s)
    @inbounds dX1 = tx - s[1, j]
    @inbounds dX2 = ty - s[2, j]
    @inbounds dX3 = tz - s[3, j]
    r2 = dX1*dX1 + dX2*dX2 + dX3*dX3
    r = sqrt(r2)
    r3 = r*r2

    # Mapping to variables
    @inbounds gam1 = s[4, j]
    @inbounds gam2 = s[5, j]
    @inbounds gam3 = s[6, j]
    @inbounds sigma = s[7, j]

    UJ = @MVector zeros(T, 12)

    if r2 > T(eps2) && abs(sigma) > T(eps2)
        # Regularizing function and deriv
        # g_sgm = g_val(r/sigma)
        # dg_sgmdr = dg_val(r/sigma)
        g_sgm, dg_sgmdr = kernel(r/sigma)

        # K × Γp
        crss1 = -T(const4) / r3 * ( dX2*gam3 - dX3*gam2 )
        crss2 = -T(const4) / r3 * ( dX3*gam1 - dX1*gam3 )
        crss3 = -T(const4) / r3 * ( dX1*gam2 - dX2*gam1 )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        @inbounds UJ[1] = g_sgm * crss1
        @inbounds UJ[2] = g_sgm * crss2
        @inbounds UJ[3] = g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r2
        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux2 = -T(const4) * g_sgm / r3
        # j=1
        @inbounds UJ[4] = aux * crss1 * dX1
        @inbounds UJ[5] = aux * crss2 * dX1 - aux2 * gam3
        @inbounds UJ[6] = aux * crss3 * dX1 + aux2 * gam2
        # j=2
        @inbounds UJ[7] = aux * crss1 * dX2 + aux2 * gam3
        @inbounds UJ[8] = aux * crss2 * dX2
        @inbounds UJ[9] = aux * crss3 * dX2 - aux2 * gam1
        # j=3
        @inbounds UJ[10] = aux * crss1 * dX3 - aux2 * gam2
        @inbounds UJ[11] = aux * crss2 * dX3 + aux2 * gam1
        @inbounds UJ[12] = aux * crss3 * dX3
    end

    return UJ
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
function gpu_atomic_direct!(s, t, p, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x

    # Row and column indices of threads in a block
    row::Int32 = (ithread-1) % p + 1
    col::Int32 = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    if itarget <= t_size
        @inbounds tx = t[1, itarget]
        @inbounds ty = t[2, itarget]
        @inbounds tz = t[3, itarget]
    end

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    # 12 for UJ variables that have to be reduced at the end
    sh_mem = CuDynamicSharedArray(eltype(t), (7, p))
    # sh_mem = CuDynamicSharedArray(eltype(t), (12*p, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    out = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            isource = row + (itile-1)*p
            idim = 1
            if isource <= s_size
                while idim <= 7
                    @inbounds sh_mem[idim, row] = s[idim, isource]
                    idim += 1
                end
            else
                while idim <= 7
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                if itarget <= t_size
                    out .= gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)
                end

                # Sum up influences for each source in a tile
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    idim = 1
    if itarget <= t_size
        while idim <= 3
            @inbounds CUDA.@atomic t[9+idim, itarget] += UJ[idim]
            idim += 1
        end
        idim = 4
        while idim <= 12
            @inbounds CUDA.@atomic t[12+idim, itarget] += UJ[idim]
            idim += 1
        end
    end
    return
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
# Final summation through parallel reduction instead of atomic reduction
# Low-storage parallel reduction
# - p is no. of targets per block. Typically same as no. of sources per block.
# - q is no. of columns per tile
function gpu_reduction_direct!(s, t, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x
    p::Int32 = t_size/gridDim().x

    # Row and column indices of threads in a block
    row::Int32 = (ithread-1) % p + 1
    col::Int32 = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    @inbounds tx = t[1, itarget]
    @inbounds ty = t[2, itarget]
    @inbounds tz = t[3, itarget]

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    sh_mem = CuDynamicSharedArray(eltype(t), (12, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    out = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    idx::Int32 = 0
    i::Int32 = 0
    isource::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1)
            idx = row + (itile-1)*p
            idim = 1
            if idx <= s_size
                while idim <= 7
                    @inbounds sh_mem[idim, row] = s[idim, idx]
                    idim += 1
                end
            else
                while idim <= 7
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1)
            if isource <= s_size
                out .= gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)

                # Sum up influences for each source in a column in the tile
                # This UJ resides in the local memory of the thread corresponding
                # to each column, so we haven't summed up over the tile yet.
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] += out[idim]
                    idim += 1
                end
            end
            i += 1
        end
        itile += 1
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    if num_cols != 1
        # Perform write to shared memory
        # Columns correspond to each of the q threads
        # Iterate over targets and do reduction
        it::Int32 = 1
        while it <= p
            # Threads corresponding to itarget will copy their data to shared mem
            if itarget == it+p*(blockIdx().x-1)
                idim = 1
                while idim <= 12
                    @inbounds sh_mem[idim, col] = UJ[idim]
                    idim += 1
                end
            end
            sync_threads()

            # All p*q threads do parallel reduction on data
            stride::Int32 = 1
            while stride < num_cols
                i = (threadIdx().x-1)*stride*2+1
                if i+stride <= num_cols
                    idim = 1
                    while idim <= 12  # This can be parallelized too
                        @inbounds sh_mem[idim, i] += sh_mem[idim, i+stride]
                        idim += 1
                    end
                end
                stride *= 2
                sync_threads()
            end

            # col 1 of the threads that handle it target
            # writes reduced data to its own local memory
            if itarget == it+p*(blockIdx().x-1) && col == 1
                idim = 1
                while idim <= 12
                    @inbounds UJ[idim] = sh_mem[idim, 1]
                    idim += 1
                end
            end

            it += 1
        end
    end

    # Now, each col 1 has the net influence of all sources on its target
    # Write all data back to global memory
    if col == 1
        idim = 1
        while idim <= 3
            @inbounds t[9+idim, itarget] += UJ[idim]
            idim += 1
        end
        idim = 4
        while idim<= 12
            @inbounds t[12+idim, itarget] += UJ[idim]
            idim += 1
        end
    end

    return
end

function expand_indices!(expanded_indices, indices)
    i = 1
    for index in indices
        expanded_indices[i:i+length(index)-1] .= index
        i += length(index)
    end
    return
end

function count_leaves(target_indices, source_indices)
    leaf_idx = Vector{Int}(undef, length(target_indices))
    leaf_idx[1] = 1
    count = 1
    idx = target_indices[1][1]
    for i = 1:length(target_indices)
        if idx != target_indices[i][1]
            count += 1
            idx = target_indices[i][1]
        end
        leaf_idx[i] = count
    end

    leaf_target_indices = Vector{UnitRange{Int}}(undef, count)
    leaf_source_indices = [Vector{UnitRange{Int}}() for i = 1:count]
    idx = 0
    for i = 1:length(target_indices)
        push!(leaf_source_indices[leaf_idx[i]], source_indices[i])
        if idx != leaf_idx[i]
            leaf_target_indices[leaf_idx[i]] = target_indices[i]
            idx += 1
        end
    end
    return count, leaf_target_indices, leaf_source_indices
end

# Convenience function to compile the GPU kernel
# so compilation doesn't take time later
function warmup_gpu(verbose=false; n=100)
    ngpu::Int = length(CUDA.devices())
    if ngpu == 0
        @warn("No CUDA device/s found")
    else
        verbose && @info("$ngpu CUDA device/s found")

        # Create particle field
        pfield = ParticleField(n; useGPU=2)

        # Set no. of dummy particles
        pfield.np = n

        # Derivative switch for direct function
        d_switch = FastMultipole.DerivativesSwitch()

        # Create ngpu leaves each with 1:n particles
        target_indices = fill(1:n, ngpu)
        source_indices = fill(1:n, ngpu)

        # Run direct computation on particles
        fmm.direct_gpu!(pfield, target_indices, d_switch, pfield, source_indices)

        verbose && @info("CUDA kernel compiled successfully on $ngpu device/s")
    end

    return
end
