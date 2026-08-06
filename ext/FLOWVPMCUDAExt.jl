#=##############################################################################
# DESCRIPTION
    CUDA.jl package extension: GPU kernels for the direct (no-FMM) N-body sum.
    Loaded automatically whenever both FLOWVPM and CUDA are loaded in the same
    environment (Julia >=1.9 package extension mechanism).
=###############################################################################
module FLOWVPMCUDAExt

using FLOWVPM
using CUDA
using CUDA: i32
using StaticArrays: @MVector
using Primes: divisors
import FastMultipole
const fmm = FastMultipole

const default_max_threads_per_block::Int32 = 512

function check_launch(n, p, q,
        max_threads_per_block=default_max_threads_per_block; throw_error=false)

    if p > n; throw_error && error("p must be less than or equal to n"); return false; end
    if p*q >= max_threads_per_block; throw_error && error("p*q must be less than $max_threads_per_block"); return false; end
    if q > p; throw_error && error("q must be less than or equal to p"); return false; end
    if n % p != 0; throw_error && error("n must be divisible by p"); return false; end
    if p % q != 0; throw_error && error("p must be divisible by q"); return false; end

    return true
end

function check_launch(nt, ns, p, q, r,
        max_threads_per_block=default_max_threads_per_block; throw_error=false)

    if p > nt; throw_error && error("p must be less than or equal to nt"); return false; end
    if p*q > max_threads_per_block; throw_error && error("p*q must be less than $max_threads_per_block"); return false; end
    # if q > p; throw_error && error("q must be less than or equal to p"); return false; end
    if q > r; throw_error && error("q must be less than or equal to r"); return false; end
    if nt % p != 0; throw_error && error("nt must be divisible by p"); return false; end
    # if p % q != 0; throw_error && error("p must be divisible by q"); return false; end
    if ns % r != 0; throw_error && error("ns must be divisible by p"); return false; end
    if r % q != 0; throw_error && error("r must be divisible by q"); return false; end

    return true
end


function check_shared_memory(dev, shmem_required, throw_error=true)
    dev_shmem = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    if shmem_required > dev_shmem
        msg = "Shared memory requested ($shmem B), exceeds available space ($dev_shmem B) on GPU. Try reducing ncrit, using more GPUs or reduce Chunk size if using ForwardDiff."
        if throw_error
            error(msg)
        else
            @warn msg
        end
    end
    return
end

@inline function get_launch_config(nt; p_max=0, q_max=0,
        max_threads_per_block=default_max_threads_per_block)

    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? p_max : q_max

    divs_n = sort(divisors(nt))
    p = 1
    q = 1
    r = 1  # r is returned only for consistency with the other get_launch_config()
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

    return p, q, r
end

@inline function get_launch_config(nt, ns; p_max=0, q_max=0, r_max=875,
        max_threads_per_block=default_max_threads_per_block)

    # r_max=875 corresponds to 48KB in shared memory
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (q_max == 0) ? max_threads_per_block : q_max

    # Find p
    divs_nt = sort(divisors(nt))
    p = 1
    q = 1
    ip = 1
    for (i, div) in enumerate(divs_nt)
        if div <= p_max
            p = div
            ip = i
        else
            break
        end
    end

    # Find r
    divs_ns = sort(divisors(ns))
    r = 1
    ir = 1
    for (i, div) in enumerate(divs_ns)
        if div <= r_max
            r = div
            ir = i
        else
            break
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    # Find q based on r
    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*1
    isgood = true
    for i in 1:ip
        for j in 1:ir
            isgood = check_launch(nt, ns, divs_nt[i], divs_ns[j], r, max_threads_per_block)
            # isgood = divs_nt[i]*divs_ns[j] < max_threads_per_block
            if isgood && (divs_nt[i] <= p_max)
                # Check if this is the max achievable ij value
                # in the p, q choice matrix
                obj_val = i_weight*i+j_weight*j
                if (obj_val >= max_ij) && (divs_ns[j] <= q_max)
                    max_ij = obj_val
                    p = divs_nt[i]
                    q = divs_ns[j]
                end
            end
        end
    end

    return p, q, r
end

const eps2 = 1e-6
const const4 = 0.25/pi
@inline function gpu_interaction!(UJ, tx, ty, tz, s, j, kernel)
    T = eltype(s)
    @inbounds dX1 = tx - s[1i32, j]
    @inbounds dX2 = ty - s[2i32, j]
    @inbounds dX3 = tz - s[3i32, j]
    r2 = dX1^2 + dX2^2 + dX3^2
    r = sqrt(r2)

    # Mapping to variables
    @inbounds sigma = s[7i32, j]

    if r2 > T(eps2) && abs(sigma) > T(eps2)
        # Mapping to variables
        c4 = -T(const4)/(r*r2)
        @inbounds gam1 = c4 * s[4i32, j]
        @inbounds gam2 = c4 * s[5i32, j]
        @inbounds gam3 = c4 * s[6i32, j]

        # Regularizing function and deriv
        # g_sgm = g_val(r/sigma)
        # dg_sgmdr = dg_val(r/sigma)
        g_sgm, dg_sgmdr = kernel(r/sigma)

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r2

        # K × Γp
        crss1 = dX2*gam3 - dX3*gam2
        crss2 = dX3*gam1 - dX1*gam3
        crss3 = dX1*gam2 - dX2*gam1

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        @inbounds UJ[1i32] += g_sgm * crss1
        @inbounds UJ[2i32] += g_sgm * crss2
        @inbounds UJ[3i32] += g_sgm * crss3

        gam1 *= g_sgm
        gam2 *= g_sgm
        gam3 *= g_sgm

        dX1 *= aux
        dX2 *= aux
        dX3 *= aux

        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        # j=1
        @inbounds UJ[4i32] += crss1 * dX1
        @inbounds UJ[5i32] += crss2 * dX1 - gam3
        @inbounds UJ[6i32] += crss3 * dX1 + gam2
        # j=2
        @inbounds UJ[7i32] += crss1 * dX2 + gam3
        @inbounds UJ[8i32] += crss2 * dX2
        @inbounds UJ[9i32] += crss3 * dX2 - gam1
        # j=3
        @inbounds UJ[10i32] += crss1 * dX3 - gam2
        @inbounds UJ[11i32] += crss2 * dX3 + gam1
        @inbounds UJ[12i32] += crss3 * dX3
    end

    return
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
function gpu_atomic_square!(out, s, t, p, q, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x

    # Row and column indices of threads in a block
    row::Int32 = (ithread-1i32) % p + 1i32
    col::Int32 = floor(Int32, (ithread-1i32)/p) + 1i32

    itarget::Int32 = row + (blockIdx().x-1i32)*p
    if itarget <= t_size
        @inbounds tx = t[1i32, itarget]
        @inbounds ty = t[2i32, itarget]
        @inbounds tz = t[3i32, itarget]
    end

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / q)

    sh_mem = CuDynamicSharedArray(eltype(t), (7, p))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    isource::Int32 = 0
    i::Int32 = 0

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        if (col == 1i32)
            isource = row + (itile-1i32)*p
            idim = 1i32
            if isource <= s_size
                while idim <= 7i32
                    @inbounds sh_mem[idim, row] = s[idim, isource]
                    idim += 1i32
                end
            else
                while idim <= 7i32
                    @inbounds sh_mem[idim, row] = zero(eltype(s))
                    idim += 1i32
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources
        # in the shared memory on the target corresponding to its index
        i = 1i32
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1i32)
            if isource <= s_size
                if itarget <= t_size
                    gpu_interaction!(UJ, tx, ty, tz, sh_mem, isource, kernel)
                end
            end
            i += 1i32
        end
        itile += 1i32
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    if itarget <= t_size
        idim = 1i32
        while idim <= 12i32
            @inbounds CUDA.@atomic out[idim, itarget] += UJ[idim]
            idim += 1i32
        end
    end
    return
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
# p - no. of targets in a block
# q - no. of threads handling a single target (should be factor of r)
# r - no. of sources in a tile
# rectangular - true: rectangular tile, false: square tile
function gpu_atomic!(out, s, t, p, q, r, rectangular, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x

    # Row and column indices of threads in a block
    row::Int32 = (ithread-1i32) % p + 1i32
    col::Int32 = floor(Int32, (ithread-1i32)/p) + 1i32

    itarget::Int32 = row + (blockIdx().x-1i32)*p
    if itarget <= t_size
        @inbounds tx = t[1i32, itarget]
        @inbounds ty = t[2i32, itarget]
        @inbounds tz = t[3i32, itarget]
    end

    n_tiles::Int32 = rectangular ? CUDA.ceil(Int32, s_size / r) : CUDA.ceil(Int32, s_size / p)

    bodies_per_col::Int32 = rectangular ? CUDA.ceil(Int32, r / q) : CUDA.ceil(Int32, p / q)

    sh_mem_size = rectangular ? r : p
    sh_mem = CuDynamicSharedArray(eltype(t), (7, sh_mem_size))

    # Variable initialization
    UJ = @MVector zeros(eltype(t), 12)
    idim::Int32 = 0
    isource::Int32 = 0
    i::Int32 = 0

    # For shared memory copying by blocks
    shblk::Int32 = 0
    shmem_idx::Int32 = 0
    n_shblks = CUDA.ceil(Int32, r / blockDim().x)

    itile::Int32 = 1
    while itile <= n_tiles
        # Each thread will copy source coordinates corresponding to its index into shared memory. This will be done for each tile.
        shblk = 1i32
        if rectangular
            while shblk <= n_shblks
                shmem_idx = ithread + (shblk-1i32)*blockDim().x
                idim = 1i32
                if shmem_idx <= r
                    isource = shmem_idx + (itile-1i32)*r
                    if isource <= s_size
                        while idim <= 7i32
                            @inbounds sh_mem[idim, shmem_idx] = s[idim, isource]
                            idim += 1i32
                        end
                    else
                        while idim <= 7i32
                            @inbounds sh_mem[idim, shmem_idx] = zero(eltype(s))
                            idim += 1i32
                        end
                    end
                end
                shblk += 1i32
            end
        else
            if (col == 1i32)
                isource = row + (itile-1i32)*p
                idim = 1i32
                if isource <= s_size
                    while idim <= 7i32
                        @inbounds sh_mem[idim, row] = s[idim, isource]
                        idim += 1i32
                    end
                else
                    while idim <= 7i32
                        @inbounds sh_mem[idim, row] = zero(eltype(s))
                        idim += 1i32
                    end
                end
            end
        end
        sync_threads()

        # Each thread will compute the influence of all the sources in the shared memory on the target corresponding to its index
        i = 1i32
        while i <= bodies_per_col
            isource = i + bodies_per_col*(col-1i32)
            if isource <= s_size
                if itarget <= t_size
                    gpu_interaction!(UJ, tx, ty, tz, sh_mem, isource, kernel)
                end
            end
            i += 1i32
        end
        itile += 1i32
        sync_threads()
    end

    # Sum up accelerations for each target/thread
    # Each target will be accessed by q no. of threads
    if itarget <= t_size
        idim = 1i32
        while idim <= 12i32
            @inbounds CUDA.@atomic out[idim, itarget] += UJ[idim]
            idim += 1i32
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
function gpu_reduction_direct!(out, s, t, num_cols, kernel)
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
                # Accumulates this source's contribution directly into UJ
                # (this thread's running sum over the tile).
                gpu_interaction!(UJ, tx, ty, tz, sh_mem, isource, kernel)
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
        idim = 1i32
        while idim <= 12i32
            @inbounds out[idim, itarget] += UJ[idim]
            idim += 1i32
        end
    end

    return
end

"""
    combine_source_indices(sorted_direct_list, source_branches::Vector{<:fmm.Branch})

Combines all the sources corresponding to a target branch.
The input sorted_direct_list has to be sorted by target
using the function fmm.sort_list_by_target().
"""
function combine_source_indices(sorted_direct_list, source_branches::Vector{<:fmm.Branch})
    # This algorithm needs to be changed to count the source indices first
    # and then allocate and fill instead of performing a push!() operation

    result = Vector{Vector{Int32}}()
    current_target = sorted_direct_list[1][1]
    current_sources = Int32[]

    # Loop through sorted_direct_list to accumulate sources corresponding to a target
    for pair in sorted_direct_list
        target, source = pair[1], pair[2]
        if target != current_target
            # Append both target and source to result
            push!(result, vcat([current_target], current_sources))
            # Reset for new target
            current_target = target
            empty!(current_sources)
        end
        push!(current_sources, source)
    end

    # Add last group
    push!(result, vcat([current_target], current_sources))

    return result
end

"""
    expand_source_indices(target_sources, source_branches)

Expands the bodies_index for all branches corresponding to a target branch.
`target_sources` contains the target branch index and source branch indices.
"""
function expand_source_indices(target_sources, source_branches)
    # Count cardinality of each branch
    branch_count = Vector{Int}(undef, length(target_sources)-1)
    for i in 2:length(target_sources)
        branch_count[i-1] = length(source_branches[target_sources[i]].bodies_index)
    end

    # Expand each branch's bodies_index into result
    expanded_indices = Vector{Int}(undef, sum(branch_count))
    i = 1
    for ibranch in 1:length(branch_count)
        expanded_indices[i:i+branch_count[ibranch]-1] .= source_branches[target_sources[ibranch+1]].bodies_index
        i += branch_count[ibranch]
    end
    return expanded_indices
end

# Checks the interaction list to see if it's a direct interaction only case
function is_fully_direct(target_sources)::Bool
    for i in 1:length(target_sources)
        first_element = target_sources[i][1]
        for j in 1:length(target_sources)
            if first_element != target_sources[j][i+1]
                return false
            end
        end
    end
    return true
end

"""
    FLOWVPM.gpu_direct!(pfield::ParticleField)

CUDA implementation of the direct (no-FMM) O(N²) N-body sum: overloads the
stub declared in `FLOWVPM_UJ.jl`, dispatched to from `UJ_direct` whenever
`pfield.particles isa CuArray`. Bypasses FastMultipole's own `direct!`
entirely (that path is CPU-only -- see `FastMultipole/src/direct.jl`; its
only GPU hook, `nearfield_device!`, is tree/FMM-based and belongs to the
separate, still-in-progress FMM-GPU effort). Uses the `gpu_atomic_square!`
tile kernel above with a square launch config sized for a self-interacting
target/source array of `n = pfield.np` particles.

NOTE: unlike Phase 1's broadcast rewrites, this has not been run against
real GPU hardware (none available in this environment) -- verified by code
review against `gpu_interaction!`'s per-particle math and the CPU reference
in `FLOWVPM_fmm.jl`'s `fmm.direct!` overload only. Treat as unverified until
run on the supercomputer.
"""
function FLOWVPM.gpu_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                             ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:CuArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    # Source/target array: rows 1:7 are exactly [X (1:3); Gamma (4:6); sigma
    # (7)], matching gpu_interaction!'s hardcoded row indices 1i32:7i32 --
    # see FLOWVPM.X_INDEX/GAMMA_INDEX/SIGMA_INDEX in FLOWVPM_particlefield.jl.
    s = view(P, 1:7, 1:n)

    out = CUDA.zeros(T, 12, n)

    p, q, _ = get_launch_config(n)
    nthreads = p*q
    nblocks = cld(n, p)
    shmem = 7*p*sizeof(T)

    check_shared_memory(CUDA.device(), shmem)

    @cuda threads=nthreads blocks=nblocks shmem=shmem gpu_atomic_square!(out, s, s, p, q, pfield.kernel.g_dgdr)

    # out[1:3,:] = U, out[4:12,:] = J (9-entry flattened Jacobian) -- same
    # row order gpu_interaction! writes as the CPU direct!/set_gradient! path
    # uses for FLOWVPM.U_INDEX/J_INDEX (see FLOWVPM_fmm.jl).
    view(P, FLOWVPM.U_INDEX, 1:n) .+= view(out, 1:3, :)
    view(P, FLOWVPM.J_INDEX, 1:n) .+= view(out, 4:12, :)

    return nothing
end

# Each thread handles a single target and brute-force loops over every
# source directly from global memory (no shared-memory tiling). Chosen over
# a tiled kernel (like gpu_atomic_square! above) for auditability given no
# local GPU hardware to iterate against -- see gpu_zeta_direct!/gpu_estr_direct!.
@inline function gpu_zeta_direct_kernel!(out, s, n::Int32, zeta)
    j_target::Int32 = (blockIdx().x-1i32)*blockDim().x + threadIdx().x
    if j_target <= n
        @inbounds tx = s[1i32, j_target]
        @inbounds ty = s[2i32, j_target]
        @inbounds tz = s[3i32, j_target]

        T = eltype(s)
        acc1, acc2, acc3 = zero(T), zero(T), zero(T)

        i::Int32 = 1
        while i <= n
            @inbounds dX1 = tx - s[1i32, i]
            @inbounds dX2 = ty - s[2i32, i]
            @inbounds dX3 = tz - s[3i32, i]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            @inbounds sigma = s[7i32, i]
            zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)

            @inbounds acc1 += s[4i32, i]*zeta_sgm
            @inbounds acc2 += s[5i32, i]*zeta_sgm
            @inbounds acc3 += s[6i32, i]*zeta_sgm

            i += 1i32
        end

        @inbounds out[1i32, j_target] += acc1
        @inbounds out[2i32, j_target] += acc2
        @inbounds out[3i32, j_target] += acc3
    end
    return
end

"""
    FLOWVPM.gpu_zeta_direct!(pfield::ParticleField)

CUDA implementation of `zeta_direct`'s O(N²) direct-sum basis-function
evaluation, overloading the stub declared in `FLOWVPM_viscous.jl`. Unlike
most direct-sum call sites, `zeta_direct` includes ALL particles (even
static ones) as both source and target -- matching the CPU version's
`iterator(pfield; include_static=true)` on both sides -- so no active-particle
masking is applied here.

NOTE: unverified against real GPU hardware, same caveat as `gpu_direct!` above.
"""
function FLOWVPM.gpu_zeta_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:CuArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    # rows 1:7 = [X (1:3); Gamma (4:6); sigma (7)], same layout as gpu_direct!
    s = view(P, 1:7, 1:n)

    out = CUDA.zeros(T, 3, n)

    nthreads = min(n, Int(default_max_threads_per_block))
    nblocks = cld(n, nthreads)

    @cuda threads=nthreads blocks=nblocks gpu_zeta_direct_kernel!(out, s, Int32(n), pfield.kernel.zeta)

    # CPU `zeta_direct` zeroes J[1:3] before accumulating (over ALL
    # particles, per the include_static=true above), so this is an
    # assignment, not an accumulation, to match.
    view(P, FLOWVPM.J_INDEX[1:3], 1:n) .= view(out, 1:3, :)

    return nothing
end

# Each thread handles a single target and brute-force loops over every
# source directly from global memory. Same auditability-over-perf tradeoff
# as gpu_zeta_direct_kernel! above.
@inline function gpu_estr_direct_kernel!(sfs_out, P, n::Int32, zeta, transposed::Bool,
                                          static_row::Int32, j1::Int32, j2::Int32, j3::Int32,
                                          j4::Int32, j5::Int32, j6::Int32, j7::Int32, j8::Int32, j9::Int32)
    j_target::Int32 = (blockIdx().x-1i32)*blockDim().x + threadIdx().x
    T = eltype(P)
    if j_target <= n
        @inbounds target_is_static = P[static_row, j_target]
        if target_is_static == 0
            @inbounds tx = P[1i32, j_target]
            @inbounds ty = P[2i32, j_target]
            @inbounds tz = P[3i32, j_target]
            @inbounds JT1 = P[j1, j_target]; @inbounds JT2 = P[j2, j_target]; @inbounds JT3 = P[j3, j_target]
            @inbounds JT4 = P[j4, j_target]; @inbounds JT5 = P[j5, j_target]; @inbounds JT6 = P[j6, j_target]
            @inbounds JT7 = P[j7, j_target]; @inbounds JT8 = P[j8, j_target]; @inbounds JT9 = P[j9, j_target]

            acc1, acc2, acc3 = zero(T), zero(T), zero(T)

            i::Int32 = 1
            while i <= n
                @inbounds source_is_static = P[static_row, i]
                if source_is_static == 0
                    @inbounds sx = P[1i32, i]
                    @inbounds sy = P[2i32, i]
                    @inbounds sz = P[3i32, i]
                    dX1 = tx - sx
                    dX2 = ty - sy
                    dX3 = tz - sz
                    r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

                    @inbounds sigma = P[7i32, i]
                    zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)

                    @inbounds GS1 = P[4i32, i]; @inbounds GS2 = P[5i32, i]; @inbounds GS3 = P[6i32, i]
                    @inbounds JS1 = P[j1, i]; @inbounds JS2 = P[j2, i]; @inbounds JS3 = P[j3, i]
                    @inbounds JS4 = P[j4, i]; @inbounds JS5 = P[j5, i]; @inbounds JS6 = P[j6, i]
                    @inbounds JS7 = P[j7, i]; @inbounds JS8 = P[j8, i]; @inbounds JS9 = P[j9, i]

                    if transposed
                        S1 = (JT1-JS1)*GS1 + (JT2-JS2)*GS2 + (JT3-JS3)*GS3
                        S2 = (JT4-JS4)*GS1 + (JT5-JS5)*GS2 + (JT6-JS6)*GS3
                        S3 = (JT7-JS7)*GS1 + (JT8-JS8)*GS2 + (JT9-JS9)*GS3
                    else
                        S1 = (JT1-JS1)*GS1 + (JT4-JS4)*GS2 + (JT7-JS7)*GS3
                        S2 = (JT2-JS2)*GS1 + (JT5-JS5)*GS2 + (JT8-JS8)*GS3
                        S3 = (JT3-JS3)*GS1 + (JT6-JS6)*GS2 + (JT9-JS9)*GS3
                    end

                    acc1 += zeta_sgm*S1
                    acc2 += zeta_sgm*S2
                    acc3 += zeta_sgm*S3
                end
                i += 1i32
            end

            @inbounds sfs_out[1i32, j_target] += acc1
            @inbounds sfs_out[2i32, j_target] += acc2
            @inbounds sfs_out[3i32, j_target] += acc3
        end
    end
    return
end

"""
    FLOWVPM.gpu_estr_direct!(pfield::ParticleField)

CUDA implementation of `Estr_direct!`'s O(N²) direct-sum SFS
vortex-stretching contribution, overloading the stub declared in
`FLOWVPM_subfilterscale_models.jl`. Both source and target loops skip static
particles (matching `Estr_direct_singlethreaded`/`_multithreaded`'s use of
the default `iterator(pfield)`, which excludes them), and results are
accumulated (`+=`) into `SFS_INDEX`, matching the CPU version, which never
resets SFS itself (that's done separately via `_reset_particles_sfs`, gated
by the `reset_sfs` kwarg upstream in `UJ_direct`/`UJ_fmm`).

NOTE: unverified against real GPU hardware, same caveat as `gpu_direct!`
above. Row indices for `STATIC_INDEX`/`J_INDEX` are passed in as scalar
kernel arguments (rather than hardcoded like `gpu_direct!`'s 1i32:12i32)
since `J_INDEX` isn't contiguous-from-1 the way the U/J/X/Gamma/sigma block
is -- keeps the kernel body itself free of magic numbers beyond X/Gamma/sigma
(rows 1:7, same layout as gpu_direct!/gpu_zeta_direct!).
"""
function FLOWVPM.gpu_estr_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:CuArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    out = CUDA.zeros(T, 3, n)

    nthreads = min(n, Int(default_max_threads_per_block))
    nblocks = cld(n, nthreads)

    jrows = Int32.(FLOWVPM.J_INDEX)

    @cuda threads=nthreads blocks=nblocks gpu_estr_direct_kernel!(
        out, P, Int32(n), pfield.kernel.zeta, pfield.transposed,
        Int32(FLOWVPM.STATIC_INDEX),
        jrows[1], jrows[2], jrows[3], jrows[4], jrows[5], jrows[6], jrows[7], jrows[8], jrows[9])

    view(P, FLOWVPM.SFS_INDEX, 1:n) .+= view(out, 1:3, :)

    return nothing
end

# Convenience function to compile the GPU kernel
# so compilation doesn't take time later
# NOTE: THIS DOES NOT WORK WITH THE NEW nearfield_device!() FUNCTION
# SINCE THAT REQUIRES BRANCHES AND TREE DATA STRUCTURES
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
        # This needs to be corrected
        fmm.nearfield_device!(pfield, target_indices, d_switch, pfield, source_indices)

        verbose && @info("CUDA kernel compiled successfully on $ngpu device/s")
    end

    return
end

################################################################################
# DEVICE-RESIDENT RADIX FMM HOOKS (task 034)
#
# Bulk device pack/unpack hooks for FastMultipole's device-resident radix
# lifecycle (`src/FLOWVPM_fmm_radix.jl` holds the rest of the coupling: traits,
# cache registry, recenter policy, `UJ_fmm_gpu!`). These two methods are the
# only CUDA-typed pieces: they read/write the live prefix of the 46xN CuArray
# particle matrix directly against FastMultipole's persistent device buffers,
# so a full evaluation performs zero host/device body transfer (task 023
# counter contract: body_uploads == 0, expansion_host_copies == 0).
#
# Guarded on the radix interface being present so the extension still loads
# (direct-sum kernels only) against a registry FastMultipole.
################################################################################

if FLOWVPM._FMM_HAS_RADIX

# Framework-owned persistent device source buffer, passed as an 8 x np view
# (live prefix; identity sort index). Packed layout (integration-api-spec §3):
#   rows 1:3  position            (X_INDEX)
#   row  4    MAC/error radius    rho_sigma * sigma (autotuning off, so
#                                 rho_sigma = pfield.fmm.default_rho_over_sigma
#                                 — the host `source_system_to_buffer!` value)
#   rows 5:7  vector strength     (GAMMA_INDEX)
#   row  8    raw smoothing sigma (SIGMA_INDEX; read by RegularizedVortex)
# Steady-state allocation-free: broadcasts into the existing buffer only.
function fmm.source_to_buffer!(buf::CUDA.AnyCuArray,
        pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT},
        sort_index) where {R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT<:CuArray{R}}
    np = pfield.np
    (first(sort_index) == 1 && last(sort_index) == np) || error(
        "FLOWVPM device source_to_buffer! expects the identity sort index over " *
        "the live particle prefix (got $(first(sort_index)):$(last(sort_index)) for np=$np)")
    size(buf, 1) >= 8 && size(buf, 2) == np || error(
        "unexpected device source buffer shape $(size(buf)) for np=$np")
    P = pfield.particles
    rho_sigma = R(pfield.fmm.default_rho_over_sigma)
    view(buf, 1:3, :) .= view(P, FLOWVPM.X_INDEX, 1:np)
    view(buf, 4, :) .= rho_sigma .* view(P, FLOWVPM.SIGMA_INDEX, 1:np)
    view(buf, 5:7, :) .= view(P, FLOWVPM.GAMMA_INDEX, 1:np)
    view(buf, 8, :) .= view(P, FLOWVPM.SIGMA_INDEX, 1:np)
    return buf
end

# Framework-owned per-system device output buffer, switch-relative rows, in
# global (unsorted) particle order. ACCUMULATE (.+=): FLOWVPM zeroes U/J via
# `_reset_particles` at the top of each UJ evaluation and the framework
# delivers the total influence of the evaluation (delivery semantics,
# docs/src/device_interface.md). Accumulating into all live particles
# (including static ones) matches the legacy `buffer_to_target_system!`.
function fmm.buffer_to_target!(
        pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT},
        buf::CUDA.AnyCuArray, derivatives_switch,
        sort_index) where {R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT<:CuArray{R}}
    np = pfield.np
    size(buf, 2) == np || error(
        "unexpected device output buffer shape $(size(buf)) for np=$np")
    P = pfield.particles
    grange = fmm.gradient_range(derivatives_switch)
    isempty(grange) ||
        (view(P, FLOWVPM.U_INDEX, 1:np) .+= view(buf, grange, :))
    hrange = fmm.hessian_range(derivatives_switch)
    isempty(hrange) ||
        (view(P, FLOWVPM.J_INDEX, 1:np) .+= view(buf, hrange, :))
    return pfield
end

end # FLOWVPM._FMM_HAS_RADIX

end # module FLOWVPMCUDAExt
