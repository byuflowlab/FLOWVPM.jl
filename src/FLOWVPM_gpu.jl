# Contains utilities for handling gpu kernel
function check_launch(n, p, q; T=Float32, throw_error=true)
    max_threads_per_block = T==Float32 ? 1024 : 256

    isgood = true

    if p > n; isgood = false; throw_error && error("p must be less than or equal to n"); end
    if p*q >= max_threads_per_block; isgood = false; throw_error && error("p*q must be less than $max_threads_per_block"); end
    if q > p; isgood = false; throw_error && error("q must be less than or equal to p"); end
    if n % p != 0; isgood = false; throw_error && error("n must be divisible by p"); end
    if p % q != 0; isgood = false; throw_error && error("p must be divisible by q"); end

    return isgood
end

function get_launch_config(nt; T=Float32, p_max=0, q_max=0)
    max_threads_per_block = T==Float32 ? 1024 : 256
    p_max = (p_max == 0) ? max_threads_per_block : p_max
    q_max = (p_max == 0) ? p_max : q_max

    divs_n = divisors(nt)
    p = 1
    q = 1
    ip = 1
    iq = 1
    for (i, div) in enumerate(divs_n)
        if div <= p_max
            p = div
            ip = i
        end
    end

    # Decision algorithm 1: Creates a matrix using indices and finds max of
    # weighted sum of indices

    i_weight = 0
    j_weight = 1-i_weight

    max_ij = i_weight*ip + j_weight*iq
    if nt <= 2^13
        divs_p = divs_n
        for i in 1:length(divs_n)
            for j in 1:length(divs_n)
                isgood = check_launch(nt, divs_n[i], divs_p[j]; T=T, throw_error=false)
                if isgood && (divs_n[i] <= p_max)
                    # Check if this is the max achievable ij value
                    # in the p, q choice matrix
                    obj_val = i_weight*i+j_weight*j
                    if (obj_val >= max_ij) && (divs_p[j] <= q_max)
                        max_ij = obj_val
                        p = divs_n[i]
                        q = divs_p[j]
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
        UJ[1] = g_sgm * crss1
        UJ[2] = g_sgm * crss2
        UJ[3] = g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r2
        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux2 = -T(const4) * g_sgm / r3
        # j=1
        UJ[4] = aux * crss1 * dX1
        UJ[5] = aux * crss2 * dX1 - aux2 * gam3
        UJ[6] = aux * crss3 * dX1 + aux2 * gam2
        # j=2
        UJ[7] = aux * crss1 * dX2 + aux2 * gam3
        UJ[8] = aux * crss2 * dX2
        UJ[9] = aux * crss3 * dX2 - aux2 * gam1
        # j=3
        UJ[10] = aux * crss1 * dX3 - aux2 * gam2
        UJ[11] = aux * crss2 * dX3 + aux2 * gam1
        UJ[12] = aux * crss3 * dX3
    end

    return UJ
end

# Each thread handles a single target and uses local GPU memory
# Sources divided into multiple columns and influence is computed by multiple threads
function gpu_direct!(s, t, num_cols, kernel)
    t_size::Int32 = size(t, 2)
    s_size::Int32 = size(s, 2)

    ithread::Int32 = threadIdx().x
    p::Int32 = t_size/gridDim().x

    # Row and column indices of threads in a block
    row = (ithread-1) % p + 1
    col = floor(Int32, (ithread-1)/p) + 1

    itarget::Int32 = row + (blockIdx().x-1)*p
    @inbounds tx = t[1, itarget]
    @inbounds ty = t[2, itarget]
    @inbounds tz = t[3, itarget]

    n_tiles::Int32 = CUDA.ceil(Int32, s_size / p)
    bodies_per_col::Int32 = CUDA.ceil(Int32, p / num_cols)

    # 12 for UJ variables that have to be reduced at the end
    sh_mem = CuDynamicSharedArray(eltype(t), (7, p))
    # sh_mem = CuDynamicSharedArray(eltype(t), (12*p, p))

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
                out = gpu_interaction(tx, ty, tz, sh_mem, isource, kernel)

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
    if num_cols != 1
        # Perform write to shared memory
        # Columns correspond to each of the q threads
        # sh_mem[1:12, 1] is the first target, sh_mem[13:24, 1] is the second target and so on.
        idim = 1
        while idim <= 12
            @inbounds sh_mem[idim + 12*(itarget-1), col] = UJ[idim]
            idim += 1
        end

        sync_threads()

        # Write data from shared mem to global mem (sum using single thread for now)
        if col == 1
            isource = 1
            while isource <= num_cols
                idim = 1
                while idim <= 3
                    @inbounds t[9+idim, itarget] += sh_mem[idim+12*(itarget-1), isource]
                    idim += 1
                end
                idim = 4
                while idim <= 12
                    @inbounds t[12+idim, itarget] += sh_mem[idim+12*(itarget-1), isource]
                    idim += 1
                end
                isource += 1
            end
        end
    else
        idim = 1
        while idim <= 3
            @inbounds t[9+idim, itarget] += UJ[idim]
            idim += 1
        end
        idim = 4
        while idim <= 12
            @inbounds t[12+idim, itarget] += UJ[idim]
            idim += 1
        end
    end
    return
end

# Convenience function to compile the GPU kernel
# so compilation doesn't take time later
function init_GPU(verbose=false)
    n = 100
    # Create particle field
    pfield = ParticleField(n; useGPU=true)

    # Set no. of dummy particles
    pfield.np = 100

    # Run direct computation on particles
    d_switch = FastMultipole.DerivativesSwitch()
    fmm.direct!(pfield, 1:n, d_switch, pfield, 1:n)

    if verbose
        @info("CUDA kernel compiled successfully")
    end
    return
end
