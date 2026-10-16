#=##############################################################################
# DESCRIPTION
    Galerkin (L2-projection) core-size reset: `galerkin_shrink!`, the reset of
    `CoreSpreading`.

    The reset particles (the subset, or every particle in a full reset) go to
    `sgm0`; their non-reset neighbours keep their own cores and take strength
    increments (the halo, which carries the old tails); new `sgm0` points are
    added where the old tails reach past the last particle (edge points). The
    strengths minimise the L2 error of the Gaussian vorticity field:

        (M + mu diag M) c = f,   M_ij = N(x_i - x_j; s_i^2 + s_j^2),
                                 f_i  = sum_k Gold_k N(x_i - x_k; s_i^2 + so_k^2),

    N the normalized 3D Gaussian (FLOWVPM's gaussianerf zeta), the overlaps cut
    at r^2 <= gram_cut^2 (s_i^2 + s_j^2). Study and parameters: research-notes
    flowunsteady-update/details/2026-10-14_rbf-reset-fix_1.md and
    2026-10-15_cs-galerkin_1.md.

    Two parts: the selection (`_gal_plan`: thinning, halo, edge points; indices
    only, no derivative) and the linear solve on that fixed geometry
    (`_gal_solve`, block-Jacobi preconditioned CG; reverse-mode AD through it
    uses the implicit rule in ext/FLOWVPMEnzymeExt.jl). Host fields evaluate M
    with each pair once (`_GalOp`), storing the pairs of as many rows as fit in
    `gram_mem` and the rest on the fly; device fields evaluate it on the fly
    with the pair kernel (`_gal_row`, KA kernel in the GPU extension).
=###############################################################################

const _GAL_C3 = (2pi)^-1.5

# Cell-sorted point set: positions `x` (3 x n) and variances `v` sorted by cell
# (cell size `h`, box corner `lo`, `n` cells per axis); cell k (1-based linear id)
# holds cstart[k]:cstart[k+1]-1. `vmax` bounds the variances.
struct _GalSet{TX,TV,TI,R}
    x::TX; v::TV; cstart::TI
    n::NTuple{3,Int32}; lo::NTuple{3,R}; h::R; vmax::R
end

# Cell size h, enlarged until the dense cell table has at most 2^26 cells (no
# generator over h: it would box h and leave the set's type uninferred)
function _gal_cellsize(ext, h)
    while (floor(Int, ext[1] / h) + 1) * (floor(Int, ext[2] / h) + 1) * (floor(Int, ext[3] / h) + 1) > 2^26
        h *= 1.25
    end
    return h
end

# Sort (X, v) into cells of size h; returns the set and the sort permutation.
function _gal_set(X::Matrix{Float64}, v::Vector{Float64}, h0::Float64)
    N = size(X, 2)
    lo = ntuple(d -> minimum(view(X, d, :)) - 1e-6 * h0, 3)
    ext = ntuple(d -> maximum(view(X, d, :)) - lo[d], 3)
    h = _gal_cellsize(ext, h0)
    n = ntuple(d -> Int32(floor(Int, ext[d] / h) + 1), 3)
    cell = Vector{Int}(undef, N)
    for i in 1:N
        cell[i] = 1 + floor(Int, (X[1, i] - lo[1]) / h) +
                  n[1] * (floor(Int, (X[2, i] - lo[2]) / h) + n[2] * floor(Int, (X[3, i] - lo[3]) / h))
    end
    nc = Int(n[1]) * n[2] * n[3]
    cstart = zeros(Int32, nc + 1); cstart[1] = 1
    for i in 1:N; cstart[cell[i] + 1] += 1; end
    cumsum!(cstart, cstart)
    perm = sortperm(cell)
    return _GalSet(X[:, perm], v[perm], cstart, n, lo, h, maximum(v)), perm
end

# Gaussian pair sum at target i: sum_p c[:, p] N(t_i - s_p; tv_i + sv_p) over the
# sources within the cut (cut2 = gram_cut^2). Shared by the host loops and the
# device kernel (ext/FLOWVPMGPUExt.jl), so it uses only indexing and arithmetic.
@inline function _gal_row(tx, tv, i, sx, sv, c, cstart, n::NTuple{3,Int32}, lo, h, cut2, vmax)
    @inbounds begin
        x1 = tx[1, i]; x2 = tx[2, i]; x3 = tx[3, i]; vi = tv[i]
        T = typeof(vi)
        o1 = zero(T); o2 = zero(T); o3 = zero(T)
        m = unsafe_trunc(Int32, ceil(sqrt(cut2 * (vi + vmax)) / h))
        a = unsafe_trunc(Int32, floor((x1 - lo[1]) / h))
        b = unsafe_trunc(Int32, floor((x2 - lo[2]) / h))
        cc = unsafe_trunc(Int32, floor((x3 - lo[3]) / h))
        a0 = max(a - m, Int32(0)); a1 = min(a + m, n[1] - Int32(1))
        C3 = T(_GAL_C3); half = T(0.5)
        if a0 <= a1
            for kc in max(cc - m, Int32(0)):min(cc + m, n[3] - Int32(1))
                for kb in max(b - m, Int32(0)):min(b + m, n[2] - Int32(1))
                    k0 = Int32(1) + a0 + n[1] * (kb + n[2] * kc)
                    for p in cstart[k0]:cstart[k0 + a1 - a0 + Int32(1)] - Int32(1)
                        d1 = sx[1, p] - x1; d2 = sx[2, p] - x2; d3 = sx[3, p] - x3
                        r2 = d1 * d1 + d2 * d2 + d3 * d3
                        vv = vi + sv[p]
                        if r2 <= cut2 * vv
                            w = C3 * exp(-half * r2 / vv) / (vv * sqrt(vv))
                            o1 += w * c[1, p]; o2 += w * c[2, p]; o3 += w * c[3, p]
                        end
                    end
                end
            end
        end
    end
    return o1, o2, o3
end

# out[:, i] = pair sum at targets (tx, tv) over the set S with strengths c (host;
# the device method is in the GPU extension)
function _gal_pairsum!(out::Matrix, tx::Matrix, tv::Vector, S::_GalSet, c::Matrix, cut2)
    Threads.@threads for i in 1:size(tx, 2)
        o1, o2, o3 = _gal_row(tx, tv, i, S.x, S.v, c, S.cstart, S.n, S.lo, S.h, cut2, S.vmax)
        out[1, i] = o1; out[2, i] = o2; out[3, i] = o3
    end
    return out
end

# Host copies stay Float64; a device field gets its own precision and backend.
_gal_to(::Matrix, A::Array{Float64}) = A
_gal_to(::Matrix, A::Array{Int32}) = A
_gal_to(P, A::Array{Int32}) = copyto!(similar(P, Int32, size(A)), A)
_gal_to(P, A::Array{Float64}) = copyto!(similar(P, eltype(P), size(A)), convert(Array{eltype(P)}, A))
_gal_to(P::Matrix, S::_GalSet) = S
function _gal_to(P, S::_GalSet)
    T = eltype(P)
    return _GalSet(_gal_to(P, S.x), _gal_to(P, S.v), _gal_to(P, S.cstart), S.n, T.(S.lo), T(S.h), T(S.vmax))
end
# Release device copies as soon as a phase is done (GPU extension; host arrays are
# left to the GC)
_gal_free!(xs...) = foreach(_gal_free1!, xs)
_gal_free1!(a) = nothing
_gal_free1!(S::_GalSet) = _gal_free!(S.x, S.v, S.cstart)
_gal_free1!(t::NamedTuple) = foreach(_gal_free1!, t)

# Host Gram operator. Each pair once, from the row of the wider core (ties: the
# larger index), so row i searches r <= gram_cut sqrt(2 v_i) only; the transposed
# half goes to per-thread buffers. Rows 1:R keep their pairs (Int32 column +
# Float32 value, 8 bytes a pair; R as many rows as fit in `gram_mem`), the other
# rows are evaluated on the fly. Buffers: 40 bytes per unknown and thread.
# Storage reused across resets (grown on demand): the stored Gram pairs
struct _GalWork
    col::Vector{Int32}; val::Vector{Float32}
end
_GalWork() = _GalWork(Int32[], Float32[])

struct _GalOp
    R::Int; rp::Vector{Int}; col::Vector{Int32}; val::Vector{Float32}
    y::Vector{Matrix{Float64}}; ib::Vector{Vector{Int32}}; rb::Vector{Vector{Float64}}
end

# Row i's own pairs (candidates of each cell run compacted branch-free, then
# weighted): y === nothing counts them, or stores them from col[p] when col is
# not empty; else adds the row's products to y (diagonal included).
function _gal_halfrow!(y, x, S::_GalSet, i, cut2, ib, rb, col, val, p)
    @inbounds begin
        x1 = S.x[1, i]; x2 = S.x[2, i]; x3 = S.x[3, i]; vi = S.v[i]
        c1 = 0.0; c2 = 0.0; c3 = 0.0; o1 = 0.0; o2 = 0.0; o3 = 0.0
        if y !== nothing
            c1 = x[1, i]; c2 = x[2, i]; c3 = x[3, i]
            w = _GAL_C3 / (2vi * sqrt(2vi)); o1 = w * c1; o2 = w * c2; o3 = w * c3
        end
        m = ceil(Int, sqrt(2cut2 * vi) / S.h)
        a = floor(Int, (x1 - S.lo[1]) / S.h); b = floor(Int, (x2 - S.lo[2]) / S.h); c = floor(Int, (x3 - S.lo[3]) / S.h)
        a0 = max(a - m, 0); a1 = min(a + m, S.n[1] - 1)
        cnt = 0
        for kc in max(c - m, 0):min(c + m, S.n[3] - 1), kb in max(b - m, 0):min(b + m, S.n[2] - 1)
            k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
            n = 0
            for q in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
                vq = S.v[q]
                d1 = S.x[1, q] - x1; d2 = S.x[2, q] - x2; d3 = S.x[3, q] - x3
                r2 = d1 * d1 + d2 * d2 + d3 * d3
                ib[n + 1] = q; rb[n + 1] = r2
                n += (r2 <= cut2 * (vi + vq)) & ((vq < vi) | ((vq == vi) & (q < i)))
            end
            if y === nothing
                if !isempty(col)
                    for k in 1:n
                        vv = vi + S.v[ib[k]]
                        col[p + cnt + k - 1] = ib[k]; val[p + cnt + k - 1] = _GAL_C3 * exp(-0.5 * rb[k] / vv) / (vv * sqrt(vv))
                    end
                end
            else
                for k in 1:n
                    q = ib[k]; vv = vi + S.v[q]
                    w = _GAL_C3 * exp(-0.5 * rb[k] / vv) / (vv * sqrt(vv))
                    o1 += w * x[1, q]; o2 += w * x[2, q]; o3 += w * x[3, q]
                    y[1, q] += w * c1; y[2, q] += w * c2; y[3, q] += w * c3
                end
            end
            cnt += n
        end
        if y !== nothing
            y[1, i] += o1; y[2, i] += o2; y[3, i] += o3
        end
        return cnt
    end
end

function _GalOp(S::_GalSet, cut2, maxbytes, ws::_GalWork)
    N = size(S.x, 2); nt = Threads.nthreads()
    ib = [zeros(Int32, N) for _ in 1:nt]; rb = [zeros(N) for _ in 1:nt]
    cnt = zeros(Int, N); col = ws.col; val = ws.val
    empt = Int32[]; emptv = Float32[]
    Threads.@threads :static for t in 1:nt
        for i in t:nt:N
            cnt[i] = _gal_halfrow!(nothing, nothing, S, i, cut2, ib[t], rb[t], empt, emptv, 0)
        end
    end
    rp = cumsum(vcat(1, cnt))
    R = searchsortedlast(rp, maxbytes ÷ 8 + 1) - 1
    resize!(col, rp[R + 1] - 1); resize!(val, rp[R + 1] - 1)
    Threads.@threads :static for t in 1:nt
        for i in t:nt:R
            _gal_halfrow!(nothing, nothing, S, i, cut2, ib[t], rb[t], col, val, rp[i])
        end
    end
    return _GalOp(R, rp, col, val, [zeros(3, N) for _ in 1:nt], ib, rb), rp[end] - 1
end

function _gal_gram!(y, A::_GalOp, S, x, cut2)
    N = size(x, 2); nt = length(A.y); nb = cld(N, 256)
    Threads.@threads :static for t in 1:nt          # blocks of 256 rows, round robin
        yt = A.y[t]; fill!(yt, 0)
        @inbounds for k in t:nt:nb, i in 256(k - 1) + 1:min(256k, N)
            if i <= A.R
                c1 = x[1, i]; c2 = x[2, i]; c3 = x[3, i]
                w = _GAL_C3 / (2S.v[i] * sqrt(2S.v[i])); o1 = w * c1; o2 = w * c2; o3 = w * c3
                for p in A.rp[i]:A.rp[i + 1] - 1
                    q = A.col[p]; w = Float64(A.val[p])
                    o1 += w * x[1, q]; o2 += w * x[2, q]; o3 += w * x[3, q]
                    yt[1, q] += w * c1; yt[2, q] += w * c2; yt[3, q] += w * c3
                end
                yt[1, i] += o1; yt[2, i] += o2; yt[3, i] += o3
            else
                _gal_halfrow!(yt, x, S, i, cut2, A.ib[t], A.rb[t], A.col, A.val, 0)
            end
        end
    end
    Threads.@threads for j in 1:N
        o1 = 0.0; o2 = 0.0; o3 = 0.0
        for t in 1:nt
            o1 += A.y[t][1, j]; o2 += A.y[t][2, j]; o3 += A.y[t][3, j]
        end
        y[1, j] = o1; y[2, j] = o2; y[3, j] = o3
    end
    return y
end
_gal_gram!(y, ::Nothing, S, x, cut2) = _gal_pairsum!(y, S.x, S.v, S, x, cut2)   # device

# split `ids` (local indices into the columns of X) into octants of the cell at
# `lo` with edge `h` until each piece holds at most `cap`; coincident points (a
# piece that never splits) are cut into `cap`-sized runs
function _cs_split!(blocks, X, ids, lo, h, cap, depth=0)
    if length(ids) <= cap
        push!(blocks, ids); return blocks
    end
    if depth > 40
        for c in Iterators.partition(ids, cap); push!(blocks, collect(c)); end
        return blocks
    end
    hh = h / 2
    oct = [Int(X[1, k] - lo[1] >= hh) + 2 * Int(X[2, k] - lo[2] >= hh) + 4 * Int(X[3, k] - lo[3] >= hh) for k in ids]
    for o in 0:7
        sub = ids[oct .== o]
        isempty(sub) || _cs_split!(blocks, X, sub, lo .+ hh .* (o & 1, (o >> 1) & 1, (o >> 2) & 1), hh, cap, depth + 1)
    end
    return blocks
end

# Z = M^-1 R (host): one small dense product per block; `perm` the unknowns block
# by block, `start` each block's first position (and one past the end), `moff`
# each block's first entry in `minv` (the block inverses, column-major)
function _cs_block_apply!(Z::AbstractMatrix, R::AbstractMatrix, blk)
    Threads.@threads for b in 1:blk.nblocks
        s = Int(blk.start[b]); n = Int(blk.start[b + 1]) - s; m0 = blk.moff[b]
        for q in 1:n
            z1 = zero(eltype(Z)); z2 = zero(eltype(Z)); z3 = zero(eltype(Z))
            for k in 1:n
                m = blk.minv[m0 + (q - 1) + (k - 1) * n]; j = blk.perm[s + k - 1]
                z1 += m * R[1, j]; z2 += m * R[2, j]; z3 += m * R[3, j]
            end
            i = blk.perm[s + q - 1]
            Z[1, i] = z1; Z[2, i] = z2; Z[3, i] = z3
        end
    end
    return Z
end

# Block-Jacobi preconditioner of M + D: the unknowns grouped on a grid of cell h,
# crowded cells split into octants down to `cap` (`_cs_split!`), each block's
# dense matrix inverted once per reset (perm = positions in the set, `block_of`
# each position's block for the device kernel), applied by `_cs_block_apply!`
# (device: GPU extension).
function _gal_blocks(S::_GalSet, d::Vector{Float64}, cut2, h, cap, ::Type{TF}) where TF
    N = size(S.x, 2)
    keys = [(floor(Int, S.x[1, k] / h), floor(Int, S.x[2, k] / h), floor(Int, S.x[3, k] / h)) for k in 1:N]
    order = sortperm(keys)
    blocks = Vector{Vector{Int}}()
    i = 1
    while i <= N
        j = i
        while j < N && keys[order[j + 1]] == keys[order[i]]; j += 1; end
        _cs_split!(blocks, S.x, order[i:j], Float64.(keys[order[i]]) .* h, h, cap)
        i = j + 1
    end
    nb = length(blocks)
    start = Vector{Int32}(undef, nb + 1); moff = Vector{Int}(undef, nb + 1)
    perm = reduce(vcat, blocks); block_of = Vector{Int32}(undef, N)
    start[1] = 1; moff[1] = 1
    for (b, blk) in enumerate(blocks)
        n = length(blk)
        start[b + 1] = start[b] + n; moff[b + 1] = moff[b] + n * n
        block_of[start[b]:start[b + 1] - 1] .= b
    end
    minv = Vector{TF}(undef, moff[end] - 1)
    # each block factored and inverted in place in a per-thread buffer (LAPACK
    # potrf + potri, i.e. cholesky + inv); not positive: the diagonal's inverse
    nt = Threads.nthreads(); nmax = maximum(length, blocks; init = 0)
    W = [Matrix{Float64}(undef, nmax, nmax) for _ in 1:nt]
    Threads.@threads :static for t in 1:nt
        for b in t:nt:nb
            blk = blocks[b]; n = length(blk); A = view(W[t], 1:n, 1:n)
            for (j, c) in enumerate(blk), (i, a) in enumerate(blk)
                vv = S.v[a] + S.v[c]; r2 = (S.x[1, a] - S.x[1, c])^2 + (S.x[2, a] - S.x[2, c])^2 + (S.x[3, a] - S.x[3, c])^2
                A[i, j] = (r2 <= cut2 * vv) * _GAL_C3 * exp(-0.5 * r2 / vv) / (vv * sqrt(vv)) + (a == c) * d[a]
            end
            m0 = moff[b] - 1
            if LAPACK.potrf!('U', A)[2] == 0
                copytri!(LAPACK.potri!('U', A), 'U', true)
                for j in 1:n, i in 1:n
                    minv[m0 + i + (j - 1) * n] = A[i, j]
                end
            else                       # A is overwritten: the diagonal again
                for j in 1:n, i in 1:n
                    a = blk[i]; vv = 2S.v[a]
                    minv[m0 + i + (j - 1) * n] = i == j ? 1 / (_GAL_C3 * exp(-0.5 * 0.0 / vv) / (vv * sqrt(vv)) + d[a]) : 0.0
                end
            end
        end
    end
    return (; perm = Int32.(perm), start, block_of, moff, minv)
end
_gal_precond!(z, r, blk::NamedTuple) = _gal_block_apply!(z, r, blk)
_gal_precond!(z, r, ::Nothing) = (z .= r)
_gal_block_apply!(z, r, blk) = _cs_block_apply!(z, r, (; blk..., nblocks = length(blk.start) - 1))

# Preconditioned conjugate gradient on (M + D) x = f, the three components stacked
# (one step length), from the x on entry. Each iteration lowers the damped squared
# field error, oo + 2 (x'(M + D)x/2 - x'f), by alpha r'z (no extra product); the
# solve has converged when the last 10 iterations lowered it by less than tol^2 oo
# (oo <= 0: tol^2 of the total decrease so far) or r'z = 0, and stops there, at
# itmax, or where the cut Gram matrix stops being positive (p'Ap <= 0).
# Returns (iterations, converged).
function _gal_cg!(x, r, p, Ap, z, f, d, A, S, cut2, blk, itmax, tol, oo)
    T = eltype(x)
    _gal_gram!(Ap, A, S, x, cut2); Ap .+= d .* x
    r .= f .- Ap
    _gal_precond!(z, r, blk); p .= z
    rz = Float64(dot(r, z))
    dec = zeros(itmax)
    its = 0; conv = false; tot = 0.0
    for it in 1:itmax
        rz > 0 || (conv = true; break)
        _gal_gram!(Ap, A, S, p, cut2); Ap .+= d .* p
        pAp = Float64(dot(p, Ap))
        pAp > 0 || break
        alpha = rz / pAp
        x .+= T(alpha) .* p; r .-= T(alpha) .* Ap
        _gal_precond!(z, r, blk)
        rzn = Float64(dot(r, z))
        its = it; dec[it] = alpha * rz; tot += dec[it]
        it >= 10 && sum(view(dec, it-9:it)) <= tol^2 * (oo > 0 ? oo : tot) && (conv = true; break)
        p .= z .+ T(rzn / rz) .* p
        rz = rzn
    end
    return its, conv
end

# Relative L2 error of the new field against the old: with r = f - (M + D) x,
# ||w_new - w_old||^2 = oo - x'f - x'r - x'Dx (oo = ||w_old||^2)
_gal_ferr(x, r, f, d, oo) =
    sqrt(max(oo - Float64(dot(x, f)) - Float64(dot(x, r)) - Float64(dot(x, d .* x)), 0.0) / oo)

# One streaming pass over the field points X (threaded, no sort of the field):
# point p of the host set S covers field point j when |x_j - x_p|^2 <= S.v[p]
# (S's cell must be at least the largest radius). Sets jmark[j] (j covered by
# anything) and pmark[p] (p covers some field point).
function _gal_cover!(jmark, pmark, S::_GalSet, X::AbstractMatrix)
    nt = Threads.nthreads(); N = size(X, 2)
    pm = [zeros(Bool, length(pmark)) for _ in 1:nt]      # per-chunk marks: no shared writes
    Threads.@threads for t in 1:nt
        pt = pm[t]
        for j in (t - 1) * cld(N, nt) + 1:min(t * cld(N, nt), N)
            x1 = X[1, j]; x2 = X[2, j]; x3 = X[3, j]
            a = floor(Int, (x1 - S.lo[1]) / S.h); b = floor(Int, (x2 - S.lo[2]) / S.h); c = floor(Int, (x3 - S.lo[3]) / S.h)
            a0 = max(a - 1, 0); a1 = min(a + 1, S.n[1] - 1)
            a0 > a1 && continue
            @inbounds for kc in max(c - 1, 0):min(c + 1, S.n[3] - 1), kb in max(b - 1, 0):min(b + 1, S.n[2] - 1)
                k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
                for q in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
                    if (S.x[1, q] - x1)^2 + (S.x[2, q] - x2)^2 + (S.x[3, q] - x3)^2 <= S.v[q]
                        jmark[j] = true; pt[q] = true
                    end
                end
            end
        end
    end
    for t in 1:nt
        pmark .|= pm[t]
    end
    return nothing
end

# Greedy thinning: by decreasing |Gamma|, a point closer than q to an already
# kept one is dropped. Returns the keep flags (input order).
function _gal_thin(X::Matrix{Float64}, gnorm::Vector{Float64}, q::Float64)
    K = length(gnorm)
    keep = trues(K)
    q > 0 || return keep
    S, perm = _gal_set(X, zeros(K), q)
    pos = invperm(perm)
    kept = falses(K)                   # in set order
    for o in sortperm(gnorm; rev=true)
        x1 = X[1, o]; x2 = X[2, o]; x3 = X[3, o]
        a = floor(Int, (x1 - S.lo[1]) / q); b = floor(Int, (x2 - S.lo[2]) / q); c = floor(Int, (x3 - S.lo[3]) / q)
        a0 = max(a - 1, 0); a1 = min(a + 1, S.n[1] - 1)
        drop = false
        for kc in max(c - 1, 0):min(c + 1, S.n[3] - 1), kb in max(b - 1, 0):min(b + 1, S.n[2] - 1)
            k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
            for p in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
                kept[p] && (S.x[1, p] - x1)^2 + (S.x[2, p] - x2)^2 + (S.x[3, p] - x3)^2 < q * q && (drop = true)
            end
        end
        keep[o] = !drop
        kept[pos[o]] = !drop
    end
    return keep
end

# Edge-point candidates: the axial stencil x_k +- a sgm0 e_j of every reset
# particle (positions Xo), kept where no field particle (X, host or device) and no
# accepted point lies within `spacing` sgm0; returns the points and their parents
# (columns of Xo)
function _gal_edges(Xo::Matrix{Float64}, X::AbstractMatrix, sgm0, offsets, spacing)
    q = spacing * sgm0
    n = 6 * length(offsets) * size(Xo, 2)
    C = zeros(3, n); par = zeros(Int, n)
    col = 0
    for k in axes(Xo, 2), a in offsets, dir in 1:3, sgn in (-1, 1)
        col += 1; par[col] = k
        for j in 1:3
            C[j, col] = Xo[j, k] + (j == dir ? sgn * a * sgm0 : 0.0)
        end
    end
    S, perm = _gal_set(C, fill(q^2, n), q)
    hit = zeros(Bool, n)
    Sd = _gal_to(parent(X), S)
    _gal_cover!(zeros(Bool, size(X, 2)), hit, Sd, X)
    _gal_free!(Sd)
    e = sort(perm[.!hit])                  # acceptance below in stencil order
    isempty(e) || (e = e[_gal_thin(C[:, e], zeros(length(e)), q)])
    return C[:, e], par[e]
end

"""
    GalerkinReset(; <keyword arguments>)

Parameters of the Galerkin core-size reset (`galerkin_shrink!`, the reset of
`CoreSpreading`). Lengths are in units of `sgm0`.

- `halo_tail = 2.0`, `halo_min = 1.0`: a non-reset particle within
  `max(halo_min, halo_tail * s_k)` of reset particle k (s_k^2 = sigma_k^2 - sgm0^2,
  the width of the tail the reset removes) keeps its core and takes a strength increment.
- `halo_tol = 3e-3`: of those, only the ones where the old reset particles' field is at
  least this fraction of its peak are kept (0: all).
- `edge_offsets = [1.0, 2.0]`, `edge_spacing = 0.7`, `edge_tol = 1e-2`: new `sgm0`
  points on the stencil x_k +- a e_j where no particle lies within `edge_spacing` and
  the old field there is at least `edge_tol` of its peak. Empty offsets: none.
- `prune_tol = 1e-2`: after the solve, edge points weaker than this fraction of the
  strongest reset particle are dropped.
- `thin = 0.5`: reset particles closer than this to a stronger kept reset particle are
  removed before the solve (their field stays in the target). 0: off.
- `damping = 1e-2`: mu in (M + mu diag M) c = f.
- `gram_cut = 5.0`: overlaps cut at r <= gram_cut sqrt(s_i^2 + s_j^2). At 4 the cut
  Gram matrix is not positive definite on the 5MW wake (the preconditioned CG finds
  its negative directions); 4.5-5 is.
- `block_cell = 3.0`, `block_cap = 512`: block-Jacobi preconditioner, blocks of the
  unknowns on a grid of this cell, split to at most `block_cap` (0: plain CG). Stores
  the block inverses, at most `block_cap` values per unknown.
- `tol = 5e-3`, `itmax = 300`: the CG stops once 10 iterations lower the squared
  relative L2 field error (damped) by less than `tol^2`, or at `itmax`.
- `gram_mem = 2^30`: host fields keep the Gram pairs (8 bytes each, each pair once) of
  as many rows as fit in this many bytes (1 GiB: 1.3e8 pairs) and evaluate the rest on
  the fly (plus 40 bytes per unknown and thread); device fields evaluate them all on
  the fly.

The particle count changes (edge points in, thinned particles out): the field needs
`maxparticles` headroom (about +1 % per reset on the 5MW wake; edge points that do
not fit are dropped weakest first, with a warning).
"""
struct GalerkinReset{R}
    halo_tail::R
    halo_min::R
    halo_tol::R
    edge_offsets::Vector{R}
    edge_spacing::R
    edge_tol::R
    prune_tol::R
    thin::R
    damping::R
    gram_cut::R
    block_cell::R
    block_cap::Int
    itmax::Int
    tol::R
    gram_mem::Int
    ws::_GalWork
end
GalerkinReset{R}(; halo_tail=2.0, halo_min=1.0, halo_tol=3e-3, edge_offsets=[1.0, 2.0], edge_spacing=0.7,
                   edge_tol=1e-2, prune_tol=1e-2, thin=0.5, damping=1e-2, gram_cut=5.0,
                   block_cell=3.0, block_cap=512, itmax=300, tol=5e-3, gram_mem=2^30, ws=_GalWork()) where R =
    GalerkinReset{R}(halo_tail, halo_min, halo_tol, R.(edge_offsets), edge_spacing, edge_tol, prune_tol,
                     thin, damping, gram_cut, block_cell, block_cap, itmax, tol, gram_mem, ws)
GalerkinReset(; optargs...) = GalerkinReset{FLOAT_TYPE}(; optargs...)
Base.convert(::Type{GalerkinReset{R}}, g::GalerkinReset) where R =
    GalerkinReset{R}(; (k => getfield(g, k) for k in fieldnames(GalerkinReset))...)

"""
    galerkin_shrink!(pfield, idx, sgm0; galerkin = GalerkinReset(), iterror = false, verbose = false, v_lvl = 0) -> NamedTuple

Shrink the cores of the particles `idx` (global indices) to `sgm0` by the Galerkin
L2 projection (see `GalerkinReset`), keeping the vorticity field: their non-reset
neighbours take strength increments, edge points may be added and close reset
particles thinned, so the particle count and order change. Works on any field (no
viscous scheme needed, e.g. `Inviscid`), host or device. It is also the reset of
`CoreSpreading`. Keep one `GalerkinReset` across calls: its workspace (stored Gram
pairs) is reused.

The conjugate gradient has not converged when it stops at `itmax` (or breaks down)
before its stopping rule; if the relative field error is then above `tol`, it
throws with `iterror = true` and warns otherwise.

Example, cores above 1.5 sgm0: `idx = findall(>(1.5sgm0), Array(view(pfield.particles, FLOWVPM.SIGMA_INDEX, 1:get_np(pfield))))`.

Returns counts (K reset, Kk kept, H halo, E edge points added, D thinned),
iterations, the relative L2 field error, Gram pairs, the solve's memory (bytes:
stored Gram pairs, buffers and block inverses), and the time of each phase.

Reverse-mode AD (Enzyme, host fields): the derivative covers the linear solve on
the selected geometry, i.e. the new strengths as functions of the positions, cores
and strengths of the reset particles, the halo and the edge points' parents
(`ext/FLOWVPMEnzymeExt.jl`, implicit rule: one adjoint solve with the same
preconditioned CG). The selection (which particles are thinned, which form the
halo, where edge points go, which are pruned) is piecewise constant and carries no
derivative; so does the overlap cut. The rule differentiates the exact solution:
tighten `tol` (e.g. 1e-10) and raise `itmax` for differentiable runs.
"""
galerkin_shrink!(pfield, idx::Vector{Int}, sgm0::Real; galerkin::GalerkinReset = GalerkinReset(),
                 iterror::Bool = false, verbose::Bool = false, v_lvl::Int = 0) =
    _galerkin_shrink!(pfield, idx, Float64(sgm0), galerkin, verbose, v_lvl, iterror)

# the CoreSpreading reset
_galerkin_reset!(pfield, cs, idx::Vector{Int}) =
    _galerkin_shrink!(pfield, idx, Float64(cs.sgm0), cs.galerkin, cs.verbose, cs.v_lvl, cs.iterror)

_gal_cols(Q, rows, js) = Float64[Q[r, j] for r in rows, j in js]
# rows of the field's columns js (Float64, host)
_gal_gather(P::Matrix, js, rows = first(X_INDEX):SIGMA_INDEX) = _gal_cols(P, rows, js)
_gal_gather(P, js, rows = first(X_INDEX):SIGMA_INDEX) =
    isempty(js) ? zeros(length(rows), 0) : Float64.(_radix_oversize_gather(P, js, rows))

# Selection (no derivative): thinning of the subset, the halo (non-static particles
# within the tail width, pruned to where the old subset field is at least halo_tol
# of its peak) and the edge points (stencil points away from every particle where
# the old field is at least edge_tol of its peak) as their parent and offset.
# Qs: positions, strengths and cores of the subset (columns of `sub`); the rest of
# the field is read in place (cover passes, on the device for a device field) and
# by one gather of the halo candidates. Returns the kept / thinned subset (`ks`
# columns of Qs, `subd` global), the halo (global), and the edge points as parent
# (column of Qs) and offset.
function _gal_plan(Qs, P, np, sub::Vector{Int}, sgm0::Float64, gr::GalerkinReset)
    K = length(sub); s02 = sgm0^2; cut2 = Float64(gr.gram_cut)^2
    X = view(P, X_INDEX, 1:np)
    Xo = Qs[X_INDEX, :]; Go = Qs[GAMMA_INDEX, :]; so = Qs[SIGMA_INDEX, :]; vo = so .^ 2
    keep = _gal_thin(Xo, [sqrt(Go[1, k]^2 + Go[2, k]^2 + Go[3, k]^2) for k in 1:K], gr.thin * sgm0)
    rk2 = [(max(gr.halo_min, gr.halo_tail * sqrt(max(s^2 - s02, 0.0)) / sgm0) * sgm0)^2 for s in so]
    Hs, _ = _gal_set(Xo, rk2, sqrt(maximum(rk2)))
    mark = zeros(Bool, np)
    Hsd = _gal_to(P, Hs)
    _gal_cover!(mark, zeros(Bool, K), Hsd, X)
    _gal_free!(Hsd)
    mark[sub] .= false
    hal = findall(mark)
    hal = hal[iszero.(vec(_gal_gather(P, hal, STATIC_INDEX:STATIC_INDEX)))]   # static: no increments
    Ec, epar = isempty(gr.edge_offsets) ? (zeros(3, 0), Int[]) : _gal_edges(Xo, X, sgm0, gr.edge_offsets, gr.edge_spacing)
    # edge points and (halo_tol > 0) halo particles where the old tail is at least
    # edge_tol / halo_tol of its peak (the old subset field at the subset points)
    if !isempty(epar) || (gr.halo_tol > 0 && !isempty(hal))
        O, operm = _gal_set(Xo, vo, sqrt(cut2 * 2 * maximum(vo)) / 2)
        Od = _gal_to(P, O); God = _gal_to(P, Go[:, operm])
        nrm3(A, j) = sqrt(A[1, j]^2 + A[2, j]^2 + A[3, j]^2)
        function tail(Y, v)
            o = _gal_to(P, zeros(3, size(Y, 2))); Yd = _gal_to(P, Y); vd = _gal_to(P, v)
            _gal_pairsum!(o, Yd, vd, Od, God, cut2)
            fo = Float64.(Array(o)); _gal_free!(o, Yd, vd)
            return fo
        end
        fs = tail(Xo[:, operm], fill(s02, K)); peak = maximum(nrm3(fs, k) for k in 1:K)
        if !isempty(epar)
            fx = tail(Ec, fill(s02, size(Ec, 2)))
            e = [nrm3(fx, e) >= gr.edge_tol * peak for e in 1:size(Ec, 2)]
            Ec = Ec[:, e]; epar = epar[e]
        end
        if gr.halo_tol > 0 && !isempty(hal)
            Qh = _gal_gather(P, hal, first(X_INDEX):SIGMA_INDEX)
            fh = tail(Qh[X_INDEX, :], Qh[SIGMA_INDEX, :] .^ 2)
            hal = hal[[nrm3(fh, j) >= gr.halo_tol * peak for j in 1:length(hal)]]
        end
        _gal_free!(Od, God)
    end
    return (; ks = findall(keep), subd = sub[.!keep], hal, epar, eoff = Ec .- Xo[:, epar])
end

# The solve's settings, a template of the field's array type, and what it reports
mutable struct _GalSolve{TG<:GalerkinReset,TP}
    gr::TG; sgm0::Float64; proto::TP
    its::Int; conv::Bool; err::Float64; nnz::Int; mem::Int; t::NTuple{4,Float64}
end
_GalSolve(gr, sgm0, proto) = _GalSolve(gr, sgm0, proto, 0, false, 0.0, 0, 0, (0.0, 0.0, 0.0, 0.0))

"""
    _gal_solve(Y, v, Xo, vo, Go, x0, sol) -> c

The linear solve of the reset on fixed geometry: unknowns at `Y` (3 x N) with
variances `v`, target the old subset field (positions `Xo`, variances `vo`,
strengths `Go`), warm start `x0`; returns the strengths `c` (3 x N, `Y`'s order).
Differentiable (Enzyme rule in ext/FLOWVPMEnzymeExt.jl, host fields).
"""
_gal_solve(Y, v, Xo, vo, Go, x0, sol::_GalSolve) = first(_gal_solve_sys(Y, v, Xo, vo, Go, x0, sol))

# ... also returns the system and the solution in set order (for the adjoint)
function _gal_solve_sys(Y::Matrix{Float64}, v::Vector{Float64}, Xo::Matrix{Float64}, vo::Vector{Float64},
                        Go::Matrix{Float64}, x0::Matrix{Float64}, sol::_GalSolve)
    gr = sol.gr; P = sol.proto; R = eltype(P)
    cut2 = Float64(gr.gram_cut)^2; N = size(Y, 2); K = size(Xo, 2)
    t0 = time()
    O, operm = _gal_set(Xo, vo, sqrt(cut2 * 2 * maximum(vo)) / 2)
    Od = _gal_to(P, O); God = _gal_to(P, Go[:, operm])
    # cells: half the widest pair cut on a device (its rows search vi + vmax), half
    # the median one on the host (its rows search 2 vi)
    U, uperm = _gal_set(Y, v, sqrt(cut2 * 2 * (P isa Matrix ? sort(v)[cld(N, 2)] : maximum(v))) / 2)
    Ud = _gal_to(P, U)
    f = _gal_to(P, zeros(3, N))
    _gal_pairsum!(f, Ud.x, Ud.v, Od, God, cut2)
    fo = _gal_to(P, zeros(3, K))
    _gal_pairsum!(fo, Od.x, Od.v, Od, God, cut2)
    oo = Float64(dot(God, fo))
    d = _gal_to(P, reshape(gr.damping * _GAL_C3 .* (2 .* U.v) .^ -1.5, 1, N))
    x = _gal_to(P, x0[:, uperm])
    t1 = time()

    A, nnz = P isa Matrix ? _GalOp(U, cut2, gr.gram_mem, gr.ws) : (nothing, -1)
    mem = A === nothing ? 0 : 8 * (A.rp[A.R + 1] - 1) + 8 * (N + 1) + 40N * length(A.y)
    t2 = time()

    blk = gr.block_cap > 0 ? _gal_blocks(U, Float64.(vec(Array(d))), cut2, Float64(gr.block_cell) * sol.sgm0, gr.block_cap, R) : nothing
    mem += blk === nothing ? 0 : sizeof(R) * length(blk.minv)
    blk = blk === nothing || P isa Matrix ? blk : map(a -> copyto!(similar(P, eltype(a), length(a)), a), blk)
    t3 = time()

    r = similar(x); p = similar(x); Ap = similar(x); z = similar(x)
    its, conv = _gal_cg!(x, r, p, Ap, z, f, d, A, Ud, cut2, blk, gr.itmax, Float64(gr.tol), oo)
    err = _gal_ferr(x, r, f, d, oo)
    c = zeros(3, N); c[:, uperm] .= Float64.(Array(x))
    t4 = time()

    sol.its = its; sol.conv = conv; sol.err = err; sol.nnz = nnz; sol.mem = mem
    sol.t = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    P isa Matrix && return c, (; U, uperm, O, operm, God, d, A, blk, cut2), x
    # device fields: the system is freed, not returned (the adjoint is host only)
    _gal_free!(Ud, Od, God, f, fo, x, r, p, Ap, z, d, blk)
    return c, (; A = nothing), nothing
end

# Reverse of `_gal_solve` (host): lambda solves (M + D) lambda = cbar with the same
# preconditioned CG (the operator is symmetric); the inputs' adjoints are then the
# derivatives of lambda' f - lambda' (M + D) c at fixed lambda and c (implicit
# function theorem). Returns them in the inputs' order.
function _gal_solve_adjoint(sys, x, cbar, sol::_GalSolve)
    sys.A isa _GalOp || error("reverse-mode AD through the Galerkin reset: host fields only")
    gr = sol.gr; U = sys.U; O = sys.O; N = size(U.x, 2); K = size(O.x, 2)
    b = cbar[:, sys.uperm]
    lam = zeros(3, N); r = similar(lam); p = similar(lam); Ap = similar(lam); z = similar(lam)
    _gal_cg!(lam, r, p, Ap, z, b, sys.d, sys.A, U, sys.cut2, sys.blk, gr.itmax, Float64(gr.tol), 0.0)
    gY = zeros(3, N); gv = zeros(N); gXo = zeros(3, K); gvo = zeros(K); gGo = zeros(3, K)
    _gal_pair_adj!(gY, gv, gXo, gvo, gGo, U.x, U.v, lam, O, sys.God, sys.cut2, 1.0)    # lambda' f
    _gal_pair_adj!(gY, gv, gY, gv, nothing, U.x, U.v, lam, U, x, sys.cut2, -1.0)        # -lambda' M c
    for i in 1:N                                       # -lambda' D c, D_ii = damping N(0; 2 v_i)
        gv[i] += 1.5 * sys.d[i] * (lam[1, i] * x[1, i] + lam[2, i] * x[2, i] + lam[3, i] * x[3, i]) / U.v[i]
    end
    Y = similar(gY); Y[:, sys.uperm] .= gY; v = similar(gv); v[sys.uperm] .= gv
    Xo = similar(gXo); Xo[:, sys.operm] .= gXo; vo = similar(gvo); vo[sys.operm] .= gvo
    Go = similar(gGo); Go[:, sys.operm] .= gGo
    return (; Y, v, Xo, vo, Go)
end

# Adjoint of the pair sum sgn sum_i sum_p (a_i . b_p) N(t_i - s_p; tv_i + sv_p) over
# the pairs within the cut (host): adds its derivatives with respect to the target
# positions and variances (gt, gtv), the source ones (gs, gsv; may alias gt, gtv)
# and, unless gb === nothing, the source strengths b (gb).
function _gal_pair_adj!(gt, gtv, gs, gsv, gb, tx, tv, a, S::_GalSet, b, cut2, sgn)
    @inbounds for i in axes(tx, 2)
        x1 = tx[1, i]; x2 = tx[2, i]; x3 = tx[3, i]; vi = tv[i]
        m = ceil(Int, sqrt(cut2 * (vi + S.vmax)) / S.h)
        ia = floor(Int, (x1 - S.lo[1]) / S.h); ib = floor(Int, (x2 - S.lo[2]) / S.h); ic = floor(Int, (x3 - S.lo[3]) / S.h)
        a0 = max(ia - m, 0); a1 = min(ia + m, S.n[1] - 1)
        a0 > a1 && continue
        for kc in max(ic - m, 0):min(ic + m, S.n[3] - 1), kb in max(ib - m, 0):min(ib + m, S.n[2] - 1)
            k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
            for p in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
                d1 = x1 - S.x[1, p]; d2 = x2 - S.x[2, p]; d3 = x3 - S.x[3, p]
                r2 = d1 * d1 + d2 * d2 + d3 * d3; V = vi + S.v[p]
                r2 <= cut2 * V || continue
                Nw = sgn * _GAL_C3 * exp(-0.5 * r2 / V) / (V * sqrt(V))
                w = (a[1, i] * b[1, p] + a[2, i] * b[2, p] + a[3, i] * b[3, p]) * Nw
                gx = -w / V                            # dN/dt = -N (t - s) / V
                gt[1, i] += gx * d1; gt[2, i] += gx * d2; gt[3, i] += gx * d3
                gs[1, p] -= gx * d1; gs[2, p] -= gx * d2; gs[3, p] -= gx * d3
                gV = w * (0.5 * r2 / V - 1.5) / V      # dN/dV
                gtv[i] += gV; gsv[p] += gV
                if gb !== nothing
                    gb[1, p] += a[1, i] * Nw; gb[2, p] += a[2, i] * Nw; gb[3, p] += a[3, i] * Nw
                end
            end
        end
    end
    return nothing
end

# Convergence check and summary line (no derivative)
function _gal_report(sol::_GalSolve, iterror::Bool, verbose::Bool, v_lvl::Int, K, np, H, E, D, t)
    if !sol.conv && sol.err > sol.gr.tol
        msg = "Galerkin reset: conjugate gradient not converged after $(sol.its) iterations (itmax $(sol.gr.itmax)), " *
              "field error $(Printf.@sprintf("%.2e", sol.err)) > tol $(sol.gr.tol)"
        iterror ? error(msg) : @warn(msg)
    end
    verbose && println("\t"^v_lvl, "core reset (Galerkin, $K of $np): field error ", Printf.@sprintf("%.2e", sol.err),
                       " after $(sol.its) its, halo $H, +$E edge -$D thinned, ", Printf.@sprintf("%.2f s", t))
    return nothing
end

function _galerkin_shrink!(pfield, idx::Vector{Int}, sgm0::Float64, gr::GalerkinReset, verbose::Bool, v_lvl::Int,
                           iterror::Bool = false)
    P = pfield.particles; np = get_np(pfield); R = eltype(P)
    t0 = time()
    # the radix FMM geometry is re-derived for the new cores at the next evaluation;
    # dropping the old cache first also frees its device memory for the reset
    clear_radix_fmm_cache!(pfield)

    # positions, strengths and cores of the subset (the halo's come with the plan)
    Qs = _gal_gather(P, idx)
    pl = _gal_plan(Qs, P, np, idx, sgm0, gr)
    ks = pl.ks; subk = idx[ks]; hal = pl.hal; Qh = _gal_gather(P, hal); Kk = length(ks); H = length(hal); N = Kk + H + length(pl.epar)
    t1 = time()

    # unknowns [kept subset at sgm0; halo on own cores; edge points at parent + offset]
    Y = hcat(Qs[X_INDEX, ks], Qh[X_INDEX, :], Qs[X_INDEX, pl.epar] .+ pl.eoff)
    v = vcat(fill(sgm0^2, Kk), Qh[SIGMA_INDEX, :] .^ 2, fill(sgm0^2, N - Kk - H))
    x0 = zeros(3, N); x0[:, 1:Kk] .= Qs[GAMMA_INDEX, ks]
    sol = _GalSolve(gr, sgm0, similar(P, 0, 0))
    c = _gal_solve(Y, v, Qs[X_INDEX, :], Qs[SIGMA_INDEX, :] .^ 2, Qs[GAMMA_INDEX, :], x0, sol)
    t4 = time()

    # prune weak edge points; keep within maxparticles (weakest dropped first)
    cn = [sqrt(c[1, j]^2 + c[2, j]^2 + c[3, j]^2) for j in 1:N]
    ekeep = findall(cn[Kk+H+1:N] .>= gr.prune_tol * maximum(cn[1:Kk]))
    room = pfield.maxparticles - (np - length(pl.subd))
    if length(ekeep) > room
        @warn "Galerkin reset: $(length(ekeep)) edge points, room for $room (maxparticles $(pfield.maxparticles)); dropping the weakest"
        ekeep = ekeep[sortperm(cn[Kk+H.+ekeep]; rev=true)[1:max(room, 0)]]
    end

    # write back: subset strengths and cores, halo increments, thinned out, edge points in
    Gh = Matrix{R}(Qh[GAMMA_INDEX, :] .+ c[:, Kk+1:Kk+H])
    _radix_oversize_scatter!(P, subk, first(GAMMA_INDEX):SIGMA_INDEX, Matrix{R}(vcat(c[:, 1:Kk], fill(sgm0, 1, Kk))))
    H > 0 && _radix_oversize_scatter!(P, hal, GAMMA_INDEX, Gh)
    _remove_particles!(pfield, pl.subd)
    if !isempty(ekeep)
        blk = zeros(R, nfields, length(ekeep))
        blk[X_INDEX, :] .= Y[:, Kk+H.+ekeep]
        blk[GAMMA_INDEX, :] .= c[:, Kk+H.+ekeep]
        blk[SIGMA_INDEX, :] .= sgm0
        blk[CIRCULATION_INDEX, :] .= 1
        add_particles!(pfield, blk)
    end
    t5 = time()

    K = length(idx); E = length(ekeep); D = length(pl.subd)
    _gal_report(sol, iterror, verbose, v_lvl, K, np, H, E, D, t5 - t0)
    return (; K, Kk, H, E, D, its=sol.its, err=sol.err, nnz=sol.nnz, mem=sol.mem,
            t_select=t1 - t0, t_rhs=sol.t[1], t_gram=sol.t[2], t_pc=sol.t[3], t_cg=sol.t[4], t_write=t5 - t4)
end
