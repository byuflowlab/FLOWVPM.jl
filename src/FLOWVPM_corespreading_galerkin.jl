#=##############################################################################
# DESCRIPTION
    Galerkin (L2-projection) core-spreading reset, `CoreSpreading(...; reset=:galerkin)`.

    The reset particles (the subset, or every particle in a full reset) go to
    `sgm0`; their non-reset neighbours keep their own cores and take strength
    increments (the halo, which carries the old tails); new `sgm0` points are
    added where the old tails reach past the last particle (edge points). The
    strengths minimise the L2 error of the Gaussian vorticity field:

        (M + mu diag M) c = f,   M_ij = N(x_i - x_j; s_i^2 + s_j^2),
                                 f_i  = sum_k Gold_k N(x_i - x_k; s_i^2 + so_k^2),

    N the normalized 3D Gaussian (FLOWVPM's gaussianerf zeta), the overlaps cut
    at r^2 <= gram_cut^2 (s_i^2 + s_j^2). M is SPD: plain conjugate gradient,
    one matrix-vector product per iteration. Study and parameters:
    research-notes flowunsteady-update/details/2026-10-14_rbf-reset-fix_1.md,
    "Overnight: recommended reset method".

    Host fields evaluate M with each pair once (`_GalOp`), storing the pairs of
    as many rows as fit in `gram_mem` and the rest on the fly; device fields
    evaluate it on the fly with the pair kernel (`_gal_row`, KA kernel in the GPU
    extension). The CG is block-Jacobi preconditioned (`_gal_blocks`). The selection (thinning, halo, edge points) runs on the host
    over a copy of positions, strengths and cores.
=###############################################################################

const _GAL_C3 = (2pi)^-1.5

# Cell-sorted point set: positions `x` (3 x n) and variances `v` sorted by cell
# (cell size `h`, box corner `lo`, `n` cells per axis); cell k (1-based linear id)
# holds cstart[k]:cstart[k+1]-1. `vmax` bounds the variances.
struct _GalSet{TX,TV,TI,R}
    x::TX; v::TV; cstart::TI
    n::NTuple{3,Int32}; lo::NTuple{3,R}; h::R; vmax::R
end

# Cell size h, enlarged until the dense cell table has at most 2^26 cells
function _gal_cellsize(ext, h)
    while prod(floor(Int, ext[d] / h) + 1 for d in 1:3) > 2^26
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

# Block-Jacobi preconditioner of M + D: the unknowns grouped on a grid of cell h,
# crowded cells split into octants down to `cap` (`_cs_split!`), each block's
# dense matrix inverted once per reset. Same storage as `_cs_blocks` (perm =
# positions in the set), applied by `_cs_block_apply!` (device: GPU extension).
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
    Threads.@threads for b in 1:nb
        blk = blocks[b]
        A = [(vv = S.v[a] + S.v[c]; r2 = (S.x[1, a] - S.x[1, c])^2 + (S.x[2, a] - S.x[2, c])^2 + (S.x[3, a] - S.x[3, c])^2;
              (r2 <= cut2 * vv) * _GAL_C3 * exp(-0.5 * r2 / vv) / (vv * sqrt(vv)) + (a == c) * d[a]) for a in blk, c in blk]
        F = cholesky(Symmetric(A); check = false)
        Ai = issuccess(F) ? inv(F) : [a == c ? 1 / A[a, a] : 0.0 for a in axes(A, 1), c in axes(A, 2)]
        copyto!(view(minv, moff[b]:(moff[b + 1] - 1)), vec(Ai))
    end
    return (; perm = Int32.(perm), start, block_of, moff, minv)
end
_gal_precond!(z, r, blk::NamedTuple) = _gal_block_apply!(z, r, blk)
_gal_precond!(z, r, ::Nothing) = (z .= r)
_gal_block_apply!(z, r, blk) = _cs_block_apply!(z, r, (; blk..., nblocks = length(blk.start) - 1))

# Preconditioned conjugate gradient on (M + D) x = f, the three components stacked
# (one step length), from the x on entry. Each iteration lowers the damped squared
# field error, oo + 2 (x'(M + D)x/2 - x'f), by alpha r'z (no extra product); the
# solve stops when the last 10 iterations lowered it by less than tol^2 oo, at
# itmax, or where the cut Gram matrix stops being positive (p'Ap <= 0).
function _gal_cg!(x, r, p, Ap, z, f, d, A, S, cut2, blk, itmax, tol, oo)
    T = eltype(x)
    _gal_gram!(Ap, A, S, x, cut2); Ap .+= d .* x
    r .= f .- Ap
    _gal_precond!(z, r, blk); p .= z
    rz = Float64(dot(r, z))
    dec = zeros(itmax)
    its = 0
    for it in 1:itmax
        rz > 0 || break
        _gal_gram!(Ap, A, S, p, cut2); Ap .+= d .* p
        pAp = Float64(dot(p, Ap))
        pAp > 0 || break
        alpha = rz / pAp
        x .+= T(alpha) .* p; r .-= T(alpha) .* Ap
        _gal_precond!(z, r, blk)
        rzn = Float64(dot(r, z))
        its = it; dec[it] = alpha * rz
        it >= 10 && sum(view(dec, it-9:it)) <= tol^2 * oo && break
        p .= z .+ T(rzn / rz) .* p
        rz = rzn
    end
    return its
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
# particle, kept where no field particle and no accepted point lies within
# `spacing` sgm0
function _gal_edges(X::AbstractMatrix, sub::Vector{Int}, sgm0, offsets, spacing)
    q = spacing * sgm0
    C = zeros(3, 6 * length(offsets) * length(sub))
    col = 0
    for k in sub, a in offsets, dir in 1:3, sgn in (-1, 1)
        col += 1
        for j in 1:3
            C[j, col] = X[j, k] + (j == dir ? sgn * a * sgm0 : 0.0)
        end
    end
    S, perm = _gal_set(C, fill(q^2, size(C, 2)), q)
    hit = zeros(Bool, size(C, 2))
    _gal_cover!(zeros(Bool, size(X, 2)), hit, S, X)
    C = C[:, sort(perm[.!hit])]            # acceptance below in stencil order
    isempty(C) && return C
    return C[:, _gal_thin(C, zeros(size(C, 2)), q)]
end

"""
    GalerkinReset(; <keyword arguments>)

Parameters of the Galerkin core-spreading reset (`CoreSpreading(...; reset=:galerkin)`).
Lengths are in units of the scheme's `sgm0`.

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
    galerkin_shrink!(pfield, idx, sgm0; galerkin = GalerkinReset(), verbose = false, v_lvl = 0) -> NamedTuple

Shrink the cores of the particles `idx` (global indices) to `sgm0` by the Galerkin
L2 projection (see `GalerkinReset`), keeping the vorticity field: their non-reset
neighbours take strength increments, edge points may be added and close reset
particles thinned, so the particle count and order change. Works on any field (no
viscous scheme needed, e.g. `Inviscid`), host or device. It is also the reset of
`CoreSpreading(...; reset=:galerkin)`. Keep one `GalerkinReset` across calls: its
workspace (stored Gram pairs) is reused.

Example, cores above 1.5 sgm0: `idx = findall(>(1.5sgm0), Array(view(pfield.particles, FLOWVPM.SIGMA_INDEX, 1:get_np(pfield))))`.

Returns counts (K reset, Kk kept, H halo, E edge points added, D thinned),
iterations, the relative L2 field error, Gram pairs, the solve's memory (bytes:
stored Gram pairs, buffers and block inverses), and the time of each phase.
"""
galerkin_shrink!(pfield, idx::Vector{Int}, sgm0::Real; galerkin::GalerkinReset = GalerkinReset(),
                 verbose::Bool = false, v_lvl::Int = 0) =
    _galerkin_shrink!(pfield, idx, Float64(sgm0), galerkin, verbose, v_lvl)

# the CoreSpreading reset
_galerkin_reset!(pfield, cs, idx::Vector{Int}) =
    _galerkin_shrink!(pfield, idx, Float64(cs.sgm0), cs.galerkin::GalerkinReset, cs.verbose, cs.v_lvl)

function _galerkin_shrink!(pfield, idx::Vector{Int}, sgm0::Float64, gr::GalerkinReset, verbose::Bool, v_lvl::Int)
    P = pfield.particles; np = get_np(pfield); R = eltype(P)
    s02 = sgm0^2
    cut2 = Float64(gr.gram_cut)^2
    t0 = time()

    # positions, strengths and cores: a view of a host field, one gather of a device one
    Q = P isa Matrix ? view(P, first(X_INDEX):SIGMA_INDEX, 1:np) : _radix_oversize_gather(P, collect(1:np), first(X_INDEX):SIGMA_INDEX)
    X = view(Q, X_INDEX, :); S = view(Q, SIGMA_INDEX, :)
    sub = idx; K = length(sub)
    Xo = Float64.(X[:, sub]); Go = Float64.(Q[GAMMA_INDEX, sub]); vo = Float64.(S[sub]) .^ 2

    # thinning, halo (per-particle radius from the tail width), edge candidates
    keep = _gal_thin(Xo, [sqrt(Go[1, k]^2 + Go[2, k]^2 + Go[3, k]^2) for k in 1:K], gr.thin * sgm0)
    subk = sub[keep]; subd = sub[.!keep]; Kk = length(subk)
    rk2 = [(max(gr.halo_min, gr.halo_tail * sqrt(max(S[k]^2 - s02, 0.0)) / sgm0) * sgm0)^2 for k in sub]
    Hs, _ = _gal_set(Xo, rk2, sqrt(maximum(rk2)))
    mark = zeros(Bool, np)
    _gal_cover!(mark, zeros(Bool, K), Hs, X)
    mark[sub] .= false
    mark .&= iszero.(P isa Matrix ? view(P, STATIC_INDEX, 1:np) : Array(view(P, STATIC_INDEX, 1:np)))   # static: no increments
    hal = findall(mark); H = length(hal)
    Ec = isempty(gr.edge_offsets) ? zeros(3, 0) : _gal_edges(X, sub, sgm0, gr.edge_offsets, gr.edge_spacing)
    t1 = time()

    # old subset as the source set; edge filter (old tail >= edge_tol of its peak)
    O, operm = _gal_set(Xo, vo, sqrt(cut2 * 2 * maximum(vo)) / 2)
    Od = _gal_to(P, O); God = _gal_to(P, Go[:, operm])
    # edge points and (halo_tol > 0) halo particles where the old tail is at least
    # edge_tol / halo_tol of its peak (the old subset field at the subset points)
    nrm3(A, j) = sqrt(A[1, j]^2 + A[2, j]^2 + A[3, j]^2)
    tail(Y, v) = (o = _gal_to(P, zeros(3, size(Y, 2))); _gal_pairsum!(o, _gal_to(P, Y), _gal_to(P, v), Od, God, cut2); Float64.(Array(o)))
    if size(Ec, 2) > 0 || (gr.halo_tol > 0 && H > 0)
        fs = tail(Xo[:, operm], fill(s02, K)); peak = maximum(nrm3(fs, k) for k in 1:K)
        if size(Ec, 2) > 0
            fx = tail(Ec, fill(s02, size(Ec, 2)))
            Ec = Ec[:, [nrm3(fx, e) >= gr.edge_tol * peak for e in 1:size(Ec, 2)]]
        end
        if gr.halo_tol > 0 && H > 0
            fh = tail(Float64.(X[:, hal]), Float64.(S[hal]) .^ 2)
            hal = hal[[nrm3(fh, j) >= gr.halo_tol * peak for j in 1:H]]; H = length(hal)
        end
    end
    E = size(Ec, 2); N = Kk + H + E

    # unknowns [kept subset at sgm0; halo on own cores; edge points at sgm0], cell-sorted
    Y = hcat(X[:, subk], X[:, hal], Ec)
    v = vcat(fill(s02, Kk), S[hal] .^ 2, fill(s02, E))
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
    x0 = zeros(3, N); x0[:, 1:Kk] .= Q[GAMMA_INDEX, subk]
    x = _gal_to(P, x0[:, uperm])
    t2 = time()

    A, nnz = P isa Matrix ? _GalOp(U, cut2, gr.gram_mem, gr.ws) : (nothing, -1)
    mem = A === nothing ? 0 : 8 * (A.rp[A.R + 1] - 1) + 8 * (N + 1) + 40N * length(A.y)
    t3 = time()

    blk = gr.block_cap > 0 ? _gal_blocks(U, Float64.(vec(Array(d))), cut2, Float64(gr.block_cell) * sgm0, gr.block_cap, R) : nothing
    mem += blk === nothing ? 0 : sizeof(R) * length(blk.minv)
    blk = blk === nothing || P isa Matrix ? blk : map(a -> copyto!(similar(P, eltype(a), length(a)), a), blk)
    t3b = time()

    r = similar(x); p = similar(x); Ap = similar(x); z = similar(x)
    its = _gal_cg!(x, r, p, Ap, z, f, d, A, Ud, cut2, blk, gr.itmax, Float64(gr.tol), oo)
    err = _gal_ferr(x, r, f, d, oo)
    c = zeros(3, N); c[:, uperm] .= Float64.(Array(x))
    t4 = time()

    # prune weak edge points; keep within maxparticles (weakest dropped first)
    cn = [sqrt(c[1, j]^2 + c[2, j]^2 + c[3, j]^2) for j in 1:N]
    ekeep = findall(cn[Kk+H+1:N] .>= gr.prune_tol * maximum(cn[1:Kk]))
    room = pfield.maxparticles - (np - length(subd))
    if length(ekeep) > room
        @warn "Galerkin reset: $(length(ekeep)) edge points, room for $room (maxparticles $(pfield.maxparticles)); dropping the weakest"
        ekeep = ekeep[sortperm(cn[Kk+H.+ekeep]; rev=true)[1:max(room, 0)]]
    end

    # write back: subset strengths and cores, halo increments, thinned out, edge points in
    _radix_oversize_scatter!(P, subk, first(GAMMA_INDEX):SIGMA_INDEX, Matrix{R}(vcat(c[:, 1:Kk], fill(sgm0, 1, Kk))))
    H > 0 && _radix_oversize_scatter!(P, hal, GAMMA_INDEX, Matrix{R}(Q[GAMMA_INDEX, hal] .+ c[:, Kk+1:Kk+H]))
    for i in sort(subd; rev=true)
        remove_particle(pfield, i)
    end
    if !isempty(ekeep)
        blk = zeros(R, nfields, length(ekeep))
        blk[X_INDEX, :] .= Ec[:, ekeep]
        blk[GAMMA_INDEX, :] .= c[:, Kk+H.+ekeep]
        blk[SIGMA_INDEX, :] .= sgm0
        blk[CIRCULATION_INDEX, :] .= 1
        add_particles!(pfield, blk)
    end
    clear_radix_fmm_cache!(pfield)
    t5 = time()

    verbose && println("\t"^v_lvl, "core reset (Galerkin, $K of $np): field error ", Printf.@sprintf("%.2e", err),
            " after $its its, halo $H, +$(length(ekeep)) edge -$(length(subd)) thinned, ",
            Printf.@sprintf("%.2f s", t5 - t0))
    return (; K, Kk, H, E=length(ekeep), D=length(subd), its, err, nnz, mem,
            t_select=t1 - t0, t_rhs=t2 - t1, t_gram=t3 - t2, t_pc=t3b - t3, t_cg=t4 - t3b, t_write=t5 - t4)
end
