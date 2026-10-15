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

    Host fields store M in CSR (Float32 values, Int32 columns) below
    `gram_mem` bytes and evaluate it on the fly above; device fields evaluate it
    on the fly with the same pair kernel (`_gal_row`, KA kernel in the GPU
    extension). The selection (thinning, halo, edge points) runs on the host
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

# Gram matrix of a host set in CSR (rows sorted like the set), or `nothing`
# when it would take more than `maxbytes` (the operator is then on the fly)
struct _GalCSR
    rp::Vector{Int}; col::Vector{Int32}; val::Vector{Float32}
end

# Row i of the Gram matrix of S: counts the entries when `col` is empty, else
# writes them from position p
function _gal_csr_row!(col, val, p, S::_GalSet, i, cut2)
    x1 = S.x[1, i]; x2 = S.x[2, i]; x3 = S.x[3, i]; vi = S.v[i]
    m = ceil(Int, sqrt(cut2 * (vi + S.vmax)) / S.h)
    a = floor(Int, (x1 - S.lo[1]) / S.h); b = floor(Int, (x2 - S.lo[2]) / S.h); c = floor(Int, (x3 - S.lo[3]) / S.h)
    a0 = max(a - m, 0); a1 = min(a + m, S.n[1] - 1)
    cnt = 0
    @inbounds for kc in max(c - m, 0):min(c + m, S.n[3] - 1), kb in max(b - m, 0):min(b + m, S.n[2] - 1)
        k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
        for q in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
            d1 = S.x[1, q] - x1; d2 = S.x[2, q] - x2; d3 = S.x[3, q] - x3
            r2 = d1 * d1 + d2 * d2 + d3 * d3
            vv = vi + S.v[q]
            if r2 <= cut2 * vv
                if !isempty(col)
                    col[p + cnt] = q
                    val[p + cnt] = _GAL_C3 * exp(-0.5 * r2 / vv) / (vv * sqrt(vv))
                end
                cnt += 1
            end
        end
    end
    return cnt
end

function _gal_csr(S::_GalSet, cut2, maxbytes)
    N = size(S.x, 2); cnt = zeros(Int, N)
    empt = Int32[]; emptv = Float32[]
    Threads.@threads for i in 1:N
        cnt[i] = _gal_csr_row!(empt, emptv, 0, S, i, cut2)
    end
    nnz = sum(cnt)
    8 * nnz > maxbytes && return nothing, nnz
    rp = Vector{Int}(undef, N + 1); rp[1] = 1
    for i in 1:N; rp[i + 1] = rp[i] + cnt[i]; end
    col = Vector{Int32}(undef, nnz); val = Vector{Float32}(undef, nnz)
    Threads.@threads for i in 1:N
        _gal_csr_row!(col, val, rp[i], S, i, cut2)
    end
    return _GalCSR(rp, col, val), nnz
end

# y = M x: stored CSR, or on the fly over the set
function _gal_gram!(y, A::_GalCSR, S, x, cut2)
    Threads.@threads for i in 1:length(A.rp) - 1
        o1 = 0.0; o2 = 0.0; o3 = 0.0
        @inbounds for p in A.rp[i]:A.rp[i + 1] - 1
            j = A.col[p]; w = Float64(A.val[p])
            o1 += w * x[1, j]; o2 += w * x[2, j]; o3 += w * x[3, j]
        end
        y[1, i] = o1; y[2, i] = o2; y[3, i] = o3
    end
    return y
end
_gal_gram!(y, ::Nothing, S, x, cut2) = _gal_pairsum!(y, S.x, S.v, S, x, cut2)

# Conjugate gradient on (M + D) x = f, the three components stacked (one step
# length), from the x on entry. Stops at itmax or, with field_tol > 0, when the
# relative L2 field error (`_gal_ferr`) reaches it. Returns the iterations.
function _gal_cg!(x, r, p, Ap, f, d, A, S, cut2, itmax, field_tol, oo)
    T = eltype(x)
    _gal_gram!(Ap, A, S, x, cut2); Ap .+= d .* x
    r .= f .- Ap; p .= r
    rr = Float64(dot(r, r))
    its = 0
    for it in 1:itmax
        rr > 0 || break
        _gal_gram!(Ap, A, S, p, cut2); Ap .+= d .* p
        alpha = rr / Float64(dot(p, Ap))
        x .+= T(alpha) .* p; r .-= T(alpha) .* Ap
        rrn = Float64(dot(r, r))
        its = it
        field_tol > 0 && _gal_ferr(x, r, f, d, oo) <= field_tol && break
        p .= r .+ T(rrn / rr) .* p
        rr = rrn
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
function _gal_cover!(jmark, pmark, S::_GalSet, X::Matrix{Float64})
    Threads.@threads for j in 1:size(X, 2)
        x1 = X[1, j]; x2 = X[2, j]; x3 = X[3, j]
        a = floor(Int, (x1 - S.lo[1]) / S.h); b = floor(Int, (x2 - S.lo[2]) / S.h); c = floor(Int, (x3 - S.lo[3]) / S.h)
        a0 = max(a - 1, 0); a1 = min(a + 1, S.n[1] - 1)
        a0 > a1 && continue
        @inbounds for kc in max(c - 1, 0):min(c + 1, S.n[3] - 1), kb in max(b - 1, 0):min(b + 1, S.n[2] - 1)
            k0 = 1 + a0 + S.n[1] * (kb + S.n[2] * kc)
            for q in S.cstart[k0]:S.cstart[k0 + a1 - a0 + 1] - 1
                if (S.x[1, q] - x1)^2 + (S.x[2, q] - x2)^2 + (S.x[3, q] - x3)^2 <= S.v[q]
                    jmark[j] = true; pmark[q] = true
                end
            end
        end
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
function _gal_edges(X::Matrix{Float64}, sub::Vector{Int}, sgm0, offsets, spacing)
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
    hit = zeros(Bool, size(C, 2))           # written from threads: not a BitVector
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
- `edge_offsets = [1.0, 2.0]`, `edge_spacing = 0.7`, `edge_tol = 1e-2`: new `sgm0`
  points on the stencil x_k +- a e_j where no particle lies within `edge_spacing` and
  the old field there is at least `edge_tol` of its peak. Empty offsets: none.
- `prune_tol = 1e-2`: after the solve, edge points weaker than this fraction of the
  strongest reset particle are dropped.
- `thin = 0.5`: reset particles closer than this to a stronger kept reset particle are
  removed before the solve (their field stays in the target). 0: off.
- `damping = 1e-2`: mu in (M + mu diag M) c = f.
- `gram_cut = 4.0`: overlaps cut at r <= gram_cut sqrt(s_i^2 + s_j^2).
- `itmax = 300`, `field_tol = 0`: CG iterations; with `field_tol > 0` the solve stops
  once the relative L2 field error reaches it.
- `gram_mem = Sys.total_memory() ÷ 4`: host fields store the Gram matrix (8 bytes per
  pair) up to this many bytes and evaluate it on the fly above; device fields always
  evaluate it on the fly.

The particle count changes (edge points in, thinned particles out): the field needs
`maxparticles` headroom (about +1 % per reset on the 5MW wake; edge points that do
not fit are dropped weakest first, with a warning).
"""
struct GalerkinReset{R}
    halo_tail::R
    halo_min::R
    edge_offsets::Vector{R}
    edge_spacing::R
    edge_tol::R
    prune_tol::R
    thin::R
    damping::R
    gram_cut::R
    itmax::Int
    field_tol::R
    gram_mem::Int
end
GalerkinReset{R}(; halo_tail=2.0, halo_min=1.0, edge_offsets=[1.0, 2.0], edge_spacing=0.7,
                   edge_tol=1e-2, prune_tol=1e-2, thin=0.5, damping=1e-2, gram_cut=4.0,
                   itmax=300, field_tol=0.0, gram_mem=Int(Sys.total_memory() ÷ 4)) where R =
    GalerkinReset{R}(halo_tail, halo_min, R.(edge_offsets), edge_spacing, edge_tol, prune_tol,
                     thin, damping, gram_cut, itmax, field_tol, gram_mem)
GalerkinReset(; optargs...) = GalerkinReset{FLOAT_TYPE}(; optargs...)

"""
    _galerkin_reset!(pfield, cs, idx) -> NamedTuple

The Galerkin reset of the particles `idx` (global indices; every particle in a full
reset). Returns counts (K reset, Kk kept, H halo, E edge points added, D thinned),
iterations, the relative L2 field error, Gram pairs, and the time of each phase
(select, rhs, gram, cg, write).
"""
function _galerkin_reset!(pfield, cs, idx::Vector{Int})
    gr = cs.galerkin::GalerkinReset
    P = pfield.particles; np = get_np(pfield); R = eltype(P)
    sgm0 = Float64(cs.sgm0); s02 = sgm0^2
    cut2 = Float64(gr.gram_cut)^2
    t0 = time()

    # host copy of positions, strengths and cores (one gather on a device field)
    Q = Float64.(_radix_oversize_gather(P, collect(1:np), first(X_INDEX):SIGMA_INDEX))
    X = Q[X_INDEX, :]; S = Q[SIGMA_INDEX, :]
    sub = idx; K = length(sub)
    Xo = X[:, sub]; Go = Q[GAMMA_INDEX, sub]; vo = S[sub] .^ 2

    # thinning, halo (per-particle radius from the tail width), edge candidates
    keep = _gal_thin(Xo, [sqrt(Go[1, k]^2 + Go[2, k]^2 + Go[3, k]^2) for k in 1:K], gr.thin * sgm0)
    subk = sub[keep]; subd = sub[.!keep]; Kk = length(subk)
    rk2 = [(max(gr.halo_min, gr.halo_tail * sqrt(max(S[k]^2 - s02, 0.0)) / sgm0) * sgm0)^2 for k in sub]
    Hs, _ = _gal_set(Xo, rk2, sqrt(maximum(rk2)))
    mark = zeros(Bool, np)
    _gal_cover!(mark, zeros(Bool, K), Hs, X)
    mark[sub] .= false
    hal = findall(mark); H = length(hal)
    Ec = isempty(gr.edge_offsets) ? zeros(3, 0) : _gal_edges(X, sub, sgm0, gr.edge_offsets, gr.edge_spacing)
    t1 = time()

    # old subset as the source set; edge filter (old tail >= edge_tol of its peak)
    O, operm = _gal_set(Xo, vo, sqrt(cut2 * 2 * maximum(vo)) / 2)
    Od = _gal_to(P, O); God = _gal_to(P, Go[:, operm])
    if size(Ec, 2) > 0
        fs = _gal_to(P, zeros(3, K)); fx = _gal_to(P, zeros(3, size(Ec, 2)))
        _gal_pairsum!(fs, Od.x, _gal_to(P, fill(s02, K)), Od, God, cut2)
        _gal_pairsum!(fx, _gal_to(P, Ec), _gal_to(P, fill(s02, size(Ec, 2))), Od, God, cut2)
        fs = Float64.(Array(fs)); fx = Float64.(Array(fx))
        thr = gr.edge_tol * maximum(sqrt(fs[1, k]^2 + fs[2, k]^2 + fs[3, k]^2) for k in 1:K)
        Ec = Ec[:, [sqrt(fx[1, e]^2 + fx[2, e]^2 + fx[3, e]^2) >= thr for e in 1:size(Ec, 2)]]
    end
    E = size(Ec, 2); N = Kk + H + E

    # unknowns [kept subset at sgm0; halo on own cores; edge points at sgm0], cell-sorted
    Y = hcat(X[:, subk], X[:, hal], Ec)
    v = vcat(fill(s02, Kk), S[hal] .^ 2, fill(s02, E))
    U, uperm = _gal_set(Y, v, sqrt(cut2 * 2 * maximum(v)) / 2)
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

    A, nnz = P isa Matrix ? _gal_csr(U, cut2, gr.gram_mem) : (nothing, -1)
    t3 = time()

    r = similar(x); p = similar(x); Ap = similar(x)
    its = _gal_cg!(x, r, p, Ap, f, d, A, Ud, cut2, gr.itmax, Float64(gr.field_tol), oo)
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

    println("  core reset (Galerkin, $K of $np): field error ", Printf.@sprintf("%.2e", err),
            " after $its its, halo $H, +$(length(ekeep)) edge -$(length(subd)) thinned, ",
            Printf.@sprintf("%.2f s", t5 - t0))
    return (; K, Kk, H, E=length(ekeep), D=length(subd), its, err, nnz,
            t_select=t1 - t0, t_rhs=t2 - t1, t_gram=t3 - t2, t_cg=t4 - t3, t_write=t5 - t4)
end
