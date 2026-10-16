#=##############################################################################
# DESCRIPTION
    Enzyme activity rules: pieces of a time step that carry no derivative, so
    that Enzyme reverse mode (default settings) differentiates a UJ_direct step.
=###############################################################################
module FLOWVPMEnzymeExt

import FLOWVPM
import EnzymeCore
import EnzymeCore: EnzymeRules
using StaticArrays: SVector, SMatrix
using LinearAlgebra: dot, cross

const vpm = FLOWVPM

# Regularization-radius tuning (Roots.Brent breaks Enzyme type analysis). Exact
# while ρ/σ is fixed (autotune_reg_error=false; the direct path never reads it).
EnzymeRules.inactive(::typeof(vpm.solve_ρ_over_σ), args...) = nothing

# Filament topology (BitMatrix fields); built by the ParticleField constructor.
EnzymeRules.inactive_type(::Type{<:vpm.FilamentEdgeGraph}) = true
EnzymeRules.inactive_type(::Type{<:vpm.FilamentEdgeWorkspace}) = true

# Two-level dynamic SFS request/delivery flags (global IdDict{Any,Bool}): Bool plumbing.
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_request!), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_delivered!), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_requested), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._sfs_dsigma_delivered), args...) = nothing

# Enzyme forward mode leaves the shadow of an array that a non-inlined callee
# fills with a constant (`zeros`, `fill!`) and returns uninitialized: the
# tangent then reads heap garbage (NaN, 1e300, wrong values, varying per call;
# Enzyme 0.13.216). UJ_direct's FastMultipole buffers are allocated that way, so
# return explicitly zeroed shadows.
for f in (vpm.fmm.allocate_target_buffer, vpm.fmm.allocate_source_buffer)
    @eval function EnzymeRules.forward(config::EnzymeRules.FwdConfig, func::EnzymeCore.Const{typeof($f)},
                                       ::Type{RT}, args::EnzymeCore.Annotation...) where RT
        b = func.val(map(a -> a.val, args)...)
        N = EnzymeRules.width(config)
        db = N == 1 ? zero(b) : ntuple(_ -> zero(b), Val(N))
        if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
            return N == 1 ? EnzymeCore.Duplicated(b, db) : EnzymeCore.BatchDuplicated(b, db)
        end
        return EnzymeRules.needs_shadow(config) ? db : EnzymeRules.needs_primal(config) ? b : nothing
    end
end

# Device-kernel scheme/time-step constants (dt, RK3 coefficients, rlxf, SFS
# bounds): no derivative, and KernelAbstractions rejects Active GPU arguments.
EnzymeRules.inactive_type(::Type{<:vpm.KConst}) = true

# Cache plumbing: drops the radix coupling after a reset (delete! on a global
# IdDict, a runtime-activity error under Enzyme); carries no derivative.
EnzymeRules.inactive(::typeof(vpm.clear_radix_fmm_cache!), args...) = nothing

# Core-size reset (Galerkin): the selection is integer bookkeeping and the report
# only prints; the linear solve on the selected geometry carries an implicit rule
# (its adjoint is one solve with the same preconditioned CG, `_gal_solve_adjoint`)
# instead of being differentiated iteration by iteration.
EnzymeRules.inactive(::typeof(vpm._gal_plan), args...) = nothing
# which particles to reset (index lists of the cores above a threshold / the largest)
EnzymeRules.inactive(::typeof(vpm._radix_oversize_above), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._radix_oversize_top), args...) = nothing
EnzymeRules.inactive(::typeof(vpm._gal_report), args...) = nothing
EnzymeRules.inactive_type(::Type{<:vpm._GalSolve}) = true

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm._gal_solve)}, ::Type{RT},
                                      Y, v, Xo, vo, Go, x0, sol) where RT
    rp = get(task_local_storage(), :FLOWVPM_gal_replay, nothing)
    if rp === nothing
        c, sys, x = vpm._gal_solve_sys(Y.val, v.val, Xo.val, vo.val, Go.val, x0.val, sol.val)
    else                       # checkpointed_gradient! re-run: the forward run's solution, no CG
        r = popfirst!(rp)
        r.key == vpm._gal_key(Y.val, v.val, Xo.val, vo.val, Go.val) ||
            error("checkpointed_gradient!: Galerkin reset inputs differ from the forward run")
        c, sys, x = vpm._gal_solve_sys(Y.val, v.val, Xo.val, vo.val, Go.val, r.c, sol.val; solve = false)
        sol.val.its = r.its; sol.val.conv = r.conv; sol.val.err = r.err
    end
    A = sys.A                  # its stored pairs live in the GalerkinReset's workspace: keep a copy
    A isa vpm._GalOp && (sys = merge(sys, (; A = vpm._GalOp(A.R, A.rp, copy(A.col), copy(A.val), A.y, A.ib, A.rb))))
    dc = zero(c)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? c : nothing,
                                       EnzymeRules.needs_shadow(config) ? dc : nothing, (sys, x, dc))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm._gal_solve)}, ::Type{RT}, tape,
                             Y, v, Xo, vo, Go, x0, sol) where RT
    sys, x, dc = tape
    g = vpm._gal_solve_adjoint(sys, x, dc, sol.val)
    for (a, ga) in ((Y, g.Y), (v, g.v), (Xo, g.Xo), (vo, g.vo), (Go, g.Go))
        a isa EnzymeCore.Duplicated && (a.dval .+= ga)
    end
    dc .= 0                    # x0 (warm start) does not change the solution: no adjoint
    return ntuple(_ -> nothing, 7)
end

# ------------------------------------------------------------------------------
# Direct-sum adjoints (host fields): U, J of the regularized Biot-Savart sum and
# the SFS stretching Estr_direct, written as one reverse pair loop each instead
# of Enzyme differentiating the FastMultipole/pair loops. Forward: the primal,
# plus a copy of the particle columns it read. Reverse: cotangents of the
# accumulated outputs (U, J; SFS) pulled back to X, Gamma, sigma (and J for
# Estr); the outputs' own cotangents pass through (they are accumulated).
# Pair (target i, source j), d = x_i - x_j, r = |d|, q = r/sigma_j:
#   U = g c,  J = A c d' + B E(Gamma),  c = -k (d x Gamma)/r^3,  k = 1/4pi,
#   A = g'/(sigma r) - 3g/r^2,  B = -k g/r^3,  E_ab = eps_abc Gamma_c.
# ------------------------------------------------------------------------------

# second derivative of g from g' = dgdr (q > 0), per kernel (the rules apply only to these kernels)
_d2g(::typeof(vpm.g_dgdr_gauserf), q, g1) = g1*(2 - q*q)/q
_d2g(::typeof(vpm.g_dgdr_gaus), q, g1) = g1*(2 - 3*q*q*q)/q
_d2g(::typeof(vpm.g_dgdr_wnklmns), q, g1) = g1*(2 - 5*q*q)/(q*(q*q + 1))
_d2g(::typeof(vpm.g_dgdr_sing), q, g1) = zero(q)
# zeta and its derivative, per kernel
_zeta_dzeta(f::typeof(vpm.zeta_gauserf), q) = (z = f(q); (z, -q*z))
_zeta_dzeta(f::typeof(vpm.zeta_gaus), q) = (z = f(q); (z, -3*q*q*z))
_zeta_dzeta(f::typeof(vpm.zeta_wnklmns), q) = (z = f(q); (z, -7*q*z/(q*q + 1)))
_zeta_dzeta(f::typeof(vpm.zeta_sing), q) = (f(q), zero(q))
_zd(zeta) = q -> _zeta_dzeta(zeta, q)

const _KnownKernel = vpm.Kernel{<:Union{typeof.((vpm.zeta_gauserf, vpm.zeta_gaus, vpm.zeta_wnklmns, vpm.zeta_sing))...},
                                <:Any, <:Any,
                                <:Union{typeof.((vpm.g_dgdr_gauserf, vpm.g_dgdr_gaus, vpm.g_dgdr_wnklmns, vpm.g_dgdr_sing))...}}
const _HostField = vpm.ParticleField{<:AbstractFloat, <:Any, <:Any, <:Any, <:Any, <:_KnownKernel, <:Any, <:Any, <:Any, <:Matrix}

_col3(P, rows, i) = SVector{3}(P[rows[1], i], P[rows[2], i], P[rows[3], i])
_col9(P, rows, i) = SMatrix{3,3}(ntuple(k -> P[rows[k], i], Val(9)))

# reverse of one pair: (dbar, Gammabar, sigmabar) from Ubar = u, Jbar = s
@inline function _uj_pair_adj(d, G, sgm, u, s, gdg)
    k = vpm.const4
    r2 = dot(d, d); r = sqrt(r2); r3 = r2*r
    g, g1 = gdg(r/sgm); g2 = _d2g(gdg, r/sgm, g1)
    c = (-k/r3) * cross(d, G)
    A = g1/(sgm*r) - 3*g/r2
    B = -k*g/r3
    sd = s*d
    tau = SVector(s[2,3] - s[3,2], s[3,1] - s[1,3], s[1,2] - s[2,1])
    P = dot(u, c); Q = dot(c, sd); T = dot(tau, G)
    v = g*u + A*sd
    Ar = g2/(sgm*sgm*r) - 4*g1/(sgm*r2) + 6*g/r3
    Br = -k*g1/(sgm*r3) + 3*k*g/(r2*r2)
    dbar = (-k/r3)*cross(G, v) - (3*dot(v, c)/r2)*d + A*(s'*c) + ((P*g1/sgm + Q*Ar + T*Br)/r)*d
    Gbar = (-k/r3)*cross(v, d) + B*tau
    sbar = -P*g1*r/(sgm*sgm) + Q*(2*g1/(sgm*sgm*r) - g2/(sgm*sgm*sgm)) + T*k*g1/(sgm*sgm*r2)
    return dbar, Gbar, sbar
end

# outer loop over sources in nthreads chunks (source-side cotangents written in
# place); target-side cotangents go to per-chunk buffers, summed at the end
function _chunked_pairs!(f!, dP, np, ntrows)
    nc = min(Threads.nthreads(), max(np, 1))
    buf = zeros(eltype(dP), ntrows, np, nc)
    ranges = [((c - 1)*np ÷ nc + 1):(c*np ÷ nc) for c in 1:nc]
    Threads.@threads :static for c in 1:nc
        f!(view(buf, :, :, c), ranges[c])
    end
    return dropdims(sum(buf; dims=3); dims=3)
end

function _direct_UJ_reverse!(dP, P, np, gdg)
    X, Gm, S, U, J = vpm.X_INDEX, vpm.GAMMA_INDEX, vpm.SIGMA_INDEX, vpm.U_INDEX, vpm.J_INDEX
    dXt = _chunked_pairs!(dP, np, 3) do bt, js
        for j in js
            xj = _col3(P, X, j); G = _col3(P, Gm, j); sgm = P[S, j]
            dxj = zero(xj); dG = zero(G); ds = zero(sgm)
            for i in 1:np
                d = _col3(P, X, i) - xj
                iszero(dot(d, d)) && continue
                dbar, Gbar, sbar = _uj_pair_adj(d, G, sgm, _col3(dP, U, i), _col9(dP, J, i), gdg)
                bt[1, i] += dbar[1]; bt[2, i] += dbar[2]; bt[3, i] += dbar[3]
                dxj -= dbar; dG += Gbar; ds += sbar
            end
            for a in 1:3
                dP[X[a], j] += dxj[a]; dP[Gm[a], j] += dG[a]
            end
            dP[S, j] += ds
        end
    end
    for i in 1:np, a in 1:3
        dP[X[a], i] += dXt[a, i]
    end
    return nothing
end

# Estr_direct, pair (target t, source s), both non-static, t != s:
#   SFS_t += z(r) M (J_t - J_s) Gamma_s,  z = zeta(r/sigma_s)/sigma_s^3,
#   M(J) = J' if transposed else J,  r = |x_s - x_t|.
# With w_t = SFSbar_t, the pair sums factor through per-body vectors:
#   y_t = M(J_t)' w_t, h_s = M(J_s) Gamma_s, zbar = Gamma_s.y_t - h_s.w_t,
#   Mbar summed = w_t (sum_s z Gamma_s)' at t and (sum_t z w_t) Gamma_s' at s.
_M(J, transposed) = transposed ? J' : J
# `blocks`: (targets, sources) index sets whose every pair is in the sum; per
# thread, both sides accumulate in a buffer (a source set can recur in blocks
# on different threads). `zd(q)` = (zeta, dzeta/dq); w = SFSbar (zero at static
# targets); the 9 x np gradients `Jm` (default: the J rows) get cotangents in `dJm`.
_estr_w(dP, P, rows) = [Bool(P[vpm.STATIC_INDEX, t]) ? zero(SVector{3, eltype(P)}) : _col3(dP, rows, t) for t in axes(P, 2)]
function _estr_reverse!(dP, P, zd, transposed, blocks, w = _estr_w(dP, P, vpm.SFS_INDEX),
                        Jm = view(P, vpm.J_INDEX, :), dJm = view(dP, vpm.J_INDEX, :))
    X, Gm, S, ST = vpm.X_INDEX, vpm.GAMMA_INDEX, vpm.SIGMA_INDEX, vpm.STATIC_INDEX
    np = size(P, 2)
    act = [!Bool(P[ST, i]) for i in 1:np]
    y = [_M(_col9(Jm, 1:9, t), transposed)'*w[t] for t in 1:np]
    nc = min(Threads.nthreads(), max(length(blocks), 1))
    buf = zeros(eltype(P), 16, np, nc)
    Threads.@threads :static for c in 1:nc
        bt = view(buf, :, :, c)
        for k in c:nc:length(blocks)
            ts, ss = blocks[k]
            for s in ss
                act[s] || continue
                xs = _col3(P, X, s); G = _col3(P, Gm, s); sgm = P[S, s]
                h = _M(_col9(Jm, 1:9, s), transposed)*G
                dxs = zero(xs); dGy = zero(G); dsg = zero(sgm); b = zero(G)
                for t in ts
                    (t == s || iszero(w[t])) && continue
                    d = xs - _col3(P, X, t); r = sqrt(dot(d, d)); q = r/sgm
                    zq, dzq = zd(q)
                    z = zq/sgm^3
                    zbar = dot(G, y[t]) - dot(h, w[t])
                    b += z*w[t]; dGy += z*y[t]
                    bt[4, t] += z*G[1]; bt[5, t] += z*G[2]; bt[6, t] += z*G[3]
                    dsg -= zbar*(dzq*r/sgm + 3*zq)/sgm^4
                    if r > 0
                        rb = zbar*dzq/(sgm^4*r)
                        dxs += rb*d
                        bt[1, t] -= rb*d[1]; bt[2, t] -= rb*d[2]; bt[3, t] -= rb*d[3]
                    end
                end
                for a in 1:3
                    bt[6 + a, s] += dxs[a]; bt[9 + a, s] += dGy[a]; bt[13 + a, s] += b[a]
                end
                bt[13, s] += dsg
            end
        end
    end
    tot = dropdims(sum(buf; dims=3); dims=3)
    for k in 1:np
        a = SVector(tot[4, k], tot[5, k], tot[6, k]); b = SVector(tot[14, k], tot[15, k], tot[16, k])
        G = _col3(P, Gm, k); Mk = _M(_col9(Jm, 1:9, k), transposed)
        dG = SVector(tot[10, k], tot[11, k], tot[12, k]) - Mk'*b
        dJ = _M(w[k]*a', transposed) - _M(b*G', transposed)
        for i in 1:3
            dP[X[i], k] += tot[i, k] + tot[6 + i, k]; dP[Gm[i], k] += dG[i]
        end
        dP[S, k] += tot[13, k]
        for i in 1:9
            dJm[i, k] += dJ[i]
        end
    end
    return nothing
end
_estr_direct_blocks(np) = (nc = Threads.nthreads(); [(1:np, ((c - 1)*np ÷ nc + 1):(c*np ÷ nc)) for c in 1:nc])

for (f, rev!) in ((vpm._direct_UJ!, (dpf, P, pf) -> _direct_UJ_reverse!(dpf.particles, P, size(P, 2), pf.kernel.g_dgdr)),
                  (vpm.Estr_direct!, (dpf, P, pf) -> _estr_reverse!(dpf.particles, P, _zd(pf.kernel.zeta), pf.transposed, _estr_direct_blocks(size(P, 2)))))
    @eval function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT},
                                                pf::EnzymeCore.Annotation{<:_HostField}) where RT
        P = pf isa EnzymeCore.Const ? nothing : pf.val.particles[:, 1:pf.val.np]   # inputs, before later steps overwrite them
        func.val(pf.val)
        return EnzymeRules.AugmentedReturn(nothing, nothing, P)
    end
    @eval function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT}, P,
                                       pf::EnzymeCore.Annotation{<:_HostField}) where RT
        EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeExt: batched reverse not supported")
        pf isa EnzymeCore.Const || $rev!(pf.dval, P, pf.val)
        return (nothing,)
    end
end

# ------------------------------------------------------------------------------
# Two-level dynamic SFS, analytic core-scaling derivatives (`dsigma_direct!`,
# gaussianerf). With q = r/sigma_j and G = q g'(q) = sqrt(2/pi) q^3 e^{-q^2/2},
# the pair's dJ/dalpha has the J pair's form with (A, B) -> (Ah, Bh):
#   dJ_i = sum_j Ah c d' + Bh E(Gamma_j),  Ah = q^2 G/r^2,  Bh = k G/r^3;
#   M[1:3]_i = op(dJ_i) Gamma_i;  M[4:6]_i = the Estr sum with zeta -> zeta (q^2 - 3)
#   on J plus the Estr sum with zeta on dJ (both written, not accumulated).
# Reverse: the two Estr reverses, then op' for M[1:3], then one pair loop for dJ.
# ------------------------------------------------------------------------------
_zd_dsigma(q) = (z = vpm.zeta_gauserf(q); (z*(q*q - 3), z*q*(5 - q*q)))

# reverse of one dJ pair: (dbar, Gammabar, sigmabar) from dJbar = s
@inline function _dsigma_pair_adj(d, G, sgm, s)
    k = vpm.const4
    r2 = dot(d, d); r = sqrt(r2); r3 = r2*r; is2 = 1/(sgm*sgm)
    e = sqrt(2/pi)*exp(-r2*is2/2)/(sgm*sgm*sgm)
    Ah = e*r3*is2; Bh = k*e
    c = (-k/r3) * cross(d, G)
    sd = s*d
    tau = SVector(s[2,3] - s[3,2], s[3,1] - s[1,3], s[1,2] - s[2,1])
    Q = dot(c, sd); T = dot(tau, G)
    v = Ah*sd
    dbar = (-k/r3)*cross(G, v) - (3*dot(v, c)/r2)*d + Ah*(s'*c) + ((Q*Ah*(3/r - r*is2) - T*Bh*r*is2)/r)*d
    Gbar = (-k/r3)*cross(v, d) + Bh*tau
    sbar = (Q*Ah*(r2*is2 - 5) + T*Bh*(r2*is2 - 3))/sgm
    return dbar, Gbar, sbar
end

function _dsigma_reverse!(dP, P, dJ, transposed)
    X, Gm, S, M = vpm.X_INDEX, vpm.GAMMA_INDEX, vpm.SIGMA_INDEX, vpm.M_INDEX
    np = size(P, 2)
    act = [!Bool(P[vpm.STATIC_INDEX, i]) for i in 1:np]
    wL = _estr_w(dP, P, M[1:3]); wE = _estr_w(dP, P, M[4:6])
    for i in 1:np, a in 1:6
        act[i] && (dP[M[a], i] = 0)        # overwritten outputs
    end
    ddJ = zeros(eltype(P), 9, np)
    blocks = _estr_direct_blocks(np)
    _estr_reverse!(dP, P, _zd_dsigma, transposed, blocks, wE)
    _estr_reverse!(dP, P, _zd(vpm.zeta_gauserf), transposed, blocks, wE, dJ, ddJ)
    for i in 1:np
        act[i] || continue
        G = _col3(P, Gm, i); dG = _M(_col9(dJ, 1:9, i), transposed)'*wL[i]; dD = _M(wL[i]*G', transposed)
        for a in 1:3; dP[Gm[a], i] += dG[a]; end
        for a in 1:9; ddJ[a, i] += dD[a]; end
    end
    dXt = _chunked_pairs!(dP, np, 3) do bt, js
        for j in js
            sgm = P[S, j]
            sgm > 0 || continue
            xj = _col3(P, X, j); G = _col3(P, Gm, j)
            dxj = zero(xj); dG = zero(G); ds = zero(sgm)
            for i in 1:np
                (act[i] && i != j) || continue
                d = _col3(P, X, i) - xj
                iszero(dot(d, d)) && continue
                dbar, Gbar, sbar = _dsigma_pair_adj(d, G, sgm, _col9(ddJ, 1:9, i))
                bt[1, i] += dbar[1]; bt[2, i] += dbar[2]; bt[3, i] += dbar[3]
                dxj -= dbar; dG += Gbar; ds += sbar
            end
            for a in 1:3
                dP[X[a], j] += dxj[a]; dP[Gm[a], j] += dG[a]
            end
            dP[S, j] += ds
        end
    end
    for i in 1:np, a in 1:3
        dP[X[a], i] += dXt[a, i]
    end
    return nothing
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm.dsigma_direct!)}, ::Type{RT},
                                      pf::EnzymeCore.Annotation{<:_HostField}) where RT
    pf isa EnzymeCore.Const && return EnzymeRules.AugmentedReturn(func.val(pf.val), nothing, nothing)
    P = pf.val.particles[:, 1:pf.val.np]
    return EnzymeRules.AugmentedReturn(nothing, nothing, (P, vpm._dsigma_direct!(pf.val)))
end
function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm.dsigma_direct!)}, ::Type{RT}, tape,
                             pf::EnzymeCore.Annotation{<:_HostField}) where RT
    EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeExt: batched reverse not supported")
    tape === nothing || _dsigma_reverse!(pf.dval.particles, tape[1], tape[2], pf.val.transposed)
    return (nothing,)
end

# ------------------------------------------------------------------------------
# FMM adjoint (host octree `fmm!`, via `_fmm_UJ!`): the reverse pass is two
# real-valued FMM calls over the same particles with the primal's tree settings
# (fixed expansion order, no error-driven order):
#  1. sources at the primal targets i: vortex strength Ubar_i plus a vortex
#     dipole of moment Jbar_i, i.e. vector potential Ubar_i G + sum_b
#     d/dx_i,b (Jbar_i[:,b] G); at the primal sources j its velocity is
#     Gammabar_j and its gradient H gives xbar_j = H' Gamma_j;
#  2. the primal sources again, evaluated at the targets with the velocity
#     gradient J and its derivative T: xbar_i = J' Ubar_i + Jbar_i : T.
# Far field: singular kernel, as in the primal's far field. Near-field blocks
# evaluate the exact regularized pair adjoint (`_uj_pair_adj`) instead, so
# sigmabar comes from the near field only, as sigma enters the primal.
# ------------------------------------------------------------------------------
const fmm = vpm.fmm

struct _AdjSrc{TF,G}          # call 1 sources: x_i, radius, [Ubar_i; Jbar_i]; P for Gamma_j, sigma_j
    P::Matrix{TF}; W::Matrix{TF}; rad::Vector{TF}; gdg::G
end
struct _AdjTgt{TF}            # call 1 targets: x_j; accumulates into dP
    P::Matrix{TF}; dP::Matrix{TF}
end
struct _PrimSrc{TF,G}         # call 2 sources: x_j, radius, Gamma_j, sigma_j; W for Ubar_i, Jbar_i
    P::Matrix{TF}; W::Matrix{TF}; rad::Vector{TF}; gdg::G
end
struct _PrimTgt{TF}           # call 2 targets: x_i; accumulates into dP
    P::Matrix{TF}; W::Matrix{TF}; dP::Matrix{TF}
end
const _AdjSys = Union{_AdjSrc, _AdjTgt, _PrimSrc, _PrimTgt}

Base.eltype(s::_AdjSys) = eltype(s.P)
fmm.get_n_bodies(s::_AdjSys) = size(s.P, 2)
fmm.get_position(s::_AdjSys, i) = _col3(s.P, vpm.X_INDEX, i)
# targets carry their own index (sorted with the positions)
fmm.metadata_per_body(::Union{_AdjTgt, _PrimTgt}) = 1
fmm.metadata_to_buffer!(buf, switch, ib, ::Union{_AdjTgt, _PrimTgt}, i) = (buf[fmm.metadata_index(switch, 1), ib] = i; nothing)
_index(buf, switch, ib) = Int(buf[fmm.metadata_index(switch, 1), ib])
fmm.has_vector_potential(::Union{_AdjSrc, _PrimSrc}) = true
fmm.supports_third_derivative(::_PrimTgt, ::_PrimSrc) = true

fmm.data_per_body(::_AdjSrc) = 16
fmm.strength_dims(::_AdjSrc) = 12
function fmm.source_system_to_buffer!(buf, ib, s::_AdjSrc, i)
    for a in 1:3; buf[a, ib] = s.P[vpm.X_INDEX[a], i]; end
    buf[4, ib] = s.rad[i]
    for k in 1:12; buf[4 + k, ib] = s.W[k, i]; end
end
function fmm.body_to_multipole!(s::_AdjSrc, coeffs, buf, center, bodies_index, harmonics, p)
    for ib in bodies_index
        dx = _col3(buf, 1:3, ib) - center
        fmm.body_to_multipole_point!(fmm.Point{fmm.Vortex}, coeffs, harmonics, dx, _col3(buf, 5:7, ib), p)
        # dipole: d/d(dx_b) of the vortex P2M with strength Jbar[:,b], i.e. the
        # derivative of the regular harmonics at -dx through the same conversion
        ξr, ξi, ηr, ηi, z = fmm.xyz_to_ξηz(-dx)
        fmm.calculate_q!(harmonics, ξr, ξi, ηr, ηi, z, p)
        for b in 1:3
            fmm.source_to_dipole!(harmonics, 2, 1, Float64(b == 1), Float64(b == 2), Float64(b == 3), p)
            fmm.mirrored_source_to_vortex!(coeffs, harmonics, _col3(buf, (5 + 3b):(7 + 3b), ib), 2, -1.0, p)
        end
    end
end
function fmm.direct!(tbuf, tindex, switch, s::_AdjSrc, sbuf, sindex)
    for is in sindex
        xi = _col3(sbuf, 1:3, is); u = _col3(sbuf, 5:7, is); sJ = _col9(sbuf, 8:16, is)
        for it in tindex
            j = _index(tbuf, switch, it)
            d = xi - _col3(tbuf, 1:3, it)
            iszero(dot(d, d)) && continue
            dbar, Gbar, sbar = _uj_pair_adj(d, _col3(s.P, vpm.GAMMA_INDEX, j), s.P[vpm.SIGMA_INDEX, j], u, sJ, s.gdg)
            fmm.set_gradient!(tbuf, switch, it, Gbar)
            for a in 1:3; fmm.set_extra_output!(tbuf, switch, it, a, -dbar[a]); end
            fmm.set_extra_output!(tbuf, switch, it, 4, sbar)
        end
    end
end
function fmm.buffer_to_target_system!(t::_AdjTgt, j, switch, buf, ib)
    Gbar = fmm.get_gradient(buf, switch, ib); H = fmm.get_hessian(buf, switch, ib)
    xbar = H'*_col3(t.P, vpm.GAMMA_INDEX, j)
    for a in 1:3
        t.dP[vpm.GAMMA_INDEX[a], j] += Gbar[a]
        t.dP[vpm.X_INDEX[a], j] += xbar[a] + fmm.get_extra_output(buf, switch, ib, a)
    end
    t.dP[vpm.SIGMA_INDEX, j] += fmm.get_extra_output(buf, switch, ib, 4)
end

fmm.data_per_body(::_PrimSrc) = 8
fmm.strength_dims(::_PrimSrc) = 3
function fmm.source_system_to_buffer!(buf, ib, s::_PrimSrc, i)
    for a in 1:3
        buf[a, ib] = s.P[vpm.X_INDEX[a], i]; buf[4 + a, ib] = s.P[vpm.GAMMA_INDEX[a], i]
    end
    buf[4, ib] = s.rad[i]; buf[8, ib] = s.P[vpm.SIGMA_INDEX, i]
end
fmm.body_to_multipole!(s::_PrimSrc, args...) = fmm.body_to_multipole!(fmm.Point{fmm.Vortex}, s, args...)
function fmm.direct!(tbuf, tindex, switch, s::_PrimSrc, sbuf, sindex)
    for is in sindex
        xj = _col3(sbuf, 1:3, is); G = _col3(sbuf, 5:7, is); sgm = sbuf[8, is]
        for it in tindex
            i = _index(tbuf, switch, it)
            d = _col3(tbuf, 1:3, it) - xj
            iszero(dot(d, d)) && continue
            dbar, _, _ = _uj_pair_adj(d, G, sgm, _col3(s.W, 1:3, i), _col9(s.W, 4:12, i), s.gdg)
            for a in 1:3; fmm.set_extra_output!(tbuf, switch, it, a, dbar[a]); end
        end
    end
end
function fmm.buffer_to_target_system!(t::_PrimTgt, i, switch, buf, ib)
    H = fmm.get_hessian(buf, switch, ib); T = fmm.get_third_derivative(buf, switch, ib)
    u = _col3(t.W, 1:3, i); sJ = _col9(t.W, 4:12, i)
    xbar = H'*u
    for k in 1:3
        acc = xbar[k] + fmm.get_extra_output(buf, switch, ib, k)
        for b in 1:3, a in 1:3
            acc += sJ[a, b]*T[a, b, k]
        end
        t.dP[vpm.X_INDEX[k], i] += acc
    end
end

function _fmm_UJ_reverse!(dP, P, rad, o::vpm.FMM, gdg)
    W = dP[vcat(vpm.U_INDEX, vpm.J_INDEX), 1:size(P, 2)]       # Ubar, Jbar (pass through unchanged)
    kw = (; expansion_order=o.p - 1, leaf_size=max(o.ncrit, o.min_ncrit), multipole_acceptance=o.theta,
          shrink=o.shrink_recenter, recenter=o.shrink_recenter, error_tolerance=nothing, nearfield_device=false,
          scalar_potential=false, silence_warnings=true)
    fmm.fmm!(_AdjTgt(P, dP), _AdjSrc(P, W, rad, gdg); kw..., gradient=true, hessian=true, extra_outputs=4)
    fmm.fmm!(_PrimTgt(P, W, dP), _PrimSrc(P, W, rad, gdg); kw..., gradient=false, hessian=true,
             third_derivative=true, extra_outputs=3)
    return nothing
end

# the trees and lists `_fmm_UJ!` returns for the SFS passes: index data only
EnzymeRules.inactive_type(::Type{<:fmm.Tree}) = true

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm._fmm_UJ!)}, ::Type{RT},
                                      pf::EnzymeCore.Annotation{<:_HostField}, vorticity, verbose) where RT
    tape = nothing
    if !(pf isa EnzymeCore.Const)
        vorticity.val && error("FLOWVPMEnzymeExt: UJ_fmm(...; vorticity=true) has no reverse rule")
        p = pf.val; np = p.np
        buf = zeros(eltype(p), fmm.data_per_body(p), 1)
        rad = [(fmm.source_system_to_buffer!(buf, 1, p, i); buf[4, 1]) for i in 1:np]
        tape = (p.particles[:, 1:np], rad, p.fmm)
    end
    ret = func.val(pf.val, vorticity.val, verbose.val)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? ret : nothing, nothing, tape)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm._fmm_UJ!)}, ::Type{RT}, tape,
                             pf::EnzymeCore.Annotation{<:_HostField}, vorticity, verbose) where RT
    EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeExt: batched reverse not supported")
    pf isa EnzymeCore.Const || _fmm_UJ_reverse!(pf.dval.particles, tape..., pf.val.kernel.g_dgdr)
    return (nothing, nothing, nothing)
end

# Estr over the FMM's direct list (UJ_fmm's SFS pass): the Estr reverse over its leaf pairs
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm.Estr_fmm!)}, ::Type{RT},
                                      tpf::EnzymeCore.Annotation{<:_HostField}, spf::EnzymeCore.Annotation{<:_HostField},
                                      ttree, stree, dlist) where RT
    tape = nothing
    if !(tpf isa EnzymeCore.Const && spf isa EnzymeCore.Const)
        tpf.val === spf.val || error("FLOWVPMEnzymeExt: Estr_fmm! reverse needs one field as target and source")
        tt, st = ttree.val, stree.val
        idx(tree, ib) = tree.sort_index_list[1][tree.branches[ib].bodies_index[1]]
        tape = (tpf.val.particles[:, 1:tpf.val.np], [(idx(tt, T), idx(st, S)) for (T, S) in dlist.val])
    end
    func.val(tpf.val, spf.val, ttree.val, stree.val, dlist.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, tape)
end
function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm.Estr_fmm!)}, ::Type{RT}, tape,
                             tpf::EnzymeCore.Annotation{<:_HostField}, spf::EnzymeCore.Annotation{<:_HostField},
                             ttree, stree, dlist) where RT
    EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeExt: batched reverse not supported")
    tape === nothing || _estr_reverse!(tpf.dval.particles, tape[1], _zd(tpf.val.kernel.zeta), tpf.val.transposed, tape[2])
    return (nothing, nothing, nothing, nothing, nothing)
end


# Many steps with bounded memory (`FLOWVPM.checkpointed_gradient!`): periodic
# checkpoints of the state between steps, one segment under Enzyme at a time.
_ckpt_steps!(pf, step, dt, k) = (for _ in 1:k; step(pf, dt); end; nothing)
_ckpt_tsgm(v::vpm.CoreSpreading) = v.t_sgm
_ckpt_tsgm(v) = nothing
_ckpt_save(pf) = (P = pf.particles[:, 1:pf.np], np = pf.np, nt = pf.nt, t = pf.t, tsgm = _ckpt_tsgm(pf.viscous),
                  H = pf.track_H_chi ? pf.splitting_state.H_chi[1:pf.np] : nothing)
function _ckpt_restore!(pf, s)
    pf.particles[:, 1:s.np] .= s.P
    pf.np = s.np; pf.nt = s.nt; pf.t = s.t
    s.tsgm === nothing || (pf.viscous.t_sgm = s.tsgm)
    s.H === nothing || (pf.splitting_state.H_chi[1:s.np] .= s.H)
    return pf
end
_ckpt_same(a, b) = a.np == b.np && a.nt == b.nt && a.t == b.t && isequal(a.tsgm, b.tsgm) && isequal(a.H, b.H) && isequal(a.P, b.P)

function vpm.checkpointed_gradient!(dpf, objective, pf, dt, nsteps::Integer; every::Integer = max(1, isqrt(nsteps)),
                                    step = vpm.nextstep)
    pf.particles isa Array || throw(ArgumentError("checkpointed_gradient!: host fields only"))
    size(dpf.particles) == size(pf.particles) || throw(DimensionMismatch("dpfield does not match pfield"))
    nseg = cld(nsteps, every)
    len(j) = min(every, nsteps - (j - 1) * every)
    snaps = Vector{Any}(undef, nseg + 1)
    snaps[1] = _ckpt_save(pf)
    recs = [Any[] for _ in 1:nseg]     # each segment's Galerkin reset solutions, replayed below
    for j in 1:nseg
        task_local_storage(() -> _ckpt_steps!(pf, step, dt, len(j)), :FLOWVPM_gal_record, recs[j])
        snaps[j + 1] = _ckpt_save(pf)
    end
    J = EnzymeCore.autodiff(EnzymeCore.ReverseWithPrimal, EnzymeCore.Const(objective), EnzymeCore.Active,
                            EnzymeCore.Duplicated(pf, dpf))[2]
    for j in nseg:-1:1
        _ckpt_restore!(pf, snaps[j])
        task_local_storage(:FLOWVPM_gal_replay, recs[j]) do
            EnzymeCore.autodiff(EnzymeCore.Reverse, EnzymeCore.Const(_ckpt_steps!), EnzymeCore.Const,
                                EnzymeCore.Duplicated(pf, dpf), EnzymeCore.Const(step), EnzymeCore.Const(dt),
                                EnzymeCore.Const(len(j)))
        end
        isempty(recs[j]) || error("checkpointed_gradient!: segment $j re-ran fewer Galerkin resets than the forward run")
        _ckpt_same(_ckpt_save(pf), snaps[j + 1]) ||
            error("checkpointed_gradient!: segment $j recomputed differently from the forward run (state kept outside the checkpoint?)")
        GC.gc()                # the segment's tape is garbage now: keep the peak at one segment
    end
    _ckpt_restore!(pf, snaps[end])
    return J
end

end
