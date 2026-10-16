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
    c, sys, x = vpm._gal_solve_sys(Y.val, v.val, Xo.val, vo.val, Go.val, x0.val, sol.val)
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
function _estr_reverse!(dP, P, np, zeta, transposed)
    X, Gm, S, J, SF, ST = vpm.X_INDEX, vpm.GAMMA_INDEX, vpm.SIGMA_INDEX, vpm.J_INDEX, vpm.SFS_INDEX, vpm.STATIC_INDEX
    act = [!Bool(P[ST, i]) for i in 1:np]
    w = [act[t] ? _col3(dP, SF, t) : zero(SVector{3, eltype(P)}) for t in 1:np]
    y = [_M(_col9(P, J, t), transposed)'*w[t] for t in 1:np]
    bufT = _chunked_pairs!(dP, np, 6) do bt, ss
        for s in ss
            act[s] || continue
            xs = _col3(P, X, s); G = _col3(P, Gm, s); sgm = P[S, s]; Ms = _M(_col9(P, J, s), transposed)
            h = Ms*G
            dxs = zero(xs); dG = zero(G); dsg = zero(sgm); b = zero(G)
            for t in 1:np
                (t == s || iszero(w[t])) && continue
                d = xs - _col3(P, X, t); r = sqrt(dot(d, d)); q = r/sgm
                zq, dzq = _zeta_dzeta(zeta, q)
                z = zq/sgm^3
                zbar = dot(G, y[t]) - dot(h, w[t])
                b += z*w[t]; dG += z*y[t]
                bt[4, t] += z*G[1]; bt[5, t] += z*G[2]; bt[6, t] += z*G[3]
                dsg -= zbar*(dzq*r/sgm + 3*zq)/sgm^4
                if r > 0
                    rb = zbar*dzq/(sgm^4*r)
                    dxs += rb*d
                    bt[1, t] -= rb*d[1]; bt[2, t] -= rb*d[2]; bt[3, t] -= rb*d[3]
                end
            end
            dG -= Ms'*b
            dJs = _M(b*G', transposed)
            for a in 1:3
                dP[X[a], s] += dxs[a]; dP[Gm[a], s] += dG[a]
            end
            dP[S, s] += dsg
            for k in 1:9
                dP[J[k], s] -= dJs[k]
            end
        end
    end
    for t in 1:np
        dJt = _M(w[t]*SVector(bufT[4, t], bufT[5, t], bufT[6, t])', transposed)
        for a in 1:3
            dP[X[a], t] += bufT[a, t]
        end
        for k in 1:9
            dP[J[k], t] += dJt[k]
        end
    end
    return nothing
end

for (f, rev!) in ((vpm._direct_UJ!, (dpf, P, pf) -> _direct_UJ_reverse!(dpf.particles, P, size(P, 2), pf.kernel.g_dgdr)),
                  (vpm.Estr_direct!, (dpf, P, pf) -> _estr_reverse!(dpf.particles, P, size(P, 2), pf.kernel.zeta, pf.transposed)))
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
    for j in 1:nseg
        _ckpt_steps!(pf, step, dt, len(j))
        snaps[j + 1] = _ckpt_save(pf)
    end
    J = EnzymeCore.autodiff(EnzymeCore.ReverseWithPrimal, EnzymeCore.Const(objective), EnzymeCore.Active,
                            EnzymeCore.Duplicated(pf, dpf))[2]
    for j in nseg:-1:1
        _ckpt_restore!(pf, snaps[j])
        EnzymeCore.autodiff(EnzymeCore.Reverse, EnzymeCore.Const(_ckpt_steps!), EnzymeCore.Const,
                            EnzymeCore.Duplicated(pf, dpf), EnzymeCore.Const(step), EnzymeCore.Const(dt),
                            EnzymeCore.Const(len(j)))
        _ckpt_same(_ckpt_save(pf), snaps[j + 1]) ||
            error("checkpointed_gradient!: segment $j recomputed differently from the forward run (state kept outside the checkpoint?)")
        GC.gc()                # the segment's tape is garbage now: keep the peak at one segment
    end
    _ckpt_restore!(pf, snaps[end])
    return J
end

end
