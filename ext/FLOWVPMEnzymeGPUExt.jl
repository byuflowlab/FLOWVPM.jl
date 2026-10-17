#=##############################################################################
# DESCRIPTION
    Enzyme reverse-mode rules for device (GPU) fields: each device operation of a
    time step gets its adjoint as KernelAbstractions kernels, so Enzyme never
    differentiates a kernel (KA's own Enzyme support covers only some backends).
    The pair math is shared with the host rules (src/FLOWVPM_adjoint.jl).
=###############################################################################
module FLOWVPMEnzymeGPUExt

import FLOWVPM
import EnzymeCore
import EnzymeCore: EnzymeRules
using GPUArraysCore: AbstractGPUArray
using KernelAbstractions
using StaticArrays: SVector, SMatrix

const vpm = FLOWVPM
const KA = KernelAbstractions

const _DevField = vpm.ParticleField{<:AbstractFloat, <:Any, <:Any, <:Any, <:Any, <:vpm._KnownKernel,
                                    <:Any, <:Any, <:Any, <:AbstractGPUArray}

#------- direct U, J (gpu_direct!) -------#
# Reverse of U_i += sum_j U(i, j), J_i += sum_j J(i, j) over the pairs i != j: the
# cotangents of U and J (rows of dP) pass through (accumulated outputs); the source
# side of every pair goes to x_j, Gamma_j, sigma_j (one thread per source), the
# target side to x_i (one thread per target). P: rows 1:7 before the call.

@inline _dev3(A, r0, i) = SVector(A[r0, i], A[r0 + 1, i], A[r0 + 2, i])
@inline _dev9(A, r0, i) = SMatrix{3,3}(A[r0, i], A[r0 + 1, i], A[r0 + 2, i], A[r0 + 3, i], A[r0 + 4, i],
                                       A[r0 + 5, i], A[r0 + 6, i], A[r0 + 7, i], A[r0 + 8, i])

@kernel function ka_uj_adj_source_kernel!(dP, @Const(P), n::Int32, gdg, u0::Int32, j0::Int32)
    j = @index(Global)
    @inbounds if j <= n
        xj = _dev3(P, 1, j); G = _dev3(P, 4, j); sgm = P[7, j]
        dx = zero(xj); dG = zero(G); ds = zero(sgm)
        if abs(sgm) > zero(sgm)
            i = Int32(1)
            while i <= n
                d = _dev3(P, 1, i) - xj
                if !iszero(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
                    dbar, Gbar, sbar = vpm._uj_pair_adj(d, G, sgm, _dev3(dP, u0, i), _dev9(dP, j0, i), gdg)
                    dx -= dbar; dG += Gbar; ds += sbar
                end
                i += Int32(1)
            end
        end
        for a in 1:3
            dP[a, j] += dx[a]; dP[3 + a, j] += dG[a]
        end
        dP[7, j] += ds
    end
end

@kernel function ka_uj_adj_target_kernel!(dX, @Const(dP), @Const(P), n::Int32, gdg, u0::Int32, j0::Int32)
    i = @index(Global)
    @inbounds if i <= n
        xi = _dev3(P, 1, i); u = _dev3(dP, u0, i); s = _dev9(dP, j0, i)
        acc = zero(xi)
        j = Int32(1)
        while j <= n
            sgm = P[7, j]
            d = xi - _dev3(P, 1, j)
            if abs(sgm) > zero(sgm) && !iszero(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
                dbar, _, _ = vpm._uj_pair_adj(d, _dev3(P, 4, j), sgm, u, s, gdg)
                acc += dbar
            end
            j += Int32(1)
        end
        dX[1, i] = acc[1]; dX[2, i] = acc[2]; dX[3, i] = acc[3]
    end
end

function _direct_UJ_reverse_dev!(dP, P, n, gdg)
    backend = KA.get_backend(dP)
    u0 = Int32(first(vpm.U_INDEX)); j0 = Int32(first(vpm.J_INDEX))
    dX = KA.zeros(backend, eltype(dP), 3, n)
    ka_uj_adj_target_kernel!(backend, 256)(dX, dP, P, Int32(n), gdg, u0, j0; ndrange = n)
    ka_uj_adj_source_kernel!(backend, 256)(dP, P, Int32(n), gdg, u0, j0; ndrange = n)
    view(dP, 1:3, 1:n) .+= dX
    KA.synchronize(backend)
    return nothing
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm.gpu_direct!)}, ::Type{RT},
                                      pf::EnzymeCore.Annotation{<:_DevField}) where RT
    P = pf isa EnzymeCore.Const ? nothing : pf.val.particles[1:7, 1:pf.val.np]   # inputs, before later steps overwrite them
    func.val(pf.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, P)
end
function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm.gpu_direct!)}, ::Type{RT}, P,
                             pf::EnzymeCore.Annotation{<:_DevField}) where RT
    EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeGPUExt: batched reverse not supported")
    pf isa EnzymeCore.Const || _direct_UJ_reverse_dev!(pf.dval.particles, P, size(P, 2), pf.val.kernel.g_dgdr)
    return (nothing,)
end

#------- SFS stretching Estr (gpu_estr_direct!) -------#
# Reverse of SFS_t += sum_s z(r) M(J_t - J_s) Gamma_s over non-static t != s
# (src/FLOWVPM_adjoint.jl, ext/FLOWVPMEnzymeExt.jl `_estr_reverse!` for the
# derivation): one thread per source (B: its x, Gamma, sigma cotangents and
# b = sum_t z w_t), one per target (C: its x cotangent and a = sum_s z Gamma_s),
# then one per particle for the J and Gamma terms. w = SFSbar (zero at static targets).

@inline _devstatic(P, st, i) = !iszero(P[st, i])

@kernel function ka_estr_adj_source_kernel!(B, @Const(P), @Const(dP), n::Int32, zeta, transposed::Bool,
                                            st::Int32, j0::Int32, f0::Int32)
    s = @index(Global)
    @inbounds if s <= n
        xs = _dev3(P, 1, s); G = _dev3(P, 4, s); sgm = P[7, s]
        dxs = zero(xs); dGy = zero(G); dsg = zero(sgm); b = zero(G)
        if !_devstatic(P, st, s)
            h = vpm._M(_dev9(P, j0, s), transposed) * G
            t = Int32(1)
            while t <= n
                w = _dev3(dP, f0, t)
                if t != s && !_devstatic(P, st, t) && !iszero(w)
                    y = vpm._M(_dev9(P, j0, t), transposed)' * w
                    d = xs - _dev3(P, 1, t); r = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
                    zq, dzq = vpm._zeta_dzeta(zeta, r/sgm)
                    z = zq/(sgm*sgm*sgm)
                    zbar = (G[1]*y[1] + G[2]*y[2] + G[3]*y[3]) - (h[1]*w[1] + h[2]*w[2] + h[3]*w[3])
                    b += z*w; dGy += z*y
                    dsg -= zbar*(dzq*r/sgm + 3*zq)/(sgm*sgm*sgm*sgm)
                    if r > zero(r)
                        dxs += (zbar*dzq/(sgm*sgm*sgm*sgm*r))*d
                    end
                end
                t += Int32(1)
            end
        end
        for a in 1:3
            B[a, s] = dxs[a]; B[3 + a, s] = dGy[a]; B[7 + a, s] = b[a]
        end
        B[7, s] = dsg
    end
end

@kernel function ka_estr_adj_target_kernel!(C, @Const(P), @Const(dP), n::Int32, zeta, transposed::Bool,
                                            st::Int32, j0::Int32, f0::Int32)
    t = @index(Global)
    @inbounds if t <= n
        xt = _dev3(P, 1, t); w = _dev3(dP, f0, t)
        dxt = zero(xt); a = zero(xt)
        if !_devstatic(P, st, t) && !iszero(w)
            y = vpm._M(_dev9(P, j0, t), transposed)' * w
            s = Int32(1)
            while s <= n
                if s != t && !_devstatic(P, st, s)
                    G = _dev3(P, 4, s); sgm = P[7, s]
                    h = vpm._M(_dev9(P, j0, s), transposed) * G
                    d = _dev3(P, 1, s) - xt; r = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
                    zq, dzq = vpm._zeta_dzeta(zeta, r/sgm)
                    z = zq/(sgm*sgm*sgm)
                    a += z*G
                    if r > zero(r)
                        zbar = (G[1]*y[1] + G[2]*y[2] + G[3]*y[3]) - (h[1]*w[1] + h[2]*w[2] + h[3]*w[3])
                        dxt -= (zbar*dzq/(sgm*sgm*sgm*sgm*r))*d
                    end
                end
                s += Int32(1)
            end
        end
        for k in 1:3
            C[k, t] = dxt[k]; C[3 + k, t] = a[k]
        end
    end
end

@kernel function ka_estr_adj_finish_kernel!(dP, @Const(P), @Const(B), @Const(C), n::Int32, transposed::Bool,
                                            st::Int32, j0::Int32, f0::Int32)
    k = @index(Global)
    @inbounds if k <= n
        G = _dev3(P, 4, k); Mk = vpm._M(_dev9(P, j0, k), transposed)
        w = _devstatic(P, st, k) ? zero(G) : _dev3(dP, f0, k)
        a = _dev3(C, 4, k); b = _dev3(B, 8, k)
        dG = _dev3(B, 4, k) - Mk'*b
        dJ = vpm._M(w*a', transposed) - vpm._M(b*G', transposed)
        for i in 1:3
            dP[i, k] += B[i, k] + C[i, k]; dP[3 + i, k] += dG[i]
        end
        dP[7, k] += B[7, k]
        for i in 1:9
            dP[j0 + i - 1, k] += dJ[i]
        end
    end
end

function _estr_reverse_dev!(dP, P, n, zeta, transposed)
    backend = KA.get_backend(dP); T = eltype(dP)
    st = Int32(vpm.STATIC_INDEX); j0 = Int32(first(vpm.J_INDEX)); f0 = Int32(first(vpm.SFS_INDEX))
    B = KA.zeros(backend, T, 10, n); C = KA.zeros(backend, T, 6, n)
    ka_estr_adj_source_kernel!(backend, 256)(B, P, dP, Int32(n), zeta, transposed, st, j0, f0; ndrange = n)
    ka_estr_adj_target_kernel!(backend, 256)(C, P, dP, Int32(n), zeta, transposed, st, j0, f0; ndrange = n)
    ka_estr_adj_finish_kernel!(backend, 256)(dP, P, B, C, Int32(n), transposed, st, j0, f0; ndrange = n)
    KA.synchronize(backend)
    return nothing
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm.gpu_estr_direct!)}, ::Type{RT},
                                      pf::EnzymeCore.Annotation{<:_DevField}) where RT
    P = pf isa EnzymeCore.Const ? nothing : pf.val.particles[:, 1:pf.val.np]
    func.val(pf.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, P)
end
function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm.gpu_estr_direct!)}, ::Type{RT}, P,
                             pf::EnzymeCore.Annotation{<:_DevField}) where RT
    EnzymeRules.width(config) == 1 || error("FLOWVPMEnzymeGPUExt: batched reverse not supported")
    pf isa EnzymeCore.Const || _estr_reverse_dev!(pf.dval.particles, P, size(P, 2), pf.val.kernel.zeta, pf.val.transposed)
    return (nothing,)
end

#------- row zeroing (U, vorticity, J, PSE; SFS; the RK3 M storage) -------#
# P[rows, i] = 0 for non-static i: the reverse zeroes the same rows of the
# cotangent (the old values do not reach the output). Rules so that Enzyme never
# sees the masked row-view broadcasts (illegal type analysis on CUDA).

@kernel function ka_zero_rows_adj_kernel!(dP, @Const(P), np::Int32, r0::Int32, r1::Int32, st::Int32, skipstatic::Bool)
    i = @index(Global)
    @inbounds if i <= np && !(skipstatic && !iszero(P[st, i]))
        for r in r0:r1
            dP[r, i] = zero(eltype(dP))
        end
    end
end
function _zero_rows_adj!(dP, P, np, rows; skipstatic = true)
    np == 0 && return nothing
    backend = KA.get_backend(dP)
    ka_zero_rows_adj_kernel!(backend, 256)(dP, P, Int32(np), Int32(first(rows)), Int32(last(rows)),
                                           Int32(vpm.STATIC_INDEX), skipstatic; ndrange = np)
    KA.synchronize(backend)
    return nothing
end

# (U_prev = |U| of every particle is bookkeeping read by no step: no derivative)
for (f, rows, skip) in ((vpm._reset_particles_broadcast!, first(vpm.U_INDEX):last(vpm.PSE_INDEX), true),
                        (vpm._reset_particles_sfs_broadcast!, vpm.SFS_INDEX, true),
                        (vpm._reset_M_storage!, vpm.M_INDEX, true),
                        (vpm._update_U_prev_broadcast!, vpm.U_PREV_INDEX:vpm.U_PREV_INDEX, false))
    @eval function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT},
                                                pf::EnzymeCore.Annotation{<:_DevField}) where RT
        func.val(pf.val)
        return EnzymeRules.AugmentedReturn(nothing, nothing, pf.val.np)
    end
    @eval function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT}, np,
                                       pf::EnzymeCore.Annotation{<:_DevField}) where RT
        pf isa EnzymeCore.Const || _zero_rows_adj!(pf.dval.particles, pf.val.particles, np, $rows; skipstatic = $skip)
        return (nothing,)
    end
end

#------- per-particle updates (generic column VJP) -------#
# A fused device update maps each non-static column through a pure function
# y = fcol(x, args...) of the rows `rin`, written to the rows `rout` (a subset of
# `rin`). Reverse, per particle: xbar = (dfcol/dx)' ybar by Enzyme inside the
# kernel; the written rows' cotangents are replaced, the read-only rows' added to.

@inline function _col_vjp(fcol, x, yb, args...)
    h(x) = sum(yb .* fcol(x, args...))
    return EnzymeCore.autodiff_deferred(EnzymeCore.Reverse, EnzymeCore.Const(h), EnzymeCore.Active,
                                        EnzymeCore.Active(x))[1][1]
end

@kernel function ka_col_adj_kernel!(dP, @Const(P0), np::Int32, st::Int32, skipstatic::Bool, fcol, rin, rout, args)
    i = @index(Global)
    @inbounds if i <= np && !(skipstatic && !iszero(P0[st, i]))
        x = vpm._gather_col(P0, i, rin); yb = vpm._gather_col(dP, i, rout)
        xb = _col_vjp(fcol, x, yb, args...)
        for k in 1:length(rout)
            dP[rout[k], i] = zero(eltype(dP))
        end
        for k in 1:length(rin)
            dP[rin[k], i] += xb[k]
        end
    end
end
function _col_adj!(dP, P0, np, fcol, rin, rout, args; skipstatic = true)
    np == 0 && return nothing
    backend = KA.get_backend(dP)
    ka_col_adj_kernel!(backend, 256)(dP, P0, Int32(np), Int32(vpm.STATIC_INDEX), skipstatic, fcol, rin, rout, args;
                                     ndrange = np)
    KA.synchronize(backend)
    return nothing
end

# zero adjoints for the Active scalar arguments a rule does not differentiate
_no_scalar_adjoints(args...) = (nothing, map(x -> x isa EnzymeCore.Active ? EnzymeCore.make_zero(x.val) : nothing, args)...)

# RK3 register update (reformulated VPM). The scheme constants (a, b, dt, the
# freestream, f, g, zeta0) carry no derivative here.
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm.update_particle_states_broadcast_reformulated!)},
                                      ::Type{RT}, pf::EnzymeCore.Annotation{<:_DevField}, a, b, dt, Uinf, f, g, zeta0) where RT
    P0 = pf isa EnzymeCore.Const ? nothing : pf.val.particles[:, 1:pf.val.np]
    func.val(pf.val, a.val, b.val, dt.val, Uinf.val, f.val, g.val, zeta0.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, P0)
end
function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm.update_particle_states_broadcast_reformulated!)},
                             ::Type{RT}, P0, pf::EnzymeCore.Annotation{<:_DevField}, a, b, dt, Uinf, f, g, zeta0) where RT
    if !(pf isa EnzymeCore.Const)
        R = eltype(P0)
        sc = vpm._rk3_consts(R, a.val, b.val, dt.val, Uinf.val, f.val, g.val, zeta0.val)
        _col_adj!(pf.dval.particles, P0, size(P0, 2), vpm._rk3_reformulated_col, vpm._RK3_IN, vpm._RK3_OUT,
                  (pf.val.transposed, sc))
    end
    return _no_scalar_adjoints(a, b, dt, Uinf, f, g, zeta0)
end

# the other per-particle updates, each: (function, its column map, rows read, rows
# written, the map's arguments from the call's, whether static particles are skipped)
for (f, fcol, rin, rout, margs, skip) in (
        (vpm._euler_broadcast_reformulated!, vpm._euler_reformulated_col, vpm._EULER_IN, vpm._EULER_OUT,
         (pf, R, dt, Uinf, f, g, zeta0) -> (pf.val.transposed, vpm._euler_consts(R, dt, Uinf, f, g, zeta0)), true),
        (vpm._corespreading_rk3_broadcast!, vpm._cs_rk3_col, vpm._CS_RK3_ROWS, vpm._CS_RK3_ROWS,
         (pf, R, nu, dt, aux1, aux2) -> ((R(dt) * 2 * R(nu), R(aux1), R(aux2)),), true),
        (vpm._corespreading_euler_broadcast!, vpm._cs_euler_col, vpm._CS_EULER_ROWS, vpm._CS_EULER_ROWS,
         (pf, R, nu, dt) -> (2 * R(nu) * R(dt),), true))
    @eval function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT},
                                                pf::EnzymeCore.Annotation{<:_DevField}, args...) where RT
        P0 = pf isa EnzymeCore.Const ? nothing : pf.val.particles[:, 1:pf.val.np]
        func.val(pf.val, map(a -> a.val, args)...)
        return EnzymeRules.AugmentedReturn(nothing, nothing, P0)
    end
    @eval function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof($f)}, ::Type{RT}, P0,
                                       pf::EnzymeCore.Annotation{<:_DevField}, args...) where RT
        pf isa EnzymeCore.Const ||
            _col_adj!(pf.dval.particles, P0, size(P0, 2), $fcol, $rin, $rout,
                      $margs(pf, eltype(P0), map(a -> a.val, args)...); skipstatic = $skip)
        return _no_scalar_adjoints(args...)
    end
end

# relaxation: every particle (static ones too, as the kernel)
for (fn, corrected) in ((vpm.relax_pedrizzetti, false), (vpm.relax_correctedpedrizzetti, true))
    @eval function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(vpm._relax_broadcast!)}, ::Type{RT},
                                                fa::EnzymeCore.Const{typeof($fn)}, rlxf, pf::EnzymeCore.Annotation{<:_DevField}) where RT
        P0 = pf isa EnzymeCore.Const ? nothing : pf.val.particles[:, 1:pf.val.np]
        func.val(fa.val, rlxf.val, pf.val)
        return EnzymeRules.AugmentedReturn(nothing, nothing, P0)
    end
    @eval function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(vpm._relax_broadcast!)}, ::Type{RT}, P0,
                                       fa::EnzymeCore.Const{typeof($fn)}, rlxf, pf::EnzymeCore.Annotation{<:_DevField}) where RT
        pf isa EnzymeCore.Const ||
            _col_adj!(pf.dval.particles, P0, size(P0, 2), vpm._relax_col, vpm._RELAX_IN, vpm._RELAX_OUT,
                      (eltype(P0)(rlxf.val), Val($corrected)); skipstatic = false)
        return (nothing, rlxf isa EnzymeCore.Active ? zero(rlxf.val) : nothing, nothing)
    end
end

end # module
