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

end # module
