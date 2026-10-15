#=##############################################################################
# DESCRIPTION
    Implicit-differentiation rule for the core-spreading RBF solve
    (`FLOWVPM.rbf_implicitad`). The solve is wrapped in `ImplicitAD.implicit`
    and an EnzymeRules rule calls ImplicitAD's own ChainRules rrule (no
    Enzyme ChainRulesCore extension or @import_rrule needed, so the trigger is
    the light EnzymeCore), so reverse mode never differentiates the CG iterations. The default dense pieces of ImplicitAD
    (ForwardDiff dr/dy, `\\`, ReverseDiff VJP) are replaced for this residual
    type: dr/dy = A is a matrix-free operator, the adjoint solve is a CG on it
    (A symmetric), and the x-VJP is the analytic O(N^2) pair sum.

    x = [X (3n, particle-major); b = M[7:9] (3n)], y = Gamma (3n),
    r(y, x) = A(X) y - b,  A_ij = zeta(|X_i - X_j|/s0)/s0^3.
=###############################################################################
module FLOWVPMImplicitADExt

import FLOWVPM
import ImplicitAD
import EnzymeCore
import EnzymeCore: EnzymeRules
import ImplicitAD.ChainRulesCore
using LinearAlgebra: dot, norm

const vpm = FLOWVPM
const FD = ImplicitAD.ForwardDiff

struct CSRBFResidual{Tz,R}
    zeta::Tz
    sgm0::R
end

"Matrix-free A(X); symmetric, so `A' === A`."
struct CSRBFOperator{Tz,R,TX}
    zeta::Tz
    sgm0::R
    x::TX          # packed [X; b]; only X (the first 3n entries) is read
    n::Int
    tol::Float64
    itmax::Int
end
Base.adjoint(A::CSRBFOperator) = A

# r = A(X) y - b (forward-mode/FD use; the reverse rule never evaluates it)
function (res::CSRBFResidual)(y, x, p)
    n = length(y) ÷ 3
    T = promote_type(eltype(y), eltype(x))
    r = zeros(T, 3n)
    _apply!(r, res.zeta, res.sgm0, x, y, n)
    for k in 1:3n
        r[k] -= x[3n+k]
    end
    return r
end

# out = A(X) y, y and out particle-major (3n)
function _apply!(out, zeta, s0, x, y, n)
    zs = zeta(zero(s0))/s0^3
    for i in 1:n
        a1 = zs*y[3i-2]; a2 = zs*y[3i-1]; a3 = zs*y[3i]
        for j in 1:n
            j == i && continue
            d1 = x[3i-2]-x[3j-2]; d2 = x[3i-1]-x[3j-1]; d3 = x[3i]-x[3j]
            z = zeta(sqrt(d1*d1 + d2*d2 + d3*d3)/s0)/s0^3
            a1 += z*y[3j-2]; a2 += z*y[3j-1]; a3 += z*y[3j]
        end
        out[3i-2] = a1; out[3i-1] = a2; out[3i] = a3
    end
    return out
end

cs_drdy(res::CSRBFResidual, y, x, p) =
    CSRBFOperator(res.zeta, res.sgm0, x, length(y) ÷ 3, Float64(p[2].tol), max(p[2].itmax, 3*length(y)))

# Adjoint solve A lambda = c: plain CG per vector component (A is SPD), the same
# relative-residual stopping test as rbf_conjugategradient (||r||/||c|| <= tol).
function cs_lsolve(A::CSRBFOperator, c)
    n = A.n
    lam = zeros(Float64, 3n); r = Vector{Float64}(c); p = copy(r); Ap = similar(r)
    rr = [sum(r[3j-3+k]^2 for j in 1:n) for k in 1:3]
    rr0 = copy(rr)
    for it in 1:A.itmax
        all(k -> rr[k] <= A.tol^2*rr0[k], 1:3) && return lam
        _apply!(Ap, A.zeta, A.sgm0, A.x, p, n)
        for k in 1:3
            rr[k] <= A.tol^2*rr0[k] && continue
            pAp = sum(p[3j-3+k]*Ap[3j-3+k] for j in 1:n)
            alpha = rr[k]/pAp
            rrn = 0.0
            for j in 1:n
                lam[3j-3+k] += alpha*p[3j-3+k]
                r[3j-3+k] -= alpha*Ap[3j-3+k]
                rrn += r[3j-3+k]^2
            end
            beta = rrn/rr[k]; rr[k] = rrn
            for j in 1:n
                p[3j-3+k] = r[3j-3+k] + beta*p[3j-3+k]
            end
        end
    end
    all(k -> rr[k] <= A.tol^2*rr0[k], 1:3) ||
        @warn "rbf_implicitad: adjoint CG did not reach tol $(A.tol) in $(A.itmax) iterations" relres = sqrt.(rr ./ rr0)
    return lam
end

# v' dr/dx: b part = -v; X part = sum over pairs of zeta'(rho/s0)/s0^4 (X_i - X_j)/rho (v_i.y_j + v_j.y_i)
function ImplicitAD.vjp(res::CSRBFResidual, y, x, p, v)
    n = length(y) ÷ 3
    s0 = res.sgm0; zeta = res.zeta
    g = zeros(Float64, 6n)
    for i in 1:n, j in (i+1):n
        d1 = x[3i-2]-x[3j-2]; d2 = x[3i-1]-x[3j-1]; d3 = x[3i]-x[3j]
        rho = sqrt(d1*d1 + d2*d2 + d3*d3)
        c = v[3i-2]*y[3j-2] + v[3i-1]*y[3j-1] + v[3i]*y[3j] +
            v[3j-2]*y[3i-2] + v[3j-1]*y[3i-1] + v[3j]*y[3i]
        s = c*FD.derivative(zeta, rho/s0)/(s0^4*rho)
        g[3i-2] += s*d1; g[3i-1] += s*d2; g[3i] += s*d3
        g[3j-2] -= s*d1; g[3j-1] -= s*d2; g[3j] -= s*d3
    end
    for k in 1:3n
        g[3n+k] = -v[k]
    end
    return g
end

# primal: FLOWVPM's own CG on the field (X, b = M[7:9], x0 = Gamma already in place)
function cs_solve(x, p)
    pfield, cs = p
    vpm.rbf_conjugategradient(pfield, cs)
    P = pfield.particles
    return [P[vpm.GAMMA_INDEX[k], j] for k in 1:3, j in 1:pfield.np][:]
end

function vpm.rbf_implicitad(pfield, cs::vpm.CoreSpreading)
    P = pfield.particles
    P isa Array || error("rbf_implicitad: host fields only")
    np = pfield.np
    for j in 1:np
        iszero(P[vpm.STATIC_INDEX, j]) || error("rbf_implicitad: static particles are not supported")
    end
    xin = zeros(eltype(P), 6np)
    for j in 1:np, k in 1:3
        xin[3(j-1)+k] = P[vpm.X_INDEX[k], j]
        xin[3np+3(j-1)+k] = P[vpm.M_INDEX[6+k], j]
    end
    y = cs_implicit(xin, pfield, cs)
    for j in 1:np, k in 1:3
        P[vpm.GAMMA_INDEX[k], j] = y[3(j-1)+k]
    end
    return nothing
end

# Gamma(x) = the RBF solution; primal is ImplicitAD.implicit (= cs_solve without AD)
cs_implicit(xin, pfield, cs) = ImplicitAD.implicit(cs_solve, CSRBFResidual(pfield.kernel.zeta, cs.sgm0), xin,
                                                    (pfield, cs); drdy=cs_drdy, lsolve=cs_lsolve)

# Enzyme: run ImplicitAD's rrule for _implicit (primal solve + implicit pullback)
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(cs_implicit)}, ::Type{RT},
                                      xin::EnzymeCore.Annotation, pf::EnzymeCore.Annotation,
                                      cs::EnzymeCore.Annotation) where RT
    pfield = pf.val; c = cs.val
    y, pb = ChainRulesCore.rrule(ImplicitAD._implicit, cs_solve, CSRBFResidual(pfield.kernel.zeta, c.sgm0),
                                 xin.val, (pfield, c), cs_drdy, cs_lsolve)
    dy = zero(y)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? y : nothing,
                                       EnzymeRules.needs_shadow(config) ? dy : nothing, (pb, dy))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(cs_implicit)}, ::Type{RT}, tape,
                             xin::EnzymeCore.Annotation, pf::EnzymeCore.Annotation,
                             cs::EnzymeCore.Annotation) where RT
    pb, dy = tape
    if xin isa EnzymeCore.Duplicated
        xin.dval .+= pb(dy)[4]
    end
    if pf isa EnzymeCore.Duplicated   # slots the CG overwrote (scratch, vorticity): pre-values have no influence
        dP = pf.dval.particles
        for j in 1:pf.val.np
            for k in vpm.GAMMA_INDEX; dP[k, j] = 0; end
            for k in vpm.M_INDEX[1]:vpm.M_INDEX[6]; dP[k, j] = 0; end
            for k in vpm.VORTICITY_INDEX; dP[k, j] = 0; end
        end
    end
    dy .= 0
    return (nothing, nothing, nothing)
end

end
