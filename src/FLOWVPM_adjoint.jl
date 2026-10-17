#=##############################################################################
# DESCRIPTION
    Pair adjoints of the direct sums, shared by the reverse-mode rules on the host
    (ext/FLOWVPMEnzymeExt.jl) and on the device (ext/FLOWVPMEnzymeGPUExt.jl).
    Pair (target i, source j), d = x_i - x_j, r = |d|, q = r/sigma_j:
      U = g c,  J = A c d' + B E(Gamma),  c = -k (d x Gamma)/r^3,  k = 1/4pi,
      A = g'/(sigma r) - 3g/r^2,  B = -k g/r^3,  E_ab = eps_abc Gamma_c.
=###############################################################################

# second derivative of g from g' = dgdr (q > 0), per kernel (the rules apply only to these kernels)
_d2g(::typeof(g_dgdr_gauserf), q, g1) = g1*(2 - q*q)/q
_d2g(::typeof(g_dgdr_gaus), q, g1) = g1*(2 - 3*q*q*q)/q
_d2g(::typeof(g_dgdr_wnklmns), q, g1) = g1*(2 - 5*q*q)/(q*(q*q + 1))
_d2g(::typeof(g_dgdr_sing), q, g1) = zero(q)

# zeta and its derivative, per kernel
_zeta_dzeta(f::typeof(zeta_gauserf), q) = (z = f(q); (z, -q*z))
_zeta_dzeta(f::typeof(zeta_gaus), q) = (z = f(q); (z, -3*q*q*z))
_zeta_dzeta(f::typeof(zeta_wnklmns), q) = (z = f(q); (z, -7*q*z/(q*q + 1)))
_zeta_dzeta(f::typeof(zeta_sing), q) = (f(q), zero(q))

# the Jacobian as Estr uses it
_M(J, transposed) = transposed ? J' : J

const _KnownKernel = Kernel{<:Union{typeof.((zeta_gauserf, zeta_gaus, zeta_wnklmns, zeta_sing))...},
                            <:Any, <:Any,
                            <:Union{typeof.((g_dgdr_gauserf, g_dgdr_gaus, g_dgdr_wnklmns, g_dgdr_sing))...}}

# reverse of one pair: (dbar, Gammabar, sigmabar) from Ubar = u, Jbar = s
@inline function _uj_pair_adj(d, G, sgm, u, s, gdg)
    k = oftype(sgm, const4)
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
