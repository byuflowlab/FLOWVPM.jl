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

#=##############################################################################
# Per-particle maps of the fused device updates (ext/FLOWVPMGPUExt.jl) as pure
# functions of one column: the kernels call them, and the device reverse rules
# (ext/FLOWVPMEnzymeGPUExt.jl) differentiate them particle by particle.
=###############################################################################

@inline function _sfs_stretch(transposed, j1,j2,j3,j4,j5,j6,j7,j8,j9, g1,g2,g3)
    return transposed ?
        (j1*g1 + j2*g2 + j3*g3, j4*g1 + j5*g2 + j6*g3, j7*g1 + j8*g2 + j9*g3) :
        (j1*g1 + j4*g2 + j7*g3, j2*g1 + j5*g2 + j8*g3, j3*g1 + j6*g2 + j9*g3)
end

@inline _gather_col(P, i, rows::NTuple{N,Int}) where N = SVector{N}(ntuple(k -> @inbounds(P[rows[k], i]), Val(N)))

# RK3 register update of the reformulated VPM (one stage): reads X G sigma U J
# M[1:8] C SFS (31 rows, `_RK3_IN`), writes X G sigma M[1:8] (15, `_RK3_OUT`)
const _RK3_IN = (X_INDEX..., GAMMA_INDEX..., SIGMA_INDEX, U_INDEX..., J_INDEX..., M_INDEX[1:8]...,
                 C_INDEX[1], SFS_INDEX...)
const _RK3_OUT = (X_INDEX..., GAMMA_INDEX..., SIGMA_INDEX, M_INDEX[1:8]...)
_rk3_consts(::Type{R}, a, b, dt, Uinf, f, g, zeta0) where R =
    (fR = R(f); gR = R(g); (R(a), R(b), R(dt), R(Uinf[1]), R(Uinf[2]), R(Uinf[3]),
                            (fR + gR) / (1 + 3fR), fR / (1 + 3fR), R(1 / zeta0)))
@inline function _rk3_reformulated_col(x, transposed, sc)
    a, b, dt, uinf1, uinf2, uinf3, cfg, cf, inv_zeta0 = sc
    T = eltype(x)
    @inbounds begin
        g1 = x[4]; g2 = x[5]; g3 = x[6]; sig = x[7]
        s1 = x[29]; s2 = x[30]; s3 = x[31]; c1 = x[28]
        m1n = a * x[20] + dt * (x[8]  + uinf1)
        m2n = a * x[21] + dt * (x[9]  + uinf2)
        m3n = a * x[22] + dt * (x[10] + uinf3)
        mm1, mm2, mm3 = _sfs_stretch(transposed, x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19], g1,g2,g3)
        gn2 = g1*g1 + g2*g2 + g3*g3
        sdotg = mm1*g1 + mm2*g2 + mm3*g3
        sig3z = sig*sig*sig * inv_zeta0
        cepsg = c1 * (s1*g1 + s2*g2 + s3*g3) * sig3z
        mm4 = gn2 > zero(T) ? (cfg * sdotg - cf * cepsg) / gn2 : zero(T)
        m4n = a * x[23] + dt * (mm1 - 3 * mm4 * g1 - c1 * s1 * sig3z)
        m5n = a * x[24] + dt * (mm2 - 3 * mm4 * g2 - c1 * s2 * sig3z)
        m6n = a * x[25] + dt * (mm3 - 3 * mm4 * g3 - c1 * s3 * sig3z)
        m8n = a * x[27] - dt * (sig * mm4)
        return SVector(x[1] + b * m1n, x[2] + b * m2n, x[3] + b * m3n,
                       g1 + b * m4n, g2 + b * m5n, g3 + b * m6n, sig + b * m8n,
                       m1n, m2n, m3n, m4n, m5n, m6n, x[26], m8n)
    end
end

# Euler update of the reformulated VPM (`_euler_broadcast_reformulated!`, same
# math per particle): reads X G sigma U J C SFS (23), writes X G sigma (7)
const _EULER_IN = (X_INDEX..., GAMMA_INDEX..., SIGMA_INDEX, U_INDEX..., J_INDEX..., C_INDEX[1], SFS_INDEX...)
const _EULER_OUT = (X_INDEX..., GAMMA_INDEX..., SIGMA_INDEX)
_euler_consts(::Type{R}, dt, Uinf, f, g, zeta0) where R =
    (fR = R(f); gR = R(g); (R(dt), R(Uinf[1]), R(Uinf[2]), R(Uinf[3]), (fR + gR) / (1 + 3fR), fR / (1 + 3fR), 1 / R(zeta0)))
@inline function _euler_reformulated_col(x, transposed, sc)
    dt, uinf1, uinf2, uinf3, cfg, cf, inv_zeta0 = sc
    T = eltype(x)
    @inbounds begin
        g1 = x[4]; g2 = x[5]; g3 = x[6]; sig = x[7]; c1 = x[20]; s1 = x[21]; s2 = x[22]; s3 = x[23]
        mm1, mm2, mm3 = _sfs_stretch(transposed, x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19], g1,g2,g3)
        gn2 = g1*g1 + g2*g2 + g3*g3
        sig3z = sig*sig*sig * inv_zeta0
        mm4 = gn2 > zero(T) ? (cfg * (mm1*g1 + mm2*g2 + mm3*g3) - cf * c1 * (s1*g1 + s2*g2 + s3*g3) * sig3z) / gn2 : zero(T)
        return SVector(x[1] + dt * (x[8] + uinf1), x[2] + dt * (x[9] + uinf2), x[3] + dt * (x[10] + uinf3),
                       g1 + dt * (mm1 - 3*mm4*g1 - c1*s1*sig3z), g2 + dt * (mm2 - 3*mm4*g2 - c1*s2*sig3z),
                       g3 + dt * (mm3 - 3*mm4*g3 - c1*s3*sig3z), sig - dt * (sig * mm4))
    end
end

# core spreading's growth: RK3 (M[7], sigma) and Euler (sigma)
const _CS_RK3_ROWS = (M_INDEX[7], SIGMA_INDEX)
@inline function _cs_rk3_col(x, sc)
    nu2dt, aux1, aux2 = sc
    m7 = aux1 * x[1] + nu2dt
    return SVector(m7, sqrt(x[2] * x[2] + aux2 * m7))
end
const _CS_EULER_ROWS = (SIGMA_INDEX,)
@inline _cs_euler_col(x, nu2dt) = SVector(sqrt(x[1] * x[1] + nu2dt))

# relaxation (Pedrizzetti; CORRECTED: the corrected form): reads Gamma and J, writes Gamma
const _RELAX_IN = (GAMMA_INDEX..., J_INDEX...)
const _RELAX_OUT = (GAMMA_INDEX...,)
@inline function _relax_col(x, rlxf, ::Val{CORRECTED}) where {CORRECTED}
    T = eltype(x)
    @inbounds begin
        G1 = x[1]; G2 = x[2]; G3 = x[3]
        J2 = x[5]; J3 = x[6]; J4 = x[7]; J6 = x[9]; J7 = x[10]; J8 = x[11]
        w1 = J6 - J8; w2 = J7 - J3; w3 = J2 - J4
        nrmw = sqrt(w1^2 + w2^2 + w3^2)
        nrmGamma = sqrt(G1^2 + G2^2 + G3^2)
        # the corrected form divides by |Gamma|, so guard it as the host does
        apply = CORRECTED ? (nrmw > zero(T)) & (nrmGamma > zero(T)) : nrmw > zero(T)
        safenrmw = ifelse(nrmw > zero(T), nrmw, one(T))
        omr = 1 - rlxf
        if CORRECTED
            sqrtb2 = sqrt(1 - 2 * omr * rlxf * (1 - (G1*w1 + G2*w2 + G3*w3) / (nrmGamma * safenrmw)))
            return SVector(ifelse(apply, (omr*G1 + rlxf*nrmGamma*w1/safenrmw) / sqrtb2, G1),
                           ifelse(apply, (omr*G2 + rlxf*nrmGamma*w2/safenrmw) / sqrtb2, G2),
                           ifelse(apply, (omr*G3 + rlxf*nrmGamma*w3/safenrmw) / sqrtb2, G3))
        else
            return SVector(ifelse(apply, omr*G1 + rlxf*nrmGamma*w1/safenrmw, G1),
                           ifelse(apply, omr*G2 + rlxf*nrmGamma*w2/safenrmw, G2),
                           ifelse(apply, omr*G3 + rlxf*nrmGamma*w3/safenrmw, G3))
        end
    end
end

# constant SFS: the model coefficient set to Cs, then clipped to 0 where it would
# backscatter (C (Gamma . SFS) < 0); the clipping decision carries no derivative
const _SFSC_ROWS = (C_INDEX[1],)
@inline _sfs_coefficient_col(x, Cs) = SVector(Cs + zero(x[1]))
const _CLIP_IN = (C_INDEX[1], GAMMA_INDEX..., SFS_INDEX...)
@inline _clip_backscatter_col(x) =
    @inbounds SVector(ifelse(x[1] * (x[2]*x[5] + x[3]*x[6] + x[4]*x[7]) < 0, zero(x[1]), x[1]))

# dynamic SFS on the device. After UJ (`_pseudo3level_afterUJ_broadcast!`, active
# particles): the model coefficient from the test- and domain-filter terms in M,
# relaxed, clamped (piecewise: the branch carries no derivative), then M flushed.
# Reads Gamma sigma J M[1:9] C SFS (28), writes C M[1:9] (12).
const _SFSA_IN = (GAMMA_INDEX..., SIGMA_INDEX, J_INDEX..., M_INDEX..., C_INDEX..., SFS_INDEX...)
const _SFSA_OUT = (C_INDEX..., M_INDEX...)
_sfs_afterUJ_consts(::Type{R}, pfield, alpha, rlxf, minC, maxC, twolevel) where R =
    (twolevel ? one(R) : R(3 * alpha - 2), R(rlxf), R(minC), R(maxC), R(pfield.kernel.zeta(zero(R))), R(eps()))
@inline function _sfs_afterUJ_col(x, transposed, sc, ::Val{FP}, subtract::Bool) where {FP}
    fac, rlxf, minC, maxC, zeta0, epsv = sc
    T = eltype(x)
    @inbounds begin
        g1 = x[1]; g2 = x[2]; g3 = x[3]; sig = x[4]
        m1 = x[14]; m2 = x[15]; m3 = x[16]; m4 = x[17]; m5 = x[18]; m6 = x[19]
        c2r = x[24]; c3r = x[25]; s1 = x[26]; s2 = x[27]; s3 = x[28]
        # subtract the domain-filter stretching / SFS from the test-filter values
        d1, d2, d3 = _sfs_stretch(transposed, x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13], g1,g2,g3)
        if subtract      # the two-level procedure holds the derivatives themselves
            m1 -= d1; m2 -= d2; m3 -= d3
            m4 -= s1; m5 -= s2; m6 -= s3
        end
        nume_raw = (m1*g1 + m2*g2 + m3*g3) * fac
        deno_raw = (m4*g1 + m5*g2 + m6*g3) * (sig*sig*sig / zeta0)
        c3i = c3r == zero(T) ? (deno_raw == zero(T) ? epsv : deno_raw) : c3r
        nume = rlxf * nume_raw + (one(T) - rlxf) * c2r
        deno = rlxf * deno_raw + (one(T) - rlxf) * c3i
        # clamping, in the host's branch order: `big` is tested with the OLD
        # deno, deno is then updated, and nume re-tested against the NEW one
        big = abs(nume / deno) > maxC
        if big && abs(deno) < abs(c3i)
            deno = copysign(abs(c3i), iszero(deno) ? c3i : deno)     # sign(0) kept a zero denominator (host fix mirrored)
        end
        if big && abs(nume / deno) >= maxC
            nume = sign(nume) * abs(deno) * maxC
        elseif !big && abs(nume / deno) < minC
            nume = sign(nume) * abs(deno) * minC
        end
        c1r = nume / deno
        FP && (c1r = abs(c1r))
        z = zero(T)
        return SVector(c1r, nume, deno, z, z, z, z, z, z, z, z, z)
    end
end

# Before UJ (pseudo-three-level): sigma *= alpha (the test filter width), then
# after the test-width UJ: M = [stretching, SFS, 0, 0, 0], sigma /= alpha.
const _SCALE_ROWS = (SIGMA_INDEX,)
@inline _sfs_scale_col(x, alpha) = SVector(x[1] * alpha)
const _SFSB_IN = (GAMMA_INDEX..., SIGMA_INDEX, J_INDEX..., SFS_INDEX...)
const _SFSB_OUT = (M_INDEX..., SIGMA_INDEX)
@inline function _sfs_store_col(x, transposed, alpha)
    @inbounds begin
        d1, d2, d3 = _sfs_stretch(transposed, x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13], x[1],x[2],x[3])
        z = zero(eltype(x))
        return SVector(d1, d2, d3, x[14], x[15], x[16], z, z, z, x[4] / alpha)
    end
end
# the two device launches around the test-width UJ (GPU extension)
function _sfs_scale_sigma_dev! end
function _sfs_beforeUJ_store_dev! end
