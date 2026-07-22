#=##############################################################################
# DESCRIPTION
    VPM relaxation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
=###############################################################################


################################################################################
# RELAXATION SCHEME
################################################################################
"""
    `Relaxation(relax, nsteps_relax, rlxf)`

Defines a relaxation method implemented in the function
`relax(rlxf::Real, p)` where `p` is particle,
`rlxf` is the relaxation factor between 0
and 1, with 0 == no relaxation, and 1 == full relaxation. The simulation is
relaxed every `nsteps_relax` steps.
"""
struct Relaxation{R,Trelax}
    relax::Trelax                 # Relaxation method
    nsteps_relax::Int               # Relax simulation every this many steps
    rlxf::R                         # Relaxation factor between 0 and 1
end

# Make Relaxation object callable
(rlx::Relaxation)(p) = rlx.relax(rlx.rlxf, p)
(rlx::Relaxation)(pfield, i) = rlx.relax(rlx.rlxf, pfield, i)

"""
    relax_broadcast!(rlx::Relaxation, pfield)

Whole-field broadcast relaxation, applied to every non-static particle at
once. Used unconditionally on both CPU and GPU (a single shared
implementation -- see logs/2026-07-21-gpu-full.md for the benchmark showing
the CPU cost of unifying is modest, 1.4-2.1x, unlike time integration's
4-10x, which is why that one stayed forked and this one didn't).

Not made a callable method of `Relaxation` (e.g. `rlx(pfield)`) because that
would collide with the existing single-particle callable `rlx(p)` -- both
take exactly one positional argument and Julia dispatches on argument type,
not on whether it represents a whole field or a single particle view.
"""
relax_broadcast!(rlx::Relaxation, pfield) = _relax_broadcast!(rlx.relax, rlx.rlxf, pfield)

_relax_broadcast!(relax, rlxf, pfield) = error(
    "No whole-field broadcast implementation registered for relaxation scheme `$relax`. " *
    "Since relaxation now always runs through `relax_broadcast!` (no per-particle CPU loop " *
    "fallback), a custom relaxation scheme must define a matching method of " *
    "`FLOWVPM._relax_broadcast!(::typeof($relax), rlxf, pfield)`.")


##### RELAXATION METHODS #######################################################
"""
    `relax_Pedrizzetti(rlxf::Real, p)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
"""
function relax_pedrizzetti(rlxf::Real, p)

    J = get_J(p)
    G = get_Gamma(p)

    nrmw = sqrt((J[6]-J[8])*(J[6]-J[8]) +
                (J[7]-J[3])*(J[7]-J[3]) +
                (J[2]-J[4])*(J[2]-J[4]))

    if !iszero(nrmw)
    
        nrmGamma = sqrt(G[1]^2 + G[2]^2 + G[3]^2)

        G[1] = (1-rlxf)*G[1] + rlxf*nrmGamma*(J[6]-J[8])/nrmw
        G[2] = (1-rlxf)*G[2] + rlxf*nrmGamma*(J[7]-J[3])/nrmw
        G[3] = (1-rlxf)*G[3] + rlxf*nrmGamma*(J[2]-J[4])/nrmw
    end

    return nothing
end

function relax_pedrizzetti(rlxf::Real, pfield, i)

    J = get_J(pfield, i)
    G = get_Gamma(pfield, i)

    nrmw = sqrt((J[6]-J[8])*(J[6]-J[8]) +
                (J[7]-J[3])*(J[7]-J[3]) +
                (J[2]-J[4])*(J[2]-J[4]))

    if !iszero(nrmw)
    
        nrmGamma = sqrt(G[1]^2 + G[2]^2 + G[3]^2)

        G[1] = (1-rlxf)*G[1] + rlxf*nrmGamma*(J[6]-J[8])/nrmw
        G[2] = (1-rlxf)*G[2] + rlxf*nrmGamma*(J[7]-J[3])/nrmw
        G[3] = (1-rlxf)*G[3] + rlxf*nrmGamma*(J[2]-J[4])/nrmw
    end

    return nothing
end

"""
    `relax_correctedPedrizzetti(rlxf::Real, p)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time. See notebook 20200921 for derivation.
"""
function relax_correctedpedrizzetti(rlxf::Real, p)

    J = get_J(p)
    G = get_Gamma(p)

    nrmw = sqrt((J[6]-J[8])*(J[6]-J[8]) +
                (J[7]-J[3])*(J[7]-J[3]) +
                (J[2]-J[4])*(J[2]-J[4]))

    if !iszero(nrmw)
        nrmGamma = sqrt(G[1]^2 + G[2]^2 + G[3]^2)

        b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (G[1]*(J[6]-J[8]) +
                                        G[2]*(J[7]-J[3]) +
                                        G[3]*(J[2]-J[4])) / (nrmGamma*nrmw))

        G[1] = (1-rlxf)*G[1] + rlxf*nrmGamma*(J[6]-J[8])/nrmw
        G[2] = (1-rlxf)*G[2] + rlxf*nrmGamma*(J[7]-J[3])/nrmw
        G[3] = (1-rlxf)*G[3] + rlxf*nrmGamma*(J[2]-J[4])/nrmw

        # Normalize the direction of the new vector to maintain the same strength
        G ./= sqrt(b2)
    end

    return nothing
end

function relax_correctedpedrizzetti(rlxf::Real, pfield, i)

    J = get_J(pfield, i)
    G = get_Gamma(pfield, i)

    nrmw = sqrt((J[6]-J[8])*(J[6]-J[8]) +
                (J[7]-J[3])*(J[7]-J[3]) +
                (J[2]-J[4])*(J[2]-J[4]))

    if !iszero(nrmw)
        nrmGamma = sqrt(G[1]^2 + G[2]^2 + G[3]^2)

        b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (G[1]*(J[6]-J[8]) +
                                        G[2]*(J[7]-J[3]) +
                                        G[3]*(J[2]-J[4])) / (nrmGamma*nrmw))

        G[1] = (1-rlxf)*G[1] + rlxf*nrmGamma*(J[6]-J[8])/nrmw
        G[2] = (1-rlxf)*G[2] + rlxf*nrmGamma*(J[7]-J[3])/nrmw
        G[3] = (1-rlxf)*G[3] + rlxf*nrmGamma*(J[2]-J[4])/nrmw

        # Normalize the direction of the new vector to maintain the same strength
        G ./= sqrt(b2)
    end

    return nothing
end

"GPU-compatible broadcast path for `relax_pedrizzetti`: same formula as the scalar version, vectorized over all particles."
function _relax_broadcast!(::typeof(relax_pedrizzetti), rlxf::Real, pfield)

    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 8, :); active .= 1.0 .- view(P, STATIC_INDEX, :)

    J2,J3,J4,J6,J7,J8 = (view(P, J_INDEX[k], :) for k in (2,3,4,6,7,8))
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)

    w1, w2, w3 = view(Sc, 1, :), view(Sc, 2, :), view(Sc, 3, :)
    w1 .= J6 .- J8
    w2 .= J7 .- J3
    w3 .= J2 .- J4

    nrmw = view(Sc, 4, :); nrmw .= sqrt.(w1.^2 .+ w2.^2 .+ w3.^2)
    nrmGamma = view(Sc, 5, :); nrmGamma .= sqrt.(G1.^2 .+ G2.^2 .+ G3.^2)

    apply = view(Sc, 6, :); apply .= active .* (nrmw .> 0)
    safenrmw = view(Sc, 7, :); safenrmw .= ifelse.(nrmw .> 0, nrmw, one(eltype(nrmw)))

    G1 .= ifelse.(apply .> 0, (1-rlxf).*G1 .+ rlxf.*nrmGamma.*w1./safenrmw, G1)
    G2 .= ifelse.(apply .> 0, (1-rlxf).*G2 .+ rlxf.*nrmGamma.*w2./safenrmw, G2)
    G3 .= ifelse.(apply .> 0, (1-rlxf).*G3 .+ rlxf.*nrmGamma.*w3./safenrmw, G3)

    return nothing
end

"GPU-compatible broadcast path for `relax_correctedpedrizzetti`: same formula as the scalar version, vectorized over all particles."
function _relax_broadcast!(::typeof(relax_correctedpedrizzetti), rlxf::Real, pfield)

    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 8, :); active .= 1.0 .- view(P, STATIC_INDEX, :)

    J2,J3,J4,J6,J7,J8 = (view(P, J_INDEX[k], :) for k in (2,3,4,6,7,8))
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)

    w1, w2, w3 = view(Sc, 1, :), view(Sc, 2, :), view(Sc, 3, :)
    w1 .= J6 .- J8
    w2 .= J7 .- J3
    w3 .= J2 .- J4

    nrmw = view(Sc, 4, :); nrmw .= sqrt.(w1.^2 .+ w2.^2 .+ w3.^2)
    nrmGamma = view(Sc, 5, :); nrmGamma .= sqrt.(G1.^2 .+ G2.^2 .+ G3.^2)

    apply = view(Sc, 6, :); apply .= active .* (nrmw .> 0)
    safenrmw = view(Sc, 7, :); safenrmw .= ifelse.(nrmw .> 0, nrmw, one(eltype(nrmw)))

    # sqrtb2 reuses nrmw's row: its RHS reads only w1/w2/w3/G1/G2/G3/nrmGamma/safenrmw
    # (never nrmw itself), so overwriting nrmw's row here is safe (see point 7 of
    # feedback-gpu-verification-standards on fused read/overwrite of a shared row).
    # No positivity guard, matching the scalar version's `sqrt(b2)` exactly.
    sqrtb2 = nrmw
    sqrtb2 .= sqrt.(1 .- 2 .* (1-rlxf) .* rlxf .* (1 .- (G1.*w1 .+ G2.*w2 .+ G3.*w3) ./ (nrmGamma .* safenrmw)))

    G1 .= ifelse.(apply .> 0, ((1-rlxf).*G1 .+ rlxf.*nrmGamma.*w1./safenrmw) ./ sqrtb2, G1)
    G2 .= ifelse.(apply .> 0, ((1-rlxf).*G2 .+ rlxf.*nrmGamma.*w2./safenrmw) ./ sqrtb2, G2)
    G3 .= ifelse.(apply .> 0, ((1-rlxf).*G3 .+ rlxf.*nrmGamma.*w3./safenrmw) ./ sqrtb2, G3)

    return nothing
end
##### END OF RELAXATION SCHEME #################################################
