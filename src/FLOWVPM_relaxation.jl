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
    `Relaxation(relax, nsteps_relax, rlxf[, filter])`

Defines a relaxation method implemented in the function
`relax(rlxf::Real, p)` where `p` is particle,
`rlxf` is the relaxation factor between 0
and 1, with 0 == no relaxation, and 1 == full relaxation. The simulation is
relaxed every `nsteps_relax` steps.

`filter` is an optional per-particle predicate `(p)->Bool`; relaxation is applied
to a particle only when `filter(p)` is `true`. It defaults to [`relax_filter_all`]
(all particles relaxed), which preserves the historical behavior. The predicate
receives a particle column view `p` (use `get_X(p)`, `get_Gamma(p)`, etc.).
"""
struct Relaxation{R,Trelax,Tfilter}
    relax::Trelax                 # Relaxation method
    nsteps_relax::Int               # Relax simulation every this many steps
    rlxf::R                         # Relaxation factor between 0 and 1
    filter::Tfilter                 # per-particle predicate (p)->Bool gating relaxation
end

"""
    `relax_filter_all(p)`

Default relaxation filter: a predicate that always returns `true`, so relaxation is
applied to every particle. Defined as a named function (rather than an anonymous
closure) so that all default `Relaxation`s share a single concrete type and remain
comparable by `==`.
"""
relax_filter_all(p) = true
relax_filter_all(pfield, i::Integer) = true

# Backwards-compatible constructor: default to the all-pass filter so every existing
# 3-argument `Relaxation(...)` call site keeps working unchanged.
Relaxation(relax, nsteps_relax, rlxf) =
    Relaxation(relax, nsteps_relax, rlxf, relax_filter_all)

_passes_relaxation_filter(filter, pfield, i::Integer) = filter(get_particle(pfield, Int(i)))
_passes_relaxation_filter(::typeof(relax_filter_all), pfield, i::Integer) = true

# Make Relaxation object callable, gated by the per-particle filter
(rlx::Relaxation)(p) = rlx.filter(p) ? rlx.relax(rlx.rlxf, p) : nothing
(rlx::Relaxation)(pfield, i) =
    _passes_relaxation_filter(rlx.filter, pfield, i) ? rlx.relax(rlx.rlxf, pfield, i) : nothing


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
##### END OF RELAXATION SCHEME #################################################
