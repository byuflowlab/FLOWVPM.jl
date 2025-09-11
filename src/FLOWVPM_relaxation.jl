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
        for Gi in G
            Gi /= sqrt(b2)
        end
        #G ./= sqrt(b2)
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
