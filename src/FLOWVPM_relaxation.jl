#=##############################################################################
# DESCRIPTION
    VPM relaxation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# RELAXATION SCHEME
################################################################################
"""
    `Relaxation(relax, nsteps_relax, rlxf)`

Defines a relaxation method implemented in the function
`relax(rlxf::Real, p::Particle)` where `rlxf` is the relaxation factor between 0
and 1, with 0 == no relaxation, and 1 == full relaxation. The simulation is
relaxed every `nsteps_relax` steps.
"""
struct Relaxation{R}
    relax::Function                 # Relaxation method
    nsteps_relax::Int               # Relax simulation every this many steps
    rlxf::R                         # Relaxation factor between 0 and 1
end

# Make Relaxation object callable
(rlx::Relaxation)(p::Particle) = rlx.relax(rlx.rlxf, p)


##### RELAXATION METHODS #######################################################
"""
    `relax_Pedrizzetti(rlxf::Real, p::Particle)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
"""
function relax_pedrizzetti(rlxf::Real, p::Particle)

    nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                    (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                    (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
    nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

    p.Gamma[1] = (1-rlxf)*p.Gamma[1] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.Gamma[2] = (1-rlxf)*p.Gamma[2] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.Gamma[3] = (1-rlxf)*p.Gamma[3] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

    return nothing
end


"""
    `relax_correctedPedrizzetti(rlxf::Real, p::Particle)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time. See notebook 20200921 for derivation.
"""
function relax_correctedpedrizzetti(rlxf::Real, p::Particle)

    nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                    (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                    (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
    nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

    b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (
                                    p.Gamma[1]*(p.J[3,2]-p.J[2,3]) +
                                    p.Gamma[2]*(p.J[1,3]-p.J[3,1]) +
                                    p.Gamma[3]*(p.J[2,1]-p.J[1,2])
                                   ) / (nrmGamma*nrmw))

    p.Gamma[1] = (1-rlxf)*p.Gamma[1] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.Gamma[2] = (1-rlxf)*p.Gamma[2] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.Gamma[3] = (1-rlxf)*p.Gamma[3] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

    # Normalize the direction of the new vector to maintain the same strength
    p.Gamma ./= sqrt(b2)

    return nothing
end
##### END OF RELAXATION SCHEME #################################################
