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
`relax(rlxf::Real, p::Particle)` where `rlxf` is the relaxation factor between 0
and 1, with 0 == no relaxation, and 1 == full relaxation. The simulation is
relaxed every `nsteps_relax` steps.
"""
struct Relaxation{R,Trelax}
    relax::Trelax                 # Relaxation method
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
    nrmGamma = sqrt(p.var[4]^2 + p.var[5]^2 + p.var[6]^2)

    p.var[4] = (1-rlxf)*p.var[4] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.var[5] = (1-rlxf)*p.var[5] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.var[6] = (1-rlxf)*p.var[6] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

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
    nrmGamma = sqrt(p.var[4]^2 + p.var[5]^2 + p.var[6]^2)

    b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (
                                    p.var[4]*(p.J[3,2]-p.J[2,3]) +
                                    p.var[5]*(p.J[1,3]-p.J[3,1]) +
                                    p.var[6]*(p.J[2,1]-p.J[1,2])
                                   ) / (nrmGamma*nrmw))

    p.var[4] = (1-rlxf)*p.var[4] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.var[5] = (1-rlxf)*p.var[5] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.var[6] = (1-rlxf)*p.var[6] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

    # Normalize the direction of the new vector to maintain the same strength
    p.var[4:6] ./= sqrt(b2)

    return nothing
end
##### END OF RELAXATION SCHEME #################################################
