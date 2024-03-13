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


##### RELAXATION METHODS #######################################################
"""
    `relax_Pedrizzetti(rlxf::Real, p)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
"""
function relax_pedrizzetti(rlxf::Real, p)

    nrmw = sqrt( (get_J(p)[6]-get_J(p)[8])*(get_J(p)[6]-get_J(p)[8]) +
                    (get_J(p)[7]-get_J(p)[3])*(get_J(p)[7]-get_J(p)[3]) +
                    (get_J(p)[2]-get_J(p)[4])*(get_J(p)[2]-get_J(p)[4]))
    nrmGamma = sqrt(get_Gamma(p)[1]^2 + get_Gamma(p)[2]^2 + get_Gamma(p)[3]^2)

    get_Gamma(p)[1] = (1-rlxf)*get_Gamma(p)[1] + rlxf*nrmGamma*(get_J(p)[6]-get_J(p)[8])/nrmw
    get_Gamma(p)[2] = (1-rlxf)*get_Gamma(p)[2] + rlxf*nrmGamma*(get_J(p)[7]-get_J(p)[3])/nrmw
    get_Gamma(p)[3] = (1-rlxf)*get_Gamma(p)[3] + rlxf*nrmGamma*(get_J(p)[2]-get_J(p)[4])/nrmw

    return nothing
end


"""
    `relax_correctedPedrizzetti(rlxf::Real, p)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time. See notebook 20200921 for derivation.
"""
function relax_correctedpedrizzetti(rlxf::Real, p)

    nrmw = sqrt( (get_J(p)[6]-get_J(p)[8])*(get_J(p)[6]-get_J(p)[8]) +
                    (get_J(p)[7]-get_J(p)[3])*(get_J(p)[7]-get_J(p)[3]) +
                    (get_J(p)[2]-get_J(p)[4])*(get_J(p)[2]-get_J(p)[4]))
    nrmGamma = sqrt(get_Gamma(p)[1]^2 + get_Gamma(p)[2]^2 + get_Gamma(p)[3]^2)

    b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (
                                    get_Gamma(p)[1]*(get_J(p)[6]-get_J(p)[8]) +
                                    get_Gamma(p)[2]*(get_J(p)[7]-get_J(p)[3]) +
                                    get_Gamma(p)[3]*(get_J(p)[2]-get_J(p)[4])
                                   ) / (nrmGamma*nrmw))

    get_Gamma(p)[1] = (1-rlxf)*get_Gamma(p)[1] + rlxf*nrmGamma*(get_J(p)[6]-get_J(p)[8])/nrmw
    get_Gamma(p)[2] = (1-rlxf)*get_Gamma(p)[2] + rlxf*nrmGamma*(get_J(p)[7]-get_J(p)[3])/nrmw
    get_Gamma(p)[3] = (1-rlxf)*get_Gamma(p)[3] + rlxf*nrmGamma*(get_J(p)[2]-get_J(p)[4])/nrmw

    # Normalize the direction of the new vector to maintain the same strength
    get_Gamma(p) ./= sqrt(b2)

    return nothing
end
##### END OF RELAXATION SCHEME #################################################
