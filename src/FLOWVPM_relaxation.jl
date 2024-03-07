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

    nrmw = sqrt( (p[21]-p[23])*(p[21]-p[23]) +
                    (p[22]-p[18])*(p[22]-p[18]) +
                    (p[17]-p[19])*(p[17]-p[19]))
    nrmGamma = sqrt(p[4]^2 + p[5]^2 + p[6]^2)

    p[4] = (1-rlxf)*p[4] + rlxf*nrmGamma*(p[21]-p[23])/nrmw
    p[5] = (1-rlxf)*p[5] + rlxf*nrmGamma*(p[22]-p[18])/nrmw
    p[6] = (1-rlxf)*p[6] + rlxf*nrmGamma*(p[17]-p[19])/nrmw

    return nothing
end


"""
    `relax_correctedPedrizzetti(rlxf::Real, p)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time. See notebook 20200921 for derivation.
"""
function relax_correctedpedrizzetti(rlxf::Real, p)

    nrmw = sqrt( (p[21]-p[23])*(p[21]-p[23]) +
                    (p[22]-p[18])*(p[22]-p[18]) +
                    (p[17]-p[19])*(p[17]-p[19]))
    nrmGamma = sqrt(p[4]^2 + p[5]^2 + p[6]^2)

    b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (
                                    p[4]*(p[21]-p[23]) +
                                    p[5]*(p[22]-p[18]) +
                                    p[6]*(p[17]-p[19])
                                   ) / (nrmGamma*nrmw))

    p[4] = (1-rlxf)*p[4] + rlxf*nrmGamma*(p[21]-p[23])/nrmw
    p[5] = (1-rlxf)*p[5] + rlxf*nrmGamma*(p[22]-p[18])/nrmw
    p[6] = (1-rlxf)*p[6] + rlxf*nrmGamma*(p[17]-p[19])/nrmw

    # Normalize the direction of the new vector to maintain the same strength
    p[4:6] ./= sqrt(b2)

    return nothing
end
##### END OF RELAXATION SCHEME #################################################
