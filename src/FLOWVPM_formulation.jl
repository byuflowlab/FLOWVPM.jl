#=##############################################################################
# DESCRIPTION
    VPM formulation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Nov 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# ABSTRACT VPM FORMULATION TYPE
################################################################################
abstract type Formulation{P<:AbstractParticle, R} end
##### END OF ABSTRACT VPM FORMULATION ##########################################



################################################################################
# CLASSIC VPM
################################################################################
struct ClassicVPM{P<:AbstractParticle, R} <: Formulation{P, R} end
##### END OF CLASSIC VPM #######################################################



################################################################################
# REFORMULATED VPM
################################################################################
struct ReformulatedVPM{P<:AbstractParticle, R} <: Formulation{P, R}
    f::R                     # Re-orientation parameter
    g::R                     # Stretching-compensation parameter
    h::R                     # Stretching parameter

    ReformulatedVPM{P, R}(f, g; h=(1-3*g)/(1+3*f)) where {P, R} = new(f, g, h)
end
##### END OF CLASSIC VPM #######################################################
