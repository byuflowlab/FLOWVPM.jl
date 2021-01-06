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
abstract type Formulation{R} end
##### END OF ABSTRACT VPM FORMULATION ##########################################



################################################################################
# CLASSIC VPM
################################################################################
struct ClassicVPM{R} <: Formulation{R} end
##### END OF CLASSIC VPM #######################################################



################################################################################
# REFORMULATED VPM
################################################################################
struct ReformulatedVPM{R} <: Formulation{R}
    f::R                     # Re-orientation parameter
    g::R                     # Stretching-compensation parameter
    h::R                     # Stretching parameter

    ReformulatedVPM{R}(f=R(0), g=R(1/5); h=(1-3*g)/(1+3*f)) where {R} = new(f, g, h)
end
##### END OF CLASSIC VPM #######################################################
