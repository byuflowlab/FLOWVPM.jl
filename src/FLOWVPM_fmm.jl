#=##############################################################################
# DESCRIPTION
    Fast-multipole parameters.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

################################################################################
# FMM STRUCT
################################################################################
mutable struct FMM
  # Optional user inputs
  p::Int32                        # Multipole expansion order
  ncrit::Int32                    # Max number of particles per leaf
  theta::RealFMM                  # Neighborhood criterion
  phi::RealFMM                    # Regularizing neighborhood criterion

  FMM(; p=4, ncrit=50, theta=0.4, phi=1/3) = new(p, ncrit, theta, phi)
end
