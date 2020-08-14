#=##############################################################################
# DESCRIPTION
    Viscous schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

################################################################################
# ABSTRACT VISCOUS SCHEME TYPE
################################################################################
"""
    `ViscousScheme{R}`

Type declaring viscous scheme.

Implementations must have the following properties:
    * `nu::R`                   : Kinematic viscosity.
"""
abstract type ViscousScheme{R} end
##### END OF ABSTRACT VISCOUS SCHEME ###########################################

################################################################################
# INVISCID SCHEME TYPE
################################################################################
struct Inviscid{R} <: ViscousScheme{R}
    nu::R                                 # Kinematic viscosity
    Inviscid{R}(; nu=zero(R)) where {R} = new(nu)
end

Inviscid() = Inviscid{RealFMM}()

"""
    `isinviscid(viscous::ViscousScheme)`

Returns true if viscous scheme is inviscid.
"""
isinviscid(viscous::ViscousScheme) = typeof(viscous).name == Inviscid.body.name
##### END OF INVISCID SCHEME ###################################################


################################################################################
# CORE SPEADING SCHEME TYPE
################################################################################
mutable struct CoreSpreading{R} <: ViscousScheme{R}
    # User inputs
    nu::R                                 # Kinematic viscosity
    sgm0::R                               # Core size after reset

    # Optional inputs
    beta::R                               # Maximum core size growth σ/σ_0
    itmax::Int                            # Maximum number of RBF iterations
    tol::R                                # RBF interpolation tolerance
    iterror::Bool                         # Throw error if RBF didn't converge
    verbose::Bool                         # Verbose on RBF interpolation

    # Internal properties
    t_sgm::R                              # Time since last core size reset
    CoreSpreading{R}(
                        nu, sgm0;
                        beta=R(1.5),
                        itmax=R(15), tol=R(1e-3), iterror=true, verbose=false,
                        t_sgm=R(0.0)
                    ) where {R} = new(
                        nu, sgm0,
                        beta,
                        itmax, tol, iterror, verbose,
                        t_sgm
                    )
end

CoreSpreading(sgm0::R, args...; optargs...
                       ) where {R} = CoreSpreading{R}(sgm0, args...; optargs...)

"""
   `iscorespreading(viscous::ViscousScheme)`

Returns true if viscous scheme is core spreading.
"""
iscorespreading(viscous::ViscousScheme
                            ) = typeof(viscous).name == CoreSpreading.body.name
##### END OF CORE SPREADING SCHEME #############################################


##### COMMON FUNCTIONS #########################################################
################################################################################
