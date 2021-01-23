#=##############################################################################
# DESCRIPTION
    P2P kernels

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

################################################################################
# KERNELS
################################################################################
"""
  `Kernel(zeta, g, dgdr, g_dgdr, EXAFMM_P2P, EXAFMM_L2P)`

**Arguments**
* `zeta::Function`        : Basis function zeta(r).
* `g::Function`           : Regularizing function g(r).
* `dgdr::Function`        : Derivative of g(r).
* `g_dgdr::Function`      : Efficient evaluation of g and dgdr.
* `EXAFMM_P2P::Int`       : Flag for the ExaFMM P2P function to call.
* `EXAFMM_L2P::Int`       : Flag for the ExaFMM L2P function to call.
"""
struct Kernel
  zeta::Function                        # Basis function zeta(r)
  g::Function                           # Regularizing function g(r)
  dgdr::Function                        # Derivative of g(r)
  g_dgdr::Function                      # Efficient evaluation of g and dgdr
  EXAFMM_P2P::Int32                     # Flag for the ExaFMM P2P function to call
  EXAFMM_L2P::Int32                     # Flag for the ExaFMM L2P function to call
end

# Constant values
const const1 = 1/(2*pi)^(3/2)
const const2 = sqrt(2/pi)
const const3 = 3/(4*pi)
const const4 = 1/(4*pi)
const sqr2 = sqrt(2)

# Newtonian velocity kernel
# Knew(X) = -const4 * X / norm(X)^3
function Knew(X)
    aux = -const4 / (X[1]^2 + X[2]^2 + X[3]^2)^(3/2)
    return (aux*X[1], aux*X[2], aux*X[3])
end

# Singular kernel
zeta_sing(r::Real) = r==0 ? 1 : 0
g_sing(r::Real) = 1
dgdr_sing(r::Real) = 0
g_dgdr_sing(r::Real) = (g_sing(r), dgdr_sing(r))

# erf Gaussian kernel
zeta_gauserf(r::Real) = const1*exp(-r^2/2)
g_gauserf(r::Real) = SpecialFunctions.erf(r/sqr2) - const2*r*exp(-r^2/2)
dgdr_gauserf(r::Real) = const2*r^2*exp(-r^2/2)
function g_dgdr_gauserf(r::Real)
  aux = const2*r*exp(-r^2/2)
  return SpecialFunctions.erf(r/sqr2)-aux, r*aux
end

# Gaussian kernel
zeta_gaus(r::Real) = const3*exp(-r^3)
g_gaus(r::Real) = 1-exp(-r^3)
dgdr_gaus(r::Real) = 3*r^2*exp(-r^3)
function g_dgdr_gaus(r::Real)
  aux = exp(-r^3)
  return 1-aux, 3*r^2*aux
end

# Rotor kernel
zeta_turbine(r::Real) = const3*exp(-(r/5.0)^3)
g_turbine(r::Real) = 1-exp(-(r/5.0)^3)
dgdr_turbine(r::Real) = 3*(r/5.0)^2*exp(-(r/5.0)^3)
function g_dgdr_turbine(r::Real)
  aux = exp(-(r/5.0)^3)
  return 1-aux, 3*(r/5.0)^2*aux
end

# Winckelmans algebraic kernel
zeta_wnklmns(r::Real) = const4 * 7.5 / (r^2 + 1)^3.5
g_wnklmns(r::Real) = r^3 * (r^2 + 2.5) / (r^2 + 1)^2.5
dgdr_wnklmns(r::Real) = 7.5 * r^2 / (r^2 + 1)^3.5
function g_dgdr_wnklmns(r::Real)
  aux0 = (r^2 + 1)^2.5

  # Returns g, dgdr
  return r^3 * (r^2 + 2.5) / aux0, 7.5 * r^2 / (aux0*(r^2 + 1))
end
