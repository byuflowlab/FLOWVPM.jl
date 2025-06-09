#=##############################################################################
# DESCRIPTION
    P2P kernels

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
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
"""
struct Kernel{Tz,Tg,Tdg,Tgdg}
    zeta::Tz                              # Basis function zeta(r)
    g::Tg                                 # Regularizing function g(r)
    dgdr::Tdg                             # Derivative of g(r)
    g_dgdr::Tgdg                          # Efficient evaluation of g and dgdr
end

# # Constant values
# const const1 = 1/(2*pi)^1.5
# const const2 = sqrt(2/pi)
# const const3 = 3/(4*pi)
# const const4 = 1/(4*pi)
# const sqr2 = sqrt(2)

# Newtonian velocity kernel
# Knew(X) = -const4 * X / norm(X)^3
@inline function Knew(X)
    aux = -const4 / (X[1]^2 + X[2]^2 + X[3]^2)^1.5
    return (aux*X[1], aux*X[2], aux*X[3])
end

# Singular kernel
zeta_sing(r) = iszero(r) ? 1.0 : 0.0
g_sing(r) = 1.0
dgdr_sing(r) = 0.0
@inline g_dgdr_sing(r) = (1.0, 0.0)

# erf Gaussian kernel
zeta_gauserf(r) = const1*exp(-r*r/2)
g_gauserf(r) = custom_erf(r/sqr2) - const2*r*exp(-r*r/2)
dgdr_gauserf(r) = const2*r*r*exp(-r*r/2)
@inline function g_dgdr_gauserf(r)
  aux = const2*r*exp(-r*r/2)
  return custom_erf(r/sqr2)-aux, r*aux
end

# Gaussian kernel
zeta_gaus(r) = const3*exp(-r*r*r)
g_gaus(r) = 1-exp(-r*r*r)
dgdr_gaus(r) = 3*r*r*exp(-r*r*r)
@inline function g_dgdr_gaus(r)
  aux = exp(-r*r*r)
  return 1-aux, 3*r*r*aux
end

# Winckelmans algebraic kernel
function zeta_wnklmns(r)
  temp = r*r + 1
  temp2 = temp*temp*temp
  temp = temp2*temp2*temp
  return const4 * 7.5 / sqrt(temp)
end
g_wnklmns(r) = r*r*r * (r*r + 2.5) / (r*r + 1)^2.5
dgdr_wnklmns(r) = 7.5 * r*r / (r*r + 1)^3.5
@inline function g_dgdr_wnklmns(r)
  # aux0 = (r*r + 1)^2.5
  aux0 = r*r+1
  aux02 = aux0*aux0
  aux0 = aux02*aux02*aux0
  aux0 = sqrt(aux0)
  return r*r*r * (r*r + 2.5) / aux0, 7.5 * r*r / (aux0*(r*r + 1))
end
