#=##############################################################################
# DESCRIPTION
    Particle struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################


################################################################################
# PARTICLE STRUCT
################################################################################
"""
    `Particle{T}`

Vortex particle data structure

# State variables
* `X::Array{T, 1}`                : Position (3-elem array)
* `Gamma::Array{T, 1}`            : Vectorial circulation (3-elem array)
* `sigma::Array{T, 1}`            : Smoothing radius (1-elem array)
* `vol::Array{T, 1}`              : Volume (1-elem array)
* `circulation::Array{T, 1}`      : Scalar circulation (1-elem array)

# Public calculations
* `U::Array{T, 1}`                : Velocity at particle (3-elem array)
* `J::Array{T, 2}`                : Jacobian at particle J[i,j]=dUi/dxj (9-elem array)
"""
struct Particle{T}
  # User inputs
  X::Array{T, 1}                # Position (3-elem array)
  Gamma::Array{T, 1}            # Vectorial circulation (3-elem array)
  sigma::Array{T, 0}            # Smoothing radius (1-elem array)
  vol::Array{T, 0}              # Volume (1-elem array)
  circulation::Array{T, 0}      # Scalar circulation (1-elem array)
  static::Array{Bool, 0}        # If true, this particle is not evolved in time

  # Properties
  U::Array{T, 1}                # Velocity at particle (3-elem array)
  W::Array{T, 1}                # Vorticity at particle (3-elem array)
  J::Array{T, 2}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)
  PSE::Array{T, 1}              # Particle-strength exchange at particle (3-elem array)

  # Internal variables
  M::Array{T, 2}                # 3x3 array of auxiliary memory
  C::Array{T, 1}                # C[1]=SFS coefficient, C[2]=numerator, C[3]=denominator
  S::Array{T, 1}                # Stretching term

  # ExaFMM internal variables
#   Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
#   dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::Array{Int32, 0}        # Particle index (1-elem array)
end

function init_zero(type::DataType)
    z = Array{type,0}(undef)
    z[] = zero(type)
    return z
end

# Empty initializer
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(zeros(T, 3), zeros(T, 3),
                                                      init_zero(T),  init_zero(T), init_zero(T),
                                                      init_zero(Bool),
                                                      zeros(T, 3), zeros(T, 3), zeros(T, 3, 3), zeros(T, 3),
                                                      zeros(T, 3, 3), zeros(T, 3), zeros(T, 3),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                      init_zero(Int32))

# """
#     `Particle(body::fmm.BodyRef)`

# Return a particle that is linked with this C++ Body object. All changes in body
# will be reflected in the particles and vice versa.
# """
# Particle(body::fmm.BodyRef) = Particle{FLOAT_TYPE}(fmm.get_Xref(body),
#                                                 fmm.get_qref(body),
#                                                 fmm.get_sigmaref(body),
#                                                 fmm.get_volref(body),
#                                                 zeros(Bool, 1),
#                                                 zeros(FLOAT_TYPE, 1),
#                                                 zeros(FLOAT_TYPE, 3),
#                                                 zeros(FLOAT_TYPE, 3),
#                                                 zeros(FLOAT_TYPE, 3, 3),
#                                                 fmm.get_pseref(body),
#                                                 zeros(FLOAT_TYPE, 3, 3),
#                                                 zeros(FLOAT_TYPE, 3),
#                                                 fmm.get_Jref(body),
#                                                 fmm.get_dJdx1ref(body),
#                                                 fmm.get_dJdx2ref(body),
#                                                 fmm.get_dJdx3ref(body),
#                                                 fmm.get_indexref(body))


##### FUNCTIONS ################################################################
get_U(P::Particle) = P.U

get_W(P::Particle) = (get_W1(P), get_W2(P), get_W3(P))
get_W1(P::Particle) = P.J[3,2]-P.J[2,3]
get_W2(P::Particle) = P.J[1,3]-P.J[3,1]
get_W3(P::Particle) = P.J[2,1]-P.J[1,2]

get_SFS1(P::Particle{T}) where {T} = getproperty(P, _SFS)[1]::T
get_SFS2(P::Particle{T}) where {T} = getproperty(P, _SFS)[2]::T
get_SFS3(P::Particle{T}) where {T} = getproperty(P, _SFS)[3]::T
add_SFS1(P::Particle{T}, val) where {T} = getproperty(P, _SFS)[1]::T += val
add_SFS2(P::Particle{T}, val) where {T} = getproperty(P, _SFS)[2]::T += val
add_SFS3(P::Particle{T}, val) where {T} = getproperty(P, _SFS)[3]::T += val

##### INTERNAL FUNCTIONS #######################################################
nothing

##### END OF ABSTRACT PARTICLE FIELD############################################
