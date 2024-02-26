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
  X::MVector{3,T}                # Position (3-elem array)
  # Gamma::MVector{3,T}            # Vectorial circulation (3-elem array)
  # sigma::MVector{1,T}            # Smoothing radius (1-elem array)
  # vol::MVector{1,T}              # Volume (1-elem array)
  # circulation::MVector{1,T}      # Scalar circulation (1-elem array)
  static::MVector{1,Bool}        # If true, this particle is not evolved in time

  # Properties
  U::MVector{3,T}                # Velocity at particle (3-elem array)
  W::MVector{3,T}                # Vorticity at particle (3-elem array)
  J::MMatrix{3,3,T,9}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)
  PSE::MVector{3,T}              # Particle-strength exchange at particle (3-elem array)

  # Internal variables
  M::MMatrix{3,3,T,9}                # 3x3 array of auxiliary memory
  C::MVector{3,T}                # C[1]=SFS coefficient, C[2]=numerator, C[3]=denominator
  S::MVector{3,T}                # Stretching term

  # ExaFMM internal variables
#   Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
#   dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::MVector{1,Int32}        # Particle index (1-elem array)

  # Combined vector of all variables
  var::MVector{42,T}
end

function init_zero(type::DataType)
    return MVector{1,type}(0.0)
end

function init_zeros3(type::DataType)
    return MVector{3,type}(0.0,0.0,0.0)
end

function init_zeros33(type::DataType)
    return MMatrix{3,3,type,9}(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
end

function init_zeros42(type::DataType)
    return @MVector zeros(type, 42)
end

Base.eltype(::Particle{T}) where T = T
Base.eltype(::AbstractArray{Particle{T}}) where T = T

# Empty initializer
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(init_zeros3(T),
                                                      # init_zeros3(T),
                                                      # init_zero(T),
                                                      # init_zero(T),
                                                      # init_zero(T),
                                                      init_zero(Bool),
                                                      init_zeros3(T),
                                                      init_zeros3(T),
                                                      init_zeros33(T),
                                                      init_zeros3(T),
                                                      init_zeros33(T),
                                                      init_zeros3(T),
                                                      init_zeros3(T),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                    init_zero(Int32),
                                                    init_zeros42(T))



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

##### END OF ABSTRACT PARTICLE FIELD############################################
