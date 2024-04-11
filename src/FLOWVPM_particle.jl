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
  X::Vector{T}                # Position (3-elem array)
  Gamma::Vector{T}            # Vectorial circulation (3-elem array)
  sigma::Vector{T}            # Smoothing radius (1-elem array)
  vol::Vector{T}              # Volume (1-elem array)
  circulation::Vector{T}      # Scalar circulation (1-elem array)
  static::MVector{1,Bool}        # If true, this particle is not evolved in time

  # Properties
  U::Vector{T}                # Velocity at particle (3-elem array)
  W::Vector{T}                # Vorticity at particle (3-elem array)
  J::Matrix{T}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)
  PSE::Vector{T}              # Particle-strength exchange at particle (3-elem array)

  # Internal variables
  M::Matrix{T}                # 3x3 array of auxiliary memory
  C::Vector{T}                # C[1]=SFS coefficient, C[2]=numerator, C[3]=denominator
  S::Vector{T}                # Stretching term

  # ExaFMM internal variables
#   Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
#   dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
#   dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::MVector{1,Int32}        # Particle index (1-elem array)
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

Base.eltype(::Particle{T}) where T = T
Base.eltype(::AbstractArray{Particle{T}}) where T = T

# Empty initializer
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(zeros(T,3), zeros(T,3),
                                                      zeros(T,1),  zeros(T,1), zeros(T,1),
                                                      init_zero(Bool),
                                                      zeros(T,3), zeros(T,3), zeros(T,3,3), zeros(T,3),
                                                      zeros(T,3,3), zeros(T,3), zeros(T,3),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                    #   zeros(T, 3, 3), zeros(T, 3, 3),
                                                      init_zero(Int32))


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
