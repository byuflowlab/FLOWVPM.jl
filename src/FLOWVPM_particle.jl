#=##############################################################################
# DESCRIPTION
    Particle struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################





################################################################################
# ABSTRACT PARTICLE TYPE
################################################################################
"""
Implementations of AbstractParticle are expected to have at least the following
properties:

# State variables
* `X::Array{T, 1}`                : Position (3-elem array)
* `Gamma::Array{T, 1}`            : Vectorial circulation (3-elem array)
* `sigma::Array{T, 1}`            : Smoothing radius (1-elem array)
* `vol::Array{T, 1}`              : Volume (1-elem array)

# Public calculations
* `U::Array{T, 1}`                : Velocity at particle (3-elem array)
* `J::Array{T, 2}`                : Jacobian at particle J[i,j]=dUi/dxj (9-elem array)

# Internal variables
* `M::Array{T, 2}`                : 3x3 array of auxiliary memory

# ExaFMM internal variables
* `Jexa::Array{T, 2}`             : Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
* `dJdx1exa::Array{T, 2}`         : Derivative of Jacobian (9-elem array)
* `dJdx2exa::Array{T, 2}`         : Derivative of Jacobian (9-elem array)
* `dJdx3exa::Array{T, 2}`         : Derivative of Jacobian (9-elem array)
* `index::Array{Int32, 1}`        : Particle index (1-elem array)

Also, implementation require the a constructor
`MyParticleType(body::fmm.BodyRef)` returning a particle that is linked with
the C++ Body object `body`.
"""
abstract type AbstractParticle{T} end


# Empty initializer
function Base.zero(::Type{<:AbstractParticle{T}}) where {T}
    error("Implementation doesn't exist!")
end

##### FUNCTIONS ################################################################
get_U(P::AbstractParticle) = P.U
get_W(P::AbstractParticle) = (get_W1(P), get_W2(P), get_W3(P))
get_W1(P::AbstractParticle) = P.J[3,2]-P.J[2,3]
get_W2(P::AbstractParticle) = P.J[1,3]-P.J[3,1]
get_W3(P::AbstractParticle) = P.J[2,1]-P.J[1,2]

##### INTERNAL FUNCTIONS #######################################################
nothing

##### END OF ABSTRACT PARTICLE FIELD############################################





################################################################################
# PARTICLE STRUCT
################################################################################
struct Particle{T} <: AbstractParticle{T}
  # User inputs
  X::Array{T, 1}                # Position (3-elem array)
  Gamma::Array{T, 1}            # Vectorial circulation (3-elem array)
  sigma::Array{T, 1}            # Smoothing radius (1-elem array)
  vol::Array{T, 1}              # Volume (1-elem array)

  # Properties
  U::Array{T, 1}                # Velocity at particle (3-elem array)
  J::Array{T, 2}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)

  # Internal variables
  M::Array{T, 2}                # 3x3 array of auxiliary memory

  # ExaFMM internal variables
  Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
  dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::Array{Int32, 1}        # Particle index (1-elem array)
end

# Empty initializer
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(zeros(T, 3), zeros(T, 3),
                                                      zeros(T, 1),  zeros(T, 1),
                                                      zeros(T, 3), zeros(T, 3, 3),
                                                      zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(Int32, 1))

"""
    `Particle(body::fmm.BodyRef)`

Return a particle that is linked with this C++ Body object. All changes in body
will be reflected in the particles and vice versa.
"""
Particle(body::fmm.BodyRef) = Particle{RealFMM}(fmm.get_Xref(body),
                                                fmm.get_qref(body),
                                                fmm.get_sigmaref(body),
                                                fmm.get_volref(body),
                                                zeros(RealFMM, 3),
                                                zeros(RealFMM, 3, 3),
                                                zeros(RealFMM, 3, 3),
                                                fmm.get_Jref(body),
                                                fmm.get_dJdx1ref(body),
                                                fmm.get_dJdx2ref(body),
                                                fmm.get_dJdx3ref(body),
                                                fmm.get_indexref(body))





################################################################################
# PARTICLE TUBE STRUCT
################################################################################
struct ParticleTube{T} <: AbstractParticle{T}
  # User inputs
  X::Array{T, 1}                # Position (3-elem array)
  circulation::Array{T, 1}      # Scalar circulation (1-elem array)
  l0::Array{T, 1}               # Initial vortex tube length (3-elem array)
  sigma::Array{T, 1}            # Smoothing radius (1-elem array)

  # Properties
  vol::Array{T, 1}              # Volume (1-elem array)
  # l::Array{T, 1}                # Vortex tube length (3-elem array)
  Gamma::Array{T, 1}            # Vectorial circulation (3-elem array)
  U::Array{T, 1}                # Velocity at particle (3-elem array)
  J::Array{T, 2}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)

  # Internal variables
  M::Array{T, 2}                # 3x3 array of auxiliary memory

  # ExaFMM internal variables
  Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array) Jexa[i,j]=dpj/dxi
  dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::Array{Int32, 1}        # Particle index (1-elem array)
end

# Empty initializer
Base.zero(::Type{<:ParticleTube{T}}) where {T} = ParticleTube(
                                                  zeros(T, 3),
                                                  zeros(T, 1), zeros(T, 3),
                                                  zeros(T, 1),
                                                  zeros(T, 1),
                                                  # zeros(T, 3),
                                                  zeros(T, 3),
                                                  zeros(T, 3), zeros(T, 3, 3),
                                                  zeros(T, 3, 5),
                                                  zeros(T, 3, 3), zeros(T, 3, 3),
                                                  zeros(T, 3, 3), zeros(T, 3, 3),
                                                  zeros(Int32, 1))

"""
    `ParticleTube(body::fmm.BodyRef)`

Return a particle that is linked with this C++ Body object. All changes in body
will be reflected in the particles and vice versa.
"""
ParticleTube(body::fmm.BodyRef) = ParticleTube{RealFMM}(fmm.get_Xref(body),
                                                zeros(RealFMM, 1), zeros(RealFMM, 3),
                                                fmm.get_sigmaref(body),
                                                fmm.get_volref(body),
                                                # zeros(RealFMM, 3),
                                                fmm.get_qref(body),
                                                zeros(RealFMM, 3),
                                                zeros(RealFMM, 3, 3),
                                                zeros(RealFMM, 3, 5),
                                                fmm.get_Jref(body),
                                                fmm.get_dJdx1ref(body),
                                                fmm.get_dJdx2ref(body),
                                                fmm.get_dJdx3ref(body),
                                                fmm.get_indexref(body))
