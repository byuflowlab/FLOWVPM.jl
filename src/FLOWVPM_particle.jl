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
# PARTICLE STRUCT
################################################################################
struct Particle{T}
  # User inputs
  X::Array{T, 1}                # Position (3-elem array)
  Gamma::Array{T, 1}            # Vectorial circulation (3-elem array)
  sigma::Array{T, 1}            # Smoothing radius (1-elem array)
  vol::Array{T, 1}              # Volume (1-elem array)

  # Properties
  U::Array{T, 1}                # Velocity at particle (3-elem array)
  J::Array{T, 2}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)

  # ExaFMM internal variables
  Jexa::Array{T, 2}             # Jacobian of vectorial potential (9-elem array)
  dJdx1exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx2exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  dJdx3exa::Array{T, 2}         # Derivative of Jacobian (9-elem array)
  index::Array{Int, 1}          # Particle index (1-elem array)
end

# Empty initializer
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(zeros(T, 3), zeros(T, 3),
                                                      zeros(T, 1),  zeros(T, 1),
                                                      zeros(T, 3), zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(Int, 1))

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
                                                fmm.get_Jref(body),
                                                fmm.get_dJdx1ref(body),
                                                fmm.get_dJdx2ref(body),
                                                fmm.get_dJdx3ref(body),
                                                fmm.get_indexref(body))
