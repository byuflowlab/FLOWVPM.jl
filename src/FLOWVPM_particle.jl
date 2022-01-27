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
struct Particle{T} <: AbstractArray{T,1}
  # User inputs
  X::Array{T, 1}                # Position (3-elem array)
  Gamma::Array{T, 1}            # Vectorial circulation (3-elem array)
  sigma::Array{T, 1}            # Smoothing radius (1-elem array)
  vol::Array{T, 1}              # Volume (1-elem array)
  circulation::Array{T, 1}      # Scalar circulation (1-elem array)

  # Properties
  U::Array{T, 1}                # Velocity at particle (3-elem array)
  J::Array{T, 2}                # Jacobian at particle J[i,j]=dUi/dxj (9-elem array)
  PSE::Array{T, 1}              # Particle-strength exchange at particle (3-elem array)

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
                                                      zeros(T, 1),
                                                      zeros(T, 3), zeros(T, 3, 3), zeros(T, 3),
                                                      zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(T, 3, 3), zeros(T, 3, 3),
                                                      zeros(Int32, 1))

"""
    `Particle(body::fmm.BodyRef)`

Return a particle that is linked with this C++ Body object. All changes in body
will be reflected in the particles and vice versa.
"""
Particle(body::fmm.BodyRef;R=RealFMM) = Particle{R}(fmm.get_Xref(body),
                                                fmm.get_qref(body),
                                                fmm.get_sigmaref(body),
                                                fmm.get_volref(body),
                                                zeros(RealFMM, 1),
                                                zeros(RealFMM, 3),
                                                zeros(RealFMM, 3, 3),
                                                fmm.get_pseref(body),
                                                zeros(RealFMM, 3, 3),
                                                fmm.get_Jref(body),
                                                fmm.get_dJdx1ref(body),
                                                fmm.get_dJdx2ref(body),
                                                fmm.get_dJdx3ref(body),
                                                fmm.get_indexref(body))


##### FUNCTIONS ################################################################
get_U(P::Particle) = P.U

get_W(P::Particle) = (get_W1(P), get_W2(P), get_W3(P))
get_W1(P::Particle) = P.J[3,2]-P.J[2,3]
get_W2(P::Particle) = P.J[1,3]-P.J[3,1]
get_W3(P::Particle) = P.J[2,1]-P.J[1,2]

get_SGS1(P::Particle{T}) where {T} = getproperty(P, _SGS)[1]::T
get_SGS2(P::Particle{T}) where {T} = getproperty(P, _SGS)[2]::T
get_SGS3(P::Particle{T}) where {T} = getproperty(P, _SGS)[3]::T
add_SGS1(P::Particle{T}, val) where {T} = getproperty(P, _SGS)[1]::T += val
add_SGS2(P::Particle{T}, val) where {T} = getproperty(P, _SGS)[2]::T += val
add_SGS3(P::Particle{T}, val) where {T} = getproperty(P, _SGS)[3]::T += val

################################################################################
# Eric Green's additions for DifferentialEquations.jl compatibility
################################################################################

# Additional methods for arithmatic operations so that DE.jl can run time integration. The details of this are taken from the array interface documentation, since there's
# a very similar case in one of their examples (a struct with array-type data as well as other data that needs to be properly carried over)

# Functions defined to enable using a custom array type (from the docs: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting):

"""
    'Base.size(p::Particle)'
Hardcoded to output 9, since this is the number of entries that the DifferentialEquations solver needs access to.
"""
Base.size(p::Particle) = 9

"""
    'Base.getindex(p::Particle,i)'
Maps a single-index access to the various arrays contained in a Particle.
"""
Base.getindex(p::Particle{T},i::Int) where {T} = 0<i<4 ? p.X[i] : (3<i<7 ? p.Gamma[i-3] : (i == 7 ? p.sigma[1] : (i == 8 ? p.vol[1] : (i == 9 ? p.circulation[1] : nothing))))

"""
    'Base.setindex!(p::Particle,val,inds)'
Maps single-indexed variable assignment to the various arrays contained in a Particle.
"""
function Base.setindex!(p::Particle{T},val,inds::Vararg{Int,1}) where {T}
  i = inds[1]
  0<i<4 ? p.X[i] = val : (3<i<7 ? p.Gamma[i-3] = val : (i == 7 ? p.sigma[1] = val : (i == 8 ? p.vol[1] = val : (i == 9 ? p.circulation[1] = val : nothing))))
end

Base.showarg(io::IO,p::Particle,toplevel) = print(io,typeof(p)) # Custom printing

### These next few functions should not be used directly; rather, it is used to implement broadcasting any function onto Particle objects.
# The general format is copied from https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting).
Base.BroadcastStyle(::Type{<:Particle}) = Broadcast.ArrayStyle{Particle}()
"""
    'Base.similar(bc::Broadcasted)'
Unpacks broadcasted input and constructs a new Particle of similar structure.
"""
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Particle}},::Type{ElType}) where {ElType}
    p = find_p(bc)
    out = copy(p)
    out[1:9] .= zero(ElType)
    return out
end
"""
    'find_p(bc)'
Unpacks broadcasted input. This function should probably not be called directly.
"""
find_p(bc::Base.Broadcast.Broadcasted) = find_p(bc.args)
find_p(args::Tuple) = find_p(find_p(args[1],Base.tail(args)))
find_p(x) = x
find_p(::Tuple{}) = nothing
find_p(p::Particle,rest) = p
find_p(::Any,rest) = find_p(rest)

# Some additional similar() definitions to avoid unwanted type conversion
"""
    'Base.similar(p::Particle)'
Creates a new particle similar to p but with the numerical entries used by a differential equations solver initialized to zero.
"""
function Base.similar(p::Particle{T}) where T
  out = copy(p)
  out[1:9] = zero(T)
  return out
end

"""
    'Base.similar(p::Particle,T2::Type)'
Creates a new particle similar to p but with the numerical entries used by a differential equations solver initialized to zero.
Also converts the base type of the output particle to T2.
"""
function Base.similar(p::Particle,T2::Type)
  out = copy(p)
  out[1:9] = zero(T2)
  return out
end

"""
    'Particle(val::T)'
Constructor to handle initializing a Particle with a single numerical value; this is done a few times in the DifferentialEquations.jl
internal function calls.
"""
function Particle(val::T) where {T<:Real}
  p = zero(Particle{T})
  p.X .= val
  p.Gamma .= val
  p.sigma .= val
  p.vol .= val
  p.circulation .= val
  return p
end

"""
    Base.axes(p::Particle)
Returns a range of allowable indices for accessing Particle data. Hardcoded to output (Base.OneTo(9),) since Particle types always have the same size.
"""
Base.axes(p::Particle) = (Base.OneTo(9),)

"""
    'Base.copy(p::Particle)'
Creates a copy of a Particle.
"""
function Base.copy(p::Particle{T}) where T

  out = zero(Particle{T})
  out.X .= p.X
  out.Gamma .= p.Gamma
  out.sigma .= p.sigma
  out.vol .= p.vol
  out.circulation .= p.circulation
  out.index .= p.index
  return out

end

##### INTERNAL FUNCTIONS #######################################################
nothing

##### END OF ABSTRACT PARTICLE FIELD############################################
