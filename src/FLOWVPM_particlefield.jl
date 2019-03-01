#=##############################################################################
# DESCRIPTION
    Particle field struct definition.
# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Feb 2019
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# PARTICLE STRUCT
################################################################################
"""
    `Particle{T}(X1::T, X2::T, X3::T, G1::T, G2::T, G3::T, s::T, vol::T,
id::Int64)`

Vortex particle with position `[X1, X2, X3]`, vectorial circulation Γ
`[G1, G2, G3]`, smoothing radius σ `s`, volume `vol`, and identifier `id`.
"""
struct Particle{T<:Real}
  X1::T                                     # Position
  X2::T
  X3::T
  G1::T                                     # Vectorial circulation
  G2::T
  G3::T
  s::T                                      # Smoothing radius
  vol::T                                    # Volume
  id::Int64                                 # Identifier
end

"""
    `eltype(ParticleType)`
Returns the element type of a Particle type.
"""
Base.eltype(::Type{<:Particle{T}}) where {T} = T

"""
    `zero(ParticleType)`
Returns a particle of the requested type initiated with zero values.
"""
Base.zero(::Type{<:Particle{T}}) where {T} = Particle(zero(T), zero(T), zero(T),
                                                 zero(T), zero(T), zero(T),
                                                 zero(T), zero(T), 0)


# Types of Particle Arrays
const CPUArrParticles = Union{
                                Array{P, 1},
                                SubArray{P, 1, Array{P,1}, Tuple{UnitRange{Int64}}, true},
                              } where {P<:Particle}
const GPUArrParticles = GPUenabled ? Union{
                                CuArray{P, 1},
                                SubArray{P, 1, CuArray{P,1}, Tuple{UnitRange{Int64}}, true},
                              } where {P<:Particle} : Union{}
const ArrParticles = Union{CPUArrParticles, GPUArrParticles}

##### END OF PARTICLE ##########################################################








################################################################################
# FMM STRUCT
################################################################################
"Structure of cells"
mutable struct Cell{T<:Real, ArrP<:ArrParticles}
  numChildren::Int64                          # Number of child cells
  numBodies::Int64                            # Number of descendant bodies
                                              # All children cells
  children::SubArray{Cell{T, ArrP}, 1, Array{Cell{T, ArrP},1}, Tuple{UnitRange{Int64}}, true}
                                              # All descendant bodies in this cell
  bodies::SubArray{eltype(ArrP), 1, ArrP, Tuple{UnitRange{Int64}}, true}
  X::Array{T, 1}                              # Cell center
  R::T                                        # Cell radius
  sigma::T                                    # Cell characteristic smoothing radius
  M::Array{Complex{T}, 2}                     # Multipole expansion vectorized coefs
  L::Array{Complex{T}, 2}                     # Local expansion vectorized coefs

  Cell(;
        numChildren=0, numBodies=0,
        children=view(Cell{T, ArrP}[], 1:0), bodies=view(ArrP(), 1:0),
        X=zeros(T, 3), R=zero(T), sigma=zero(T),
        M=zeros(Complex{T}, 0, 0), L=zeros(Complex{T}, 0, 0)
           ) = new{eltype(ArrP), T, ArrP}(
       numChildren, numBodies,
       children, bodies,
       X, R, sigma,
       M, L
            )
end

# @warn("Don't forget to initiate cells!")
function init_cell(cell::Cell{T}, P::Int64) where {T<:Real}
  nterms::Int64 = P*(P+1)/2
  cell.M = zeros(Complex{T}, 3, nterms)
  cell.L = zeros(Complex{T}, 3, nterms)
end

init_cell(cells::Array{Cell, 1}, P::Int64) = init_cell.(cells, P::Int64)

mutable struct FMM{T<:Real, CType<:Cell}

  # Optional user inputs
  p::Int64                        # Multipole expansion order
  ncrit::Int64                    # Max number of particles per leaf
  theta::T                        # Neighborhood criterion
  phi::T                          # Regularizing neighborhood criterion
  p2p::Function                   # UJ function to use for P2P

  # Properties
  cells::Array{CType, 1}           # Cells

  FMM(

        p=4, ncrit=50, theta=0.4, phi=0.5,
        p2p=UJ_direct,
        cells=Array{CType,1}()
     ) = new{T, CType}(
        p, ncrit, theta, phi,
        p2p,
        cells
     )
end

# FMM() =

init_cells(fmm::FMM) = init_cell(fmm.cells, fmm.p)

##### END OF FMM ###############################################################





################################################################################
# PARTICLE FIELD STRUCT
################################################################################
"""
  `ParticleField(nu, kernel, UJfun)`

Defines a particle field where particles can be added through the function
`addparticle`.

**Arguments**
  * `nu::Real`              : Kinematic viscosity of the field.
  * `kernel::Kernel`        : Vortex partcle kernel. Available are `kernel_sing`,
                              `kernel_gaus`, `kernel_wnklmns`, `kernel_gauserf`.
  * `UJfun::Function`       : Method for calculating interactions between
                              particles (velocity U and jacobian J). Available
                              are `UJ_direct`, and `UJ_FMM`.

**Optional Arguments**
  * `Uinf::Function`        : Uniform freestream function Uinf(t).
  * `transposed::Bool`      : Transposed vortex stretch scheme. True by default.
  * `relax::Bool`           : Activates relaxation scheme. True by default.
  * `rlxf::Real`            : Relaxation factor (fraction of dt). 0.3 by default.
  * `integration::String`   : Time integration scheme. "euler" by default.
  * `fmm::FMM`              : Fast Multipole Method configurations.

**Properties**
  * `particles:: Array{Particle, 1}`      : Particles in the field.
  * `Nmax::Int64`                         : Maximum number of particles.
  * `Np::Int64`                           : Current number of particles.
  * `t::Real`                             : Current time.
  * `nt::Int64`                           : Current time step number.

"""
mutable struct ParticleField{ArrP<:ArrParticles, TUJ<:AbstractArray, T<:Real}

  # User inputs
  particles::ArrP                         # Particles in the field
  UJ::TUJ                                 # Stores velocity and jacobian here
  nu::T                                   # Kinematic viscosity
  kernel::Kernel                          # Vortex particle kernel
  UJfun::Function                         # Particle-to-particle calculation

  # Optional inputs
  Uinf::Function                          # Uniform freestream function Uinf(t)
  transposed::Bool                        # Transposed vortex stretch scheme
  relax::Bool                             # Activates relaxation scheme
  rlxf::T                                 # Relaxation factor (fraction of dt)
  integration::String                     # Time integration scheme
  # fmm::FMM

  # Properties
  Nmax::Int64                             # Maximum number of particles
  Np::Int64                               # Current number of particles
  t::T                                    # Current time
  nt::Int64                               # Current time step number

  ParticleField(
                particles, UJ,
                nu, kernel, UJfun;
                Uinf=t->zeros(3),
                  transposed=true,
                  relax=true, rlxf=typeof(nu)(0.3),
                  integration="euler",
                  # fmm=FMM{typeof(nu), Cell{typeof(nu), typeof(particles)}}(),
                Nmax=length(particles), Np=0,
                  t=zero(typeof(nu)), nt=0
               ) = _check(particles, UJ) ? new{typeof(particles), typeof(UJ), typeof(nu)}(
                 particles, UJ,
                 nu, kernel, UJfun,
                 Uinf,
                  transposed,
                  relax, rlxf,
                  integration,
                  # fmm,
                 Nmax, Np,
                   t, nt
                ) : error("Logic error")
end


mutable struct Test{ArrP<:ArrParticles, TUJ<:AbstractArray, T<:Real}

  # User inputs
  particles::ArrP                         # Particles in the field
  UJ::TUJ                                 # Stores velocity and jacobian here
  nu::T                                   # Kinematic viscosity

   # Test( particles, UJ, nu ) = new{ArrP, TUJ, T}( particles, UJ, nu )
    Test( particles, UJ, nu ) = new{typeof(particles), typeof(UJ), typeof(nu)}( particles, UJ, nu )
end

function ParticleField(PType::Type{<:Particle{T}}, Nmax::Int64, args...; optargs...) where {T<:Real}
    return ParticleField(
        zeros(PType, Nmax), zeros(eltype(PType), 12, Nmax),     args...; optargs...
    )
end

if GPUenabled
    """
    Returns a particle field allocated in the GPU memory. This function receives
    the same arguments than `ParticleField()` and has the same properties.
    """
    function ParticleFieldGPU(PType::Type{<:Particle{T}}, Nmax::Int64, args...; optargs...) where {T<:Real}
        return ParticleField(
            cufill(zero(PType), Nmax), cufill(zero(eltype(PType)), 12, Nmax), args...; optargs...
        )
    end
end

#### INTERNAL FUNCTIONS ########################################################
function _check(particles::ArrParticles, UJ::AbstractArray)
    if length(particles) != size(UJ, 2)
        error("Got invalid particles-UJ pair."*
                " length(particles) != size(UJ, 2)"*
                " ($(length(particles)) != $(size(UJ, 2))).")
        return false
    elseif eltype(eltype(particles)) != eltype(UJ)
        error("Got invalid particles-UJ pair."*
                  " eltype(eltype(particles)) != eltype(UJ)"*
                  " ($(eltype(eltype(particles))) != $(eltype(UJ))).")
        return false
    elseif size(UJ, 1) != 12
        error("Got invalid UJ matrix."*
                              " Expected 1st dim length 12, got $(size(UJ,1)).")
        return false
    else
        return true
    end
end
