#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleField{R<:Real, P<:AbstractParticle, F<:Formulation,
                             V<:ViscousScheme} <: AbstractParticleField{R, P, F, V}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Array{P, 1}                      # Array of particles
    bodies::fmm.Bodies                          # ExaFMM array of bodies
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::R                                        # Current time

    # Solver setting
    kernel::Kernel                              # Vortex particle kernel
    UJ::Function                                # Particle-to-particle calculation

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    transposed::Bool                            # Transposed vortex stretch scheme
    relax::Bool                                 # Activates relaxation scheme
    rlxf::R                                     # Relaxation factor (fraction of dt)
    integration::Function                       # Time integration scheme
    fmm::FMM                                    # Fast-multipole settings

    ParticleField{R, P, F, V}(
                                maxparticles,
                                particles, bodies, formulation, viscous;
                                np=0, nt=0, t=R(0.0),
                                kernel=kernel_default,
                                UJ=UJ_fmm,
                                Uinf=t->zeros(3),
                                transposed=true,
                                relax=true, rlxf=R(0.3),
                                integration=rungekutta3,
                                fmm=FMM()
                         ) where {R, P, F, V} = new(
                                maxparticles,
                                particles, bodies, formulation, viscous,
                                np, nt, t,
                                kernel,
                                UJ,
                                Uinf,
                                transposed,
                                relax, rlxf,
                                integration,
                                fmm
                          )
end

function ParticleField(maxparticles::Int;
                                    formulation::Formulation{PType, R}=formulation_default,
                                    viscous::V=Inviscid(),
                                    optargs...
                            ) where {PType, R, V<:ViscousScheme}
    # Memory allocation by C++
    bodies = fmm.genBodies(maxparticles)

    # Have Julia point to the same memory than C++
    particles = [PType(fmm.getBody(bodies, i-1)) for i in 1:maxparticles]

    # Set index of each particle
    for (i, P) in enumerate(particles)
        P.index[1] = i
    end

    # Generate and return ParticleField
    return ParticleField{R, PType, Formulation{PType, R}, V}(maxparticles, particles, bodies,
                                         formulation, viscous; np=0, optargs...)
end

##### FUNCTIONS ################################################################
"""
  `add_particle(self::ParticleField{ParticleTube}, X, circulation, l0, sigma)`

Add a particle to the field.
"""
function add_particle(self::ParticleField{R, PType, F, V}, X, circulation::Real, l0,
                      sigma; index=-1) where {R, PType<:ParticleTube, F, V}
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    end

    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    P.X .= X
    P.circulation .= abs(circulation)    # Force circulation to be positive
    P.l0 .= sign(circulation)*l0
    P.sigma .= sigma
    P.vol .= pi*sigma.^2 * sqrt(P.l0[1]^2 + P.l0[2]^2 + P.l0[3]^2)
    # P.l .= P.l0
    P.Gamma .= P.l0
    P.Gamma .*= P.circulation[1]
    P.index .= index==-1 ? get_np(self) : index

    # Add particle to the field
    self.np += 1

    return nothing
end

"""
  `add_particle(self::ParticleField, X, Gamma, sigma; vol=0, index=np)`

Add a particle to the field.
"""
function add_particle(self::ParticleField{R, PType, F, V}, X, Gamma, sigma; vol=0,
                                    index=-1) where {R, PType<:Particle, F, V}
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    end

    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    P.X .= X
    P.Gamma .= Gamma
    P.sigma .= sigma
    P.vol .= vol
    P.index .= index==-1 ? get_np(self) : index

    # Add particle to the field
    self.np += 1

    return nothing
end


##### INTERNAL FUNCTIONS #######################################################
function _remove_particle_aux(self::ParticleField{R, PType, F, V}, i
                                            ) where {R, PType<:Particle, F, V}
    return nothing
end

function _remove_particle_aux(self::ParticleField{R, PType, F, V}, i
                                          ) where {R, PType<:ParticleTube, F, V}
    Ptarg = get_particle(self, i)
    Plast = get_particle(self, get_np(self))

    Ptarg.circulation .= Plast.circulation
    Ptarg.l0 .= Plast.l0
    # Ptarg.l .= Plast.l

    return nothing
end
##### END OF PARTICLE FIELD#####################################################
