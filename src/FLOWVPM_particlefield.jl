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
mutable struct ParticleField
    # User inputs
    maxparticles::Int                           # Maximum number of particles

    # Internal properties
    bodies::fmm.Bodies                          # ExaFMM array of bodies (particles)
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::Float64                                  # Current time

    ParticleField(
                      maxparticles;
                      bodies=fmm.genBodies(maxparticles), np=0,
                      nt=0, t=0.0
                 ) = new(
                      maxparticles,
                      bodies, np,
                      nt, t
                 )
end
##### FUNCTIONS ################################################################
"""
    `get_np(pfield::ParticleField)`

    Returns current number of particles in the field.
"""
get_np(self::ParticleField) = self.np

"""
    `get_particle(pfield::ParticleField, i)`

    Returns the i-th particle in the field.
"""
function get_particle(self::ParticleField, i::Int; emptyparticle=false)
    if i<=0
        error("Requested invalid particle index $i")
    elseif !emptyparticle && i>get_np(self)
        error("Requested particle $i, but there is only $(get_np(self))"*
                                                    " particles in the field.")
    elseif emptyparticle && i!=(get_np(self)+1)
        error("Requested empty particle $i, but next empty particle is"*
                                                          " $(get_np(self)+1)")
    end

    return fmm.getBody(self.bodies, i-1)
end

"""
    `set_X(pfield::ParticleField, X, i)`

    Set position of the i-th particle in the field.
"""
set_X(self::ParticleField, X, i) = set_X(get_particle(self, i), X)

"""
    `get_X(pfield::ParticleField, i)`

    Returns position of the i-th particle in the field.
"""
get_X(self::ParticleField, i) = get_X(get_particle(self, i))

"""
    `set_Gamma(pfield::ParticleField, Gamma, i)`

    Set vortex strength of the i-th particle in the field.
"""
set_Gamma(self::ParticleField, Gamma, i) = set_Gamma(get_particle(self, i), Gamma)

"""
    `get_Gamma(pfield::ParticleField, i)`

    Returns vortex strength of the i-th particle in the field.
"""
get_Gamma(self::ParticleField, i) = get_Gamma(get_particle(self, i))

"""
    `set_sigma(pfield::ParticleField, sigma, i)`

    Set smoothing radius of the i-th particle in the field.
"""
set_sigma(self::ParticleField, sigma, i) = set_sigma(get_particle(self, i), sigma)

"""
    `get_sigma(pfield::ParticleField, i)`

    Returns vortex strength of the i-th particle in the field.
"""
get_sigma(self::ParticleField, i) = get_sigma(get_particle(self, i))

"""
    `set_vol(pfield::ParticleField, vol, i)`

    Set volume of the i-th particle in the field.
"""
set_vol(self::ParticleField, vol, i) = set_vol(get_particle(self, i), vol)

"""
    `get_vol(pfield::ParticleField, i)`

    Returns vortex strength of the i-th particle in the field.
"""
get_vol(self::ParticleField, i) = get_vol(get_particle(self, i))

"""
  `add_particle(self::ParticleField, particle::Particle)`

Add a particle to the field.
"""
function add_particle(self::ParticleField, X, Gamma, sigma; vol=0)
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    end


    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    set_X(P, X)
    set_Gamma(P, Gamma)
    set_sigma(P, sigma)
    set_vol(P, vol)

    # Add particle to the field
    self.np += 1

    return nothing
end

add_particle(self, X, Gamma, sigma, vol) = add_particle(self, X, Gamma, sigma; vol=vol)

"""
  `remove_particle(pfield::ParticleField, i)`

Remove the i-th particle in the field. This is done by moving the last particle
that entered the field into the memory slot of the target particle. To remove
particles sequentally, you will need to go from the last particle back to the
first one.
"""
function remove_particle(self::ParticleField, i::Int)
    if i<=0
        error("Requested removal of invalid particle index $i")
    elseif i>get_np(self)
        error("Requested removal of particle $i, but there is only"*
                                " $(get_np(self)) particles in the field.")
    end

    # Overwrite target particle with last particle in the field
    fmm.overwriteBody(self.bodies, i-1, get_np(self)-1)

    # Remove last particle in the field
    self.np -= 1

    return nothing
end



##### INTERNAL FUNCTIONS #######################################################
function set_X(P::fmm.BodyRef, X)
    if length(X)!=3
        error("Invalid position $(X).")
    end

    fmm.get_Xref(P)[:] .= X

    return nothing
end

function set_Gamma(P::fmm.BodyRef, Gamma)
    if length(Gamma)!=3
        error("Invalid strength $(Gamma).")
    end

    fmm.get_qref(P)[:] .= Gamma

    return nothing
end

function set_sigma(P::fmm.BodyRef, sigma::Real)
    if sigma<=0
        error("Invalid smoothing radius $(sigma).")
    end

    fmm.set_sigma(P, sigma)

    return nothing
end

function set_vol(P::fmm.BodyRef, vol::Real)
    fmm.set_vol(P, vol)
    return nothing
end

function set_index(P::fmm.BodyRef, index::Int)
    fmm.set_index(P, Int32(index))
    return nothing
end
set_index(self::ParticleField, index, i) = set_index(get_particle(self, i), index)
get_index(self::ParticleField, i) = get_index(get_particle(self, i))

get_X(P::fmm.BodyRef) = fmm.get_Xref(P)
get_Gamma(P::fmm.BodyRef) = fmm.get_qref(P)
get_sigma(P::fmm.BodyRef) = fmm.get_sigma(P)
get_vol(P::fmm.BodyRef) = fmm.get_vol(P)
get_J(P::fmm.BodyRef) = fmm.get_Jref(P)
get_dJdx1(P::fmm.BodyRef) = fmm.get_dJdx1ref(P)
get_dJdx2(P::fmm.BodyRef) = fmm.get_dJdx2ref(P)
get_dJdx3(P::fmm.BodyRef) = fmm.get_dJdx3ref(P)
get_index(P::fmm.BodyRef) = fmm.get_index(P)
##### END OF PARTICLE FIELD#####################################################
