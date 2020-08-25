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
# ABSTRACT PARTICLE FIELD TYPE
################################################################################

abstract type AbstractParticleField{T, V<:ViscousScheme} end

##### FUNCTIONS ################################################################
"""
    `get_np(pfield::ParticleField)`

    Returns current number of particles in the field.
"""
get_np(self::AbstractParticleField) = self.np

"""
    `get_particle(pfield::ParticleField, i)`

    Returns the i-th particle in the field.
"""
function get_particle(self::AbstractParticleField, i::Int; emptyparticle=false)
    if i<=0
        error("Requested invalid particle index $i")
    elseif !emptyparticle && i>get_np(self)
        error("Requested particle $i, but there is only $(get_np(self))"*
                                                    " particles in the field.")
    elseif emptyparticle && i!=(get_np(self)+1)
        error("Requested empty particle $i, but next empty particle is"*
                                                          " $(get_np(self)+1)")
    end

    return self.particles[i]
end

"Alias for `get_particleiterator`"
iterator(args...; optargs...) = get_particleiterator(args...; optargs...)

"Alias for `get_particleiterator`"
iterate(args...; optargs...) = get_particleiterator(args...; optargs...)

get_X(self::AbstractParticleField, i::Int) = get_particle(self, i).X
get_Gamma(self::AbstractParticleField, i::Int) = get_particle(self, i).Gamma
get_sigma(self::AbstractParticleField, i::Int) = get_particle(self, i).sigma[1]
get_U(self::AbstractParticleField, i::Int) = get_particle(self, i).U
get_W(self::AbstractParticleField, i::Int) = get_W(get_particle(self, i))

"""
    `isinviscid(pfield::AbstractParticleField)`

Returns true if particle field is inviscid.
"""
isinviscid(self::AbstractParticleField) = isinviscid(self.viscous)


##### INTERNAL FUNCTIONS #######################################################

##### END OF ABSTRACT PARTICLE FIELD############################################
