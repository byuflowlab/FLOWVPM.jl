#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sept 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# ABSTRACT PARTICLE FIELD TYPE
################################################################################

abstract type AbstractParticleField{P<:AbstractParticle, F<:Formulation, V<:ViscousScheme} end

##### FUNCTIONS ################################################################
"""
    `get_np(pfield::AbstractParticleField)`

    Returns current number of particles in the field.
"""
get_np(self::AbstractParticleField) = self.np

"""
    `get_particle(pfield::AbstractParticleField, i)`

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


"""
    `get_particleiterator(pfield::AbstractParticleField; start_i=1, end_i=np)`

Return an iterator over particles that can be used as follows

```julia-repl
julia> # Initiate particle field
       pfield = FLOWVPM.ParticleField(10);

julia> # Add particles
       for i in 1:4
           FLOWVPM.add_particle(pfield, (i*10^0, i*10^1, i*10^2), zeros(3), 1.0)
       end

julia> # Iterate over particles
       for P in FLOWVPM.get_particleiterator(pfield)
           println(P.X)
       end
[1.0, 10.0, 100.0]
[2.0, 20.0, 200.0]
[3.0, 30.0, 300.0]
[4.0, 40.0, 400.0]
```
"""
function get_particleiterator(self::AbstractParticleField{PType, F, V}; start_i::Int=1,
                              end_i::Int=-1, reverse=false ) where {PType, F, V}
    # ERROR CASES
    if end_i > get_np(self)
        error("Requested end_i=$(end_i), but there is only $(get_np(self))"*
                                                    " particles in the field.")
    end

    strt = reverse ? (end_i==-1 ? get_np(self) : end_i) : start_i
    stp = reverse ? -1 : 1
    nd = reverse ? start_i : (end_i==-1 ? get_np(self) : end_i)

    return view( self.particles, strt:stp:nd
                )::SubArray{PType, 1, Array{PType, 1}, Tuple{StepRange{Int64,Int64}}, true}
end

"""
  `remove_particle(pfield::AbstractParticleField, i)`

Remove the i-th particle in the field. This is done by moving the last particle
that entered the field into the memory slot of the target particle. To remove
particles sequentally, you will need to go from the last particle back to the
first one (see documentation of `get_particleiterator` for an example).
"""
function remove_particle(self::AbstractParticleField, i::Int)
    if i<=0
        error("Requested removal of invalid particle index $i")
    elseif i>get_np(self)
        error("Requested removal of particle $i, but there is only"*
                                " $(get_np(self)) particles in the field.")
    end

    if i != get_np(self)
        # Overwrite target particle with last particle in the field
        fmm.overwriteBody(self.bodies, i-1, get_np(self)-1)
    end

    _remove_particle_aux(self, i)

    # Remove last particle in the field
    self.np -= 1

    return nothing
end


"""
  `nextstep(self::AbstractParticleField, dt; relax=false)`

Steps the particle field in time by a step `dt`.
"""
function nextstep(self::AbstractParticleField, dt::Real; optargs...)

    # Step in time
    if get_np(self)!=0
        self.integration(self, dt; optargs...)
    end

    # Updates time
    self.t += dt
    self.nt += 1
end


##### INTERNAL FUNCTIONS #######################################################
function _remove_particle_aux(self::AbstractParticleField, i)
    error("Method not implemented!")
end

function _reset_particles(self::AbstractParticleField{<:AbstractParticle{T}, F, V}) where {T, F, V}
    tzero = zero(T)
    for P in iterator(self)
        P.U[1] = tzero
        P.U[2] = tzero
        P.U[3] = tzero
        P.J[1, 1] = tzero
        P.J[2, 1] = tzero
        P.J[3, 1] = tzero
        P.J[1, 2] = tzero
        P.J[2, 2] = tzero
        P.J[3, 2] = tzero
        P.J[1, 3] = tzero
        P.J[2, 3] = tzero
        P.J[3, 3] = tzero
    end
end
##### END OF ABSTRACT PARTICLE FIELD############################################
