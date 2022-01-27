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
# Eric Green: "The abstractarray subtyping tells Julia to treat this like a 2D matrix. This lets DifferentialEquations.jl work with it if the appropriate matrix interface is added."
mutable struct ParticleField{R<:Real, F<:Formulation, V<:ViscousScheme} <: AbstractArray{R,2}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Array{Particle{R}, 1}            # Array of particles
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
    sgsmodel::Function                          # Subgrid-scale contributions model
    sgsscaling::Function                        # Scaling factor of SGS contributions
    integration::Function                       # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    relaxation::Function                        # Relaxation scheme
    relax::Bool                                 # Enables relaxation scheme
    rlxf::R                                     # Relaxation factor (fraction of dt)
    fmm::FMM                                    # Fast-multipole settings

    # Internal memory for computation
    M::Array{R, 1}

    ParticleField{R, F, V}(
                                maxparticles,
                                particles, bodies, formulation, viscous;
                                np=0, nt=0, t=R(0.0),
                                kernel=kernel_default,
                                UJ=UJ_fmm,
                                Uinf=Uinf_default,
                                sgsmodel=sgs_default,
                                sgsscaling=sgs_scaling_default,
                                integration=rungekutta3,
                                transposed=true,
                                relaxation=relaxation_default,
                                relax=true, rlxf=R(0.3),
                                fmm=FMM(),
                                M=zeros(R, 4)
                         ) where {R, F, V} = new(
                                maxparticles,
                                particles, bodies, formulation, viscous,
                                np, nt, t,
                                kernel,
                                UJ,
                                Uinf,
                                sgsmodel,
                                sgsscaling,
                                integration,
                                transposed,
                                relaxation,
                                relax, rlxf,
                                fmm,
                                M
                          )
end

function ParticleField(maxparticles::Int;
                                    formulation::F=formulation_default,
                                    viscous::V=Inviscid(),
                                    R=RealFMM,
                                    optargs...
                            ) where {F, V<:ViscousScheme}
    # Memory allocation by C++
    bodies = fmm.genBodies(maxparticles)

    # Have Julia point to the same memory than C++
    particles = [Particle(fmm.getBody(bodies, i-1);R=R) for i in 1:maxparticles]

    # Set index of each particle
    for (i, P) in enumerate(particles)
        P.index[1] = i
    end
    # Generate and return ParticleField
    return ParticleField{R, F, V}(maxparticles, particles, bodies,
                                         formulation, viscous; np=0, optargs...)
end

##### FUNCTIONS ################################################################
"""
  `add_particle(self::ParticleField, X, Gamma, sigma; vol=0, index=np)`

Add a particle to the field.
"""
function add_particle(self::ParticleField, X, Gamma, sigma;
                                           vol=0, circulation::Real=1, index=-1)
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    elseif circulation<=0
        error("Got invalid circulation less or equal to zero! ($(circulation))")
    end

    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    P.X .= X
    P.Gamma .= Gamma
    P.sigma .= sigma
    P.vol .= vol
    P.circulation .= abs(circulation)
    P.index .= index==-1 ? get_np(self) : index

    # Add particle to the field
    self.np += 1

    return nothing
end

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

    return self.particles[i]
end

"Alias for `get_particleiterator`"
iterator(args...; optargs...) = get_particleiterator(args...; optargs...)

"Alias for `get_particleiterator`"
iterate(args...; optargs...) = get_particleiterator(args...; optargs...)

get_X(self::ParticleField, i::Int) = get_particle(self, i).X
get_Gamma(self::ParticleField, i::Int) = get_particle(self, i).Gamma
get_sigma(self::ParticleField, i::Int) = get_particle(self, i).sigma[1]
get_U(self::ParticleField, i::Int) = get_particle(self, i).U
get_W(self::ParticleField, i::Int) = get_W(get_particle(self, i))

"""
    `isinviscid(pfield::ParticleField)`

Returns true if particle field is inviscid.
"""
isinviscid(self::ParticleField) = isinviscid(self.viscous)


"""
    `get_particleiterator(pfield::ParticleField; start_i=1, end_i=np)`

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
function get_particleiterator(self::ParticleField{R, F, V}; start_i::Int=1,
                              end_i::Int=-1, reverse=false ) where {R, F, V}
    # ERROR CASES
    if end_i > get_np(self)
        error("Requested end_i=$(end_i), but there is only $(get_np(self))"*
                                                    " particles in the field.")
    end

    strt = reverse ? (end_i==-1 ? get_np(self) : end_i) : start_i
    stp = reverse ? -1 : 1
    nd = reverse ? start_i : (end_i==-1 ? get_np(self) : end_i)

    return view( self.particles, strt:stp:nd
                )::SubArray{Particle{R}, 1, Array{Particle{R}, 1}, Tuple{StepRange{Int64,Int64}}, true}
end

"""
  `remove_particle(pfield::ParticleField, i)`

Remove the i-th particle in the field. This is done by moving the last particle
that entered the field into the memory slot of the target particle. To remove
particles sequentally, you will need to go from the last particle back to the
first one (see documentation of `get_particleiterator` for an example).
"""
function remove_particle(self::ParticleField, i::Int)
    if i<=0
        error("Requested removal of invalid particle index $i")
    elseif i>get_np(self)
        error("Requested removal of particle $i, but there is only"*
                                " $(get_np(self)) particles in the field.")
    end

    Plast = get_particle(self, get_np(self))

    if i != get_np(self)
        # Overwrite target particle with last particle in the field
        fmm.overwriteBody(self.bodies, i-1, get_np(self)-1)

        Ptarg = get_particle(self, i)
        Ptarg.circulation .= Plast.circulation
    end

    # Remove last particle in the field
    _reset_particle(Plast)
    _reset_particle_sgs(Plast)
    self.np -= 1

    return nothing
end


"""
  `nextstep(self::ParticleField, dt; relax=false)`

Steps the particle field in time by a step `dt`.
"""
function nextstep(self::ParticleField, dt::Real; optargs...)

    # Step in time
    if get_np(self)!=0
        self.integration(self, dt; optargs...)
    end

    # Updates time
    self.t += dt
    self.nt += 1
end


##### INTERNAL FUNCTIONS #######################################################
function _reset_particles(self::ParticleField{R, F, V}) where {R, F, V}
    tzero = zero(R)
    for P in iterator(self)
        _reset_particle(P, tzero)
    end
end

function _reset_particle(P::Particle{T}, tzero::T) where {T}
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

    P.PSE[1] = tzero
    P.PSE[2] = tzero
    P.PSE[3] = tzero
end
_reset_particle(P::Particle{T}) where {T} = _reset_particle(P, zero(T))

function _reset_particles_sgs(self::ParticleField{R, F, V}) where {R, F, V}
    tzero = zero(R)
    for P in iterator(self)
        _reset_particle_sgs(P, tzero)
    end
end

function _reset_particle_sgs(P::Particle{T}, tzero::T) where {T}
    getproperty(P, _SGS)::Array{T, 2} .= tzero
end
_reset_particle_sgs(P::Particle{T}) where {T} = _reset_particle_sgs(P, zero(T))

################################################################################
################################################################################
################################################################################

# Eric Green's additions:

# Additional methods for arithmatic operations so that DE.jl can run time integration. The details of this are taken from the array interface documentation, since there's
# a very similar case in one of their examples (a struct with array-type data as well as other data that needs to be properly carried over)

# Functions defined to enable using a custom array type (from the docs: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting):

# As with the particle struct, these functions allow broadcasting to work on the PointList type.

"""
    'Base.size(pfield::ParticleField)'

Returns the size of the particle field as a 2-entry tuple. The first entry is the number of particles and the second is the number
of data values that should be updated by a time-stepping algorithm (i.e. 9).

"""
Base.size(pfield::ParticleField) = (pfield.np,9) # point size hardcoded at 9, since there are 9 entries that the time-stepping algorithm sees.

"""
    'Base.getindex(pfield::ParticleField{R,F,V},inds)'
Allows array-like access on ParticleField objects. Required for DifferentialEquations.jl to be able to access ParticleField data correctly.
Example: pfield[1,2] will return entry 2 (i.e. X[2]) from the first particle in pfield
"""
Base.getindex(pfield::ParticleField{R,F,V},inds::Vararg{Int,1}) where {R,F,V} = pfield.particles[Int(ceil(inds[1]/size(pfield)[2]))][(inds[1]-1)%size(pfield)[2]+1] # converts a single-index array access into an access of particle data
function Base.getindex(pfield::ParticleField{R,F,V},inds::Vararg{Int,2}) where {R,F,V} # slightly longer definition to account for writing to a not-yet-defined pfield
    if isassigned(pfield.particles)
        pfield.particles[inds[1]][inds[2]]
    else
        pfield = zero(ParticleField(pfield.np;formulation = pfield.formulation,viscous = pfield.viscous,R = R))
        pfield[inds...] # recursive call
    end
end

"""
    'Base.setindex!(pfield::ParticleField,val,inds)'
Sets the element at pfield[inds...] to val.
"""
Base.setindex!(pfield::ParticleField{R,F,V},val,inds::Vararg{Int,1}) where {R,F,V} = pfield.particles[Int(ceil(inds[1]/size(pfield)[2]))][(inds[1]-1)%size(pfield)[2]+1] = val
function Base.setindex!(pfield::ParticleField{R,F,V},val,inds::Vararg{Int,2}) where {R,F,V}
    if isassigned(pfield.particles)
        ###pfield.particles[inds[1]][inds[2]] = copy(val)
        pfield.particles[inds[1]][inds[2]] = val
    else
        pfield = ParticleField(pfield.maxparticles;formulation = pfield.formulation,viscous = pfield.viscous,R = R)
        ###pfield.particles[inds[1]][inds[2]] = copy(val)
        pfield.particles[inds[1]][inds[2]] = val
    end
end
function Base.setindex!(pfield::ParticleField{R,F,V},val,inds::Vararg{Int,N}) where {R,F,V,N}
    error("error: particle field index out of bounds!") # error case: access with 3 or more indices
end

# Base.showarg(io::IO,pfield::ParticleField,toplevel) = print(io,"output text") # custom printing function, if desired

### These next few functions should not be used directly; rather, it is used to implement broadcasting any function onto ParticleField objects.
# The general format is copied from https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting).
Base.BroadcastStyle(::Type{<:ParticleField}) = Broadcast.ArrayStyle{ParticleField}()
"""
    'Base.similar(bc::Broadcasted)'
Unpacks broadcasted input and constructs a new ParticleField of similar structure.
"""
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ParticleField}},::Type{ElType}) where {ElType}
    pfield = find_pfield(bc)
    pfield_out = ParticleField(pfield.np;formulation = pfield.formulation,viscous = pfield.viscous,R = ElType)
    # copies over all the settings
    pfield_out.maxparticles = pfield.maxparticles
    for i=1:pfield.np
        p = pfield.particles[i]
        add_particle(pfield_out,ElType[0,0,0],ElType[0,0,0],zero(ElType);vol = zero(ElType),circulation = 1e-12*one(ElType),index = p.index)
    end
    copy_settings!(pfield_out,pfield)
    return pfield_out
end
"""
    'find_pfield(bc)'
Unpacks broadcasted input. This function should probably not be called directly.
"""
find_pfield(bc::Base.Broadcast.Broadcasted) = find_pfield(bc.args)
find_pfield(args::Tuple) = find_pfield(args[1],Base.tail(args))
find_pfield(x) = x
find_pfield(::Tuple{}) = nothing
find_pfield(pfield::ParticleField,rest)= pfield
find_pfield(::Any,rest) = find_pfield(rest)
find_pfield(bc::Base.Broadcast.Broadcasted,rest) = find_pfield(bc.args)
"""
    'Base.eltype(pfield::ParticleField)
Returns the base type of the particle field numerical data.
"""
Base.eltype(pfield::ParticleField{R,F,V}) where {R,F,V} = R # eltype is the base floating-point type

"""
    'Base.axes(pfield::ParticleField)'
Returns the axes of the particle type.
"""
Base.axes(pfield::ParticleField) = (Base.OneTo(pfield.np),Base.OneTo(9)) # This defines the array size for a ParticleField

# Extra functions to get the right type returned from solving the ODE:

"""
    'Base.similar(pfield::ParticleField{R,F,V})'
Returns a ParticleField similar to the input. The parametric types as well as simulation parameters will be the same. The new field
will be initialized with the same maxparticles as the original. The same number of particles will be active in the new ParticleField, but
the numerical data will be initialized to zeros.
"""
function Base.similar(pfield::ParticleField{R,F,V}) where {R,F,V}
    pfield_out = ParticleField(pfield.np;formulation = pfield.formulation,viscous = pfield.viscous,R = R)
    # copies over all the settings
    pfield_out.maxparticles = pfield.maxparticles
    for i=1:pfield.np
        p = pfield.particles[i]
        add_particle(pfield_out,R[0,0,0],R[0,0,0],zero(R);vol = zero(R),circulation = 1e-12*one(R),index = p.index)
    end
    copy_settings!(pfield_out,pfield)
    return pfield_out
end
function Base.similar(pfield::ParticleField{R,F,V},T2::Type) where {R,F,V}
    pfield_out = ParticleField(pfield.np;formulation = pfield.formulation,viscous = pfield.viscous,R = T2)
    # copies over all the settings
    for i=1:pfield.np
        p = pfield.particles[i]
        add_particle(pfield_out,R[0,0,0],R[0,0,0],zero(R);vol = zero(R),circulation = 1e-12*one(R),index = p.index)
    end
    copy_settings!(pfield_out,pfield)
    return pfield_out
end

"""
    Base.zero(pfield::ParticleField{R,F,V})
Returns a "zero" ParticleField. The parametric types as well as simulation parameters will be the same as the inpit. The new field
will be initialized with the same maxparticles as the original. The same number of particles will be active in the new ParticleField, but
the numerical data will be initialized to zeros. See Base.similar(pfield::ParticleField{R,F,V}).
"""
function Base.zero(pfield::ParticleField{R,F,V}) where {R,F,V}
    pfield_out = ParticleField(pfield.np;formulation = pfield.formulation,viscous = pfield.viscous,R = R)
    for i=1:pfield.np
        p = pfield.particles[i]
        add_particle(pfield_out,R[0,0,0],R[0,0,0],zero(R);vol = zero(R),circulation = 1e-12*one(R),index = p.index)
    end
    copy_settings!(pfield_out,pfield)
    return pfield_out
end

"""
    Base.copy(pfield::ParticleField{R,F,V})
Copies the input ParticleField. In the process, this initializes a new ParticleField to match the input. See
Base.similar(pfield::ParticleField{R,F,V}) and Base.zero(pfield::ParticleField{R,F,V}).
"""
function Base.copy(pfield::ParticleField{R,F,V}) where {R,F,V}
    pfield_out = ParticleField(pfield.maxparticles;formulation = pfield.formulation,viscous = pfield.viscous,R = R)
    for i=1:pfield.np
        p = pfield.particles[i]
        ###add_particle(pfield_out,copy(p.X),copy(p.Gamma),p.sigma[1]; vol = p.vol[1],circulation = p.circulation[1],index = p.index)
        add_particle(pfield_out,p.X,p.Gamma,p.sigma[1]; vol = p.vol[1],circulation = p.circulation[1],index = p.index)
    end
    copy_settings!(pfield_out,pfield)
    return pfield_out
end

"""
    'copy_settings!(dest::ParticleField{R,F,V},source::ParticleField{R,F,V})'
Copies the settings from one ParticleField to another. This is called in copy(), similar(), and zero(), but
is defined as a second function for convenience.
"""

function copy_settings!(dest::ParticleField{R,F,V},source::ParticleField{R,F,V}) where {R,F,V}

    dest.maxparticles = source.maxparticles
    dest.kernel = source.kernel
    dest.UJ = source.UJ
    dest.Uinf = source.Uinf
    dest.sgsmodel = source.sgsmodel
    dest.sgsscaling = source.sgsscaling
    dest.integration = source.integration
    dest.transposed = source.transposed
    dest.relaxation = source.relaxation
    dest.relax = source.relax
    dest.rlxf = source.rlxf
    dest.fmm = source.fmm
    return nothing

end

##### END OF PARTICLE FIELD#####################################################
