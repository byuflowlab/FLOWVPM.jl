#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

################################################################################
# FMM STRUCT
################################################################################
"""
    `FMM(; p::Int=4, ncrit::Int=50, theta::Real=0.4, phi::Real=0.3)`

Parameters for FMM solver.

# Arguments
* `p`       : Order of multipole expansion (number of terms).
* `ncrit`   : Maximum number of particles per leaf.
* `theta`   : Neighborhood criterion. This criterion defines the distance
                where the far field starts. The criterion is that if θ*r < R1+R2
                the interaction between two cells is resolved through P2P, where
                r is the distance between cell centers, and R1 and R2 are each
                cell radius. This means that at θ=1, P2P is done only on cells
                that have overlap; at θ=0.5, P2P is done on cells that their
                distance is less than double R1+R2; at θ=0.25, P2P is done on
                cells that their distance is less than four times R1+R2; at
                θ=0, P2P is done on cells all cells.
* `phi`     : Regularizing neighborhood criterion. This criterion avoid
                approximating interactions with the singular-FMM between
                regularized particles that are sufficiently close to each other
                across cell boundaries. Used together with the θ-criterion, P2P
                is performed between two cells if φ < σ/dx, where σ is the
                average smoothing radius in between all particles in both cells,
                and dx is the distance between cell boundaries
                ( dx = r-(R1+R2) ). This means that at φ = 1, P2P is done on
                cells with boundaries closer than the average smoothing radius;
                at φ = 0.5, P2P is done on cells closer than two times the
                smoothing radius; at φ = 0.25, P2P is done on cells closer than
                four times the smoothing radius.
"""
mutable struct FMM
  # Optional user inputs
  p::Int64                        # Multipole expansion order
  ncrit::Int64                    # Max number of particles per leaf
  theta::FLOAT_TYPE                  # Neighborhood criterion
  nonzero_sigma::Bool

  FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=false) = new(p, ncrit, theta, nonzero_sigma)
end

################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleField{R<:Real, F<:Formulation, V<:ViscousScheme, S<:SubFilterScale, Tkernel, TUJ, Tintegration}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Array{Particle{R}, 1}            # Array of particles
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::R                                        # Current time

    # Solver setting
    kernel::Tkernel                              # Vortex particle kernel
    UJ::TUJ                                # Particle-to-particle calculation

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    SFS::S                                      # Subfilter-scale contributions scheme
    integration::Tintegration                       # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    relaxation::Relaxation{R}                   # Relaxation scheme
    fmm::FMM                                    # Fast-multipole settings

    # Internal memory for computation
    M::Array{R, 1}

    # switches for dispatch in the FMM
    toggle_rbf::Bool                            # if true, the FMM computes the vorticity field rather than velocity field
    toggle_sfs::Bool                            # if true, the FMM computes the stretching term for the SFS model

    # ParticleField{R, F, V, S, Tkernel, TUJ, Tintegration}(
    #                             maxparticles,
    #                             particles, formulation, viscous;
    #                             np=0, nt=0, t=R(0.0),
    #                             kernel::Tkernel=kernel_default,
    #                             UJ::TUJ=UJ_fmm,
    #                             Uinf::Function=Uinf_default,
    #                             SFS=SFS_default,
    #                             integration::Tintegration=rungekutta3,
    #                             transposed=true,
    #                             relaxation=relaxation_default,
    #                             fmm=FMM(),
    #                             M=zeros(R, 4),
    #                             toggle_rbf=false, toggle_sfs=false
    #                      ) where {R, F, V, S, Tkernel, TUJ, Tintegration} = new(
    #                             maxparticles,
    #                             particles, formulation, viscous,
    #                             np, nt, t,
    #                             kernel,
    #                             UJ,
    #                             Uinf,
    #                             SFS,
    #                             integration,
    #                             transposed,
    #                             relaxation,
    #                             fmm,
    #                             M,
    #                             toggle_rbf, toggle_sfs
    #                       )
end

function ParticleField(maxparticles::Int, R=FLOAT_TYPE;
                                    formulation::F=formulation_default,
                                    viscous::V=Inviscid(), 
                                    np=0, nt=0, t=R(0.0),
                                    transposed=true,
                                    fmm=FMM(),
                                    M=zeros(R, 4),
                                    toggle_rbf=false, toggle_sfs=false,    
                                    SFS::S=SFS_default, kernel::Tkernel=kernel_default,
                                    UJ::TUJ=UJ_fmm, Uinf::Function=Uinf_default,
                                    relaxation=relaxation_default, 
                                    integration::Tintegration=rungekutta3,
                            ) where {F, V<:ViscousScheme, S<:SubFilterScale, Tkernel<:Kernel, TUJ, Tintegration}

    # create particle field
    particles = [zero(Particle{R}) for _ in 1:maxparticles]

    # Set index of each particle
    for (i, P) in enumerate(particles)
        P.index[1] = i
    end
    # Generate and return ParticleField
    return ParticleField{R, F, V, S, Tkernel, TUJ, Tintegration}(maxparticles, particles,
                                            formulation, viscous, np, nt, t,
                                            kernel, UJ, Uinf, SFS, integration,
                                            transposed, relaxation, fmm,
                                            M, toggle_rbf, toggle_sfs)
end

"""
    `isLES(pfield::ParticleField)`

    Returns true if the particle field solver implements a subfilter-scale model
of turbulence for large eddy simulation (LES).
"""
isLES(self::ParticleField) = isSFSenabled(self.SFS)

##### FUNCTIONS ################################################################
"""
  `add_particle(self::ParticleField, X, Gamma, sigma; vol=0, index=np)`

Add a particle to the field.
"""
function add_particle(self::ParticleField, X, Gamma, sigma;
                                           vol=0, circulation=1,
                                           C=0, static=false, index=-1)
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    # elseif circulation<=0
    #     error("Got invalid circulation less or equal to zero! ($(circulation))")
    end

    # Fetch next empty particle in the field
    P = get_particle(self, get_np(self)+1; emptyparticle=true)

    # Populate the empty particle
    P.X .= X
    P.Gamma .= Gamma
    P.sigma .= sigma
    P.vol .= vol
    P.circulation .= abs.(circulation)
    P.C .= C
    P.static .= static
    P.index .= index==-1 ? get_np(self) : index

    # Add particle to the field
    self.np += 1

    return nothing
end

"""
  `add_particle(self::ParticleField, P::Particle)`

Add a copy of Particle `P` to the field.
"""
function add_particle(self::ParticleField, P::Particle)
    return add_particle(self, P.X, P.Gamma, P.sigma;
                            vol=P.vol, circulation=P.circulation,
                            C=P.C, static=P.static)
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
function get_particleiterator(args...; include_static=false, optargs...)
    if include_static
        return _get_particleiterator(args...; optargs...)
    else
        return (P for P in _get_particleiterator(args...; optargs...) if !P.static[1])
    end
end

function _get_particleiterator(self::ParticleField{R, F, V}; start_i::Int=1,
                              end_i::Int=-1, reverse=false) where {R, F, V}
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
        # fmm.overwriteBody(self.bodies, i-1, get_np(self)-1)
        Ptarg = get_particle(self, i)
        
        Ptarg.X .= Plast.X
        Ptarg.Gamma .= Plast.Gamma
        Ptarg.sigma .= Plast.sigma
        Ptarg.vol .= Plast.vol
        Ptarg.circulation .= Plast.circulation
        Ptarg.static .= Plast.static

        Ptarg.U .= Plast.U
        Ptarg.W .= Plast.W
        Ptarg.J .= Plast.J
        Ptarg.PSE .= Plast.PSE

        Ptarg.M .= Plast.M
        Ptarg.C .= Plast.C
        Ptarg.S .= Plast.S

        Ptarg.index .= Plast.index
    end

    # Remove last particle in the field
    _reset_particle(Plast)
    _reset_particle_sfs(Plast)
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
    for P in iterator(self; include_static=true)
        _reset_particle(P, tzero)
    end
end

function _reset_particle(P::Particle{T}, tzero::T) where {T}
    P.U[1] = tzero
    P.U[2] = tzero
    P.U[3] = tzero

    P.W .= tzero

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

function _reset_particles_sfs(self::ParticleField{R, F, V}) where {R, F, V}
    tzero = zero(R)
    for P in iterator(self; include_static=true)
        _reset_particle_sfs(P, tzero)
    end
end

function _reset_particle_sfs(P::Particle{T}, tzero::T) where {T}
    getproperty(P, _SFS)::MVector{3,T} .= tzero
    # P.C .= tzero
end
_reset_particle_sfs(P::Particle{T}) where {T} = _reset_particle_sfs(P, zero(T))
##### END OF PARTICLE FIELD#####################################################
