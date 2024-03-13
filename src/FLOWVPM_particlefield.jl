#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

const nfields = 43
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
    particles::Matrix{R}                        # Array of particles
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::R                                        # Current time

    # Solver setting
    kernel::Tkernel                             # Vortex particle kernel
    UJ::TUJ                                     # Particle-to-particle calculation

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    SFS::S                                      # Subfilter-scale contributions scheme
    integration::Tintegration                   # Time integration scheme
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
    # particles = [zero(Particle{R}) for _ in 1:maxparticles]
    particles = zeros(R, nfields, maxparticles)

    # Set index of each particle
    # for (i, P) in enumerate(particles)
    #     P.index[1] = i
    # end
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
  `add_particle(self::ParticleField, X, Gamma, sigma; vol=0)`

Add a particle to the field.
"""
function add_particle(self::ParticleField, X, Gamma, sigma;
                                           vol=0, circulation=1,
                                           C=0, static=false)
    # ERROR CASES
    if get_np(self)==self.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(self.maxparticles)"*
                                                            " has been reached")
    # elseif circulation<=0
    #     error("Got invalid circulation less or equal to zero! ($(circulation))")
    end

    # Fetch the index of the next empty particle in the field
    i_next = get_np(self)+1

    # Populate the empty particle
    self.particles[1:3, i_next] .= X
    self.particles[4:6, i_next] .= Gamma
    self.particles[7, i_next] = sigma
    self.particles[8, i_next] = vol
    self.particles[9, i_next] = abs.(circulation)
    self.particles[37:39, i_next] .= C
    self.particles[43, i_next] = Float64(static)

    # Add particle to the field
    self.np += 1

    return nothing
end

"""
  `add_particle(self::ParticleField, P::Particle)`

Add a copy of Particle `P` to the field.
"""
function add_particle(self::ParticleField, particle)
    return add_particle(self, particle[1:3], particle[4:6], particle[7];
                        vol=particle[8], circulation=particle[9],
                        C=particle[37:39], static=is_static(particle))
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

    return view(self.particles, :, i)
end

"Alias for `get_particleiterator`"
iterator(args...; optargs...) = get_particleiterator(args...; optargs...)

"Alias for `get_particleiterator`"
iterate(args...; optargs...) = get_particleiterator(args...; optargs...)

"Get functions for particles"
# This is (and should be) the only place that explicitly
# maps the indices of each particle's fields
get_X(particle) = view(particle, 1:3)
get_Gamma(particle) = view(particle, 4:6)
get_sigma(particle) = view(particle, 7)
get_vol(particle) = view(particle, 8)
get_circulation(particle) = view(particle, 9)
get_U(particle) = view(particle, 10:12)
get_vorticity(particle) = view(particle, 13:15)
get_J(particle) = view(particle, 16:24)
get_PSE(particle) = view(particle, 25:27)
get_M(particle) = view(particle, 28:36)
get_C(particle) = view(particle, 37:39)
get_SFS(particle) = view(particle, 40:42)
get_static(particle) = view(particle, 43)

is_static(particle) = Bool(particle[43])

# This extra function computes the vorticity using the cross-product
get_W(particle) = (get_W1(particle), get_W2(particle), get_W3(particle))

get_W1(particle) = particle[21]-particle[23]
get_W2(particle) = particle[22]-particle[18]
get_W3(particle) = particle[17]-particle[19]

get_SFS1(particle) = particle[40]
get_SFS2(particle) = particle[41]
get_SFS3(particle) = particle[42]

add_SFS1(particle, val) = particle[40] += val
add_SFS2(particle, val) = particle[41] += val
add_SFS3(particle, val) = particle[42] += val

"Get functions for particles in ParticleField"
get_X(self::ParticleField, i::Int) = get_X(get_particle(self, i))
get_Gamma(self::ParticleField, i::Int) = get_Gamma(get_particle(self, i))
get_sigma(self::ParticleField, i::Int) = get_sigma(get_particle(self, i))
get_vol(self::ParticleField, i::Int) = get_vol(get_particle(self, i))
get_circulation(self::ParticleField, i::Int) = get_circulation(get_particle(self, i))
get_U(self::ParticleField, i::Int) = get_U(get_particle(self, i))
get_vorticity(self::ParticleField, i::Int) = get_vorticity(get_particle(self, i))
get_J(self::ParticleField, i::Int) = get_J(get_particle(self, i))
get_PSE(self::ParticleField, i::Int) = get_PSE(get_particle(self, i))
get_W(self::ParticleField, i::Int) = get_W(get_particle(self, i))
get_M(self::ParticleField, i::Int) = get_M(get_particle(self, i))
get_C(self::ParticleField, i::Int) = get_C(get_particle(self, i))
get_static(self::ParticleField, i::Int) = get_static(get_particle(self, i))

is_static(pfield::ParticleField, i::Int) = is_static(get_particle(self, i))

"Set functions for particles"
set_X(particle, val) = get_X(particle) .= val
set_Gamma(particle, val) = get_Gamma(particle) .= val
set_sigma(particle, val) = get_sigma(particle) .= val
set_vol(particle) = set_vol(particle) .= val
set_circulation(particle) = set_circulation(particle) .= val
set_U(particle, val) = get_U(particle) .= val
set_vorticity(particle, val) = get_vorticity(particle) .= val
set_J(particle, val) = get_J(particle) .= val
set_M(particle, val) = get_M(particle) .= val
set_C(particle, val) = get_C(particle) .= val
set_static(particle, val) = get_static(particle) .= val
set_PSE(particle, val) = get_PSE(particle) .= val
set_SFS(particle, val) = get_SFS(particle) .= val

"Set functions for particles in ParticleField"
set_X(self::ParticleField, i::Int, val) = set_X(get_particle(self, i), val)
set_Gamma(self::ParticleField, i::Int, val) = set_Gamma(get_particle(self, i), val)
set_sigma(self::ParticleField, i::Int, val) = set_sigma(get_particle(self, i), val)
set_vol(self::ParticleField, i::Int, val) = set_vol(get_particle(self, i), val)
set_circulation(self::ParticleField, i::Int, val) = set_circulation(get_particle(self, i), val)
set_U(self::ParticleField, i::Int, val) = set_U(get_particle(self, i), val)
set_vorticity(self::ParticleField, i::Int, val) = set_vorticity(get_particle(self, i), val)
set_J(self::ParticleField, i::Int, val) = set_J(get_particle(self, i), val)
set_M(self::ParticleField, i::Int, val) = set_M(get_particle(self, i), val)
set_C(self::ParticleField, i::Int, val) = set_C(get_particle(self, i), val)
set_static(self::ParticleField, i::Int, val) = set_static(get_particle(self, i), val)
set_PSE(self::ParticleField, i::Int, val) = set_PSE(get_particle(self, i), val)
set_SFS(self::ParticleField, i::Int, val) = set_SFS(get_particle(self, i), val)

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
           println(P.var[1:3])
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
        return (P for P in _get_particleiterator(args...; optargs...) if !is_static(P))
    end
end

function _get_particleiterator(self::ParticleField; start_i::Int=1, end_i::Int=-1, reverse=false)
    if end_i > get_np(self)
        error("Requested end_i=$(end_i), but there is only $(get_np(self))"*
              " particles in the field.")
    end

    last_i = end_i==-1 ? get_np(self) : end_i

    if reverse
        i_particles = last_i : -1 : start_i
    else
        i_particles = start_i : last_i
    end

    return eachcol(view(self.particles, :, i_particles))
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

    if i != get_np(self)
        # Overwrite target particle with last particle in the field
        self.particles[:, i] = self.particles[:, get_np(self)]
    end

    # Remove last particle in the field
    _reset_particle(self, get_np(self))
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
function _reset_particles(self::ParticleField)
    for particle in iterate(self)
        _reset_particle(particle)
    end
end

function _reset_particle(particle)
    zeroVal = zero(eltype(particle))
    set_U(particle, zeroVal)
    set_vorticity(particle, zeroVal)
    set_J(particle, zeroVal)
    set_PSE(particle, zeroVal)
end

function _reset_particles_sfs(self::ParticleField)
    for particle in iterate(self)
        _reset_particle_sfs(particle)
    end
end

function _reset_particles_sfs(self::ParticleField, i::Int)
    _reset_particle(get_particle(self, i))
end

function _reset_particle_sfs(particle)
    set_SFS(particle, zero(eltype(particle)))
end

##### END OF PARTICLE FIELD#####################################################
