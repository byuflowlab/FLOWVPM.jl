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

  FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true) = new(p, ncrit, theta, nonzero_sigma)
end

################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleField{R, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TRelaxation}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Matrix{R}                        # Array of particles
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::R                                        # Current time # should not have its type tied to other floating-point types

    # Solver setting
    kernel::Tkernel                             # Vortex particle kernel
    UJ::TUJ                                     # Particle-to-particle calculation

    # Optional inputs
    Uinf::TUinf # Uniform freestream function Uinf(t)
    SFS::S                                      # Subfilter-scale contributions scheme
    integration::Tintegration                   # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    relaxation::TRelaxation                              # Relaxation scheme
    fmm::FMM                                    # Fast-multipole settings

    # Internal memory for computation
    M::Array{R, 1} # uses particle type since this memory is used for particle-related computations.

    # switches for dispatch in the FMM
    toggle_rbf::Bool                            # if true, the FMM computes the vorticity field rather than velocity field
    toggle_sfs::Bool                            # if true, the FMM computes the stretching term for the SFS model
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
        UJ::TUJ=UJ_fmm, Uinf::TUinf=Uinf_default,
        relaxation::TR=Relaxation(relax_pedrizzetti, 1, 0.3), # default relaxation has no type input, which is a problem for AD.
        integration::Tintegration=rungekutta3,
    ) where {F, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel<:Kernel, TUJ, Tintegration, TR}

    # create particle field
    # particles = [zero(Particle{R}) for _ in 1:maxparticles]
    particles = zeros(R, nfields, maxparticles)

    # Set index of each particle
    # for (i, P) in enumerate(particles)
    #     P.index[1] = i
    # end
    # Generate and return ParticleField
    return ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR}(maxparticles, particles,
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
isLES(pfield::ParticleField) = isSFSenabled(pfield.SFS)

##### FUNCTIONS ################################################################
"""
  `add_particle(pfield::ParticleField, X, Gamma, sigma; vol=0)`

Add a particle to the field.
"""
function add_particle(pfield::ParticleField, X, Gamma, sigma;
                                           vol=0, circulation=1,
                                           C=0, static=false)
    # ERROR CASES
    if get_np(pfield)==pfield.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(pfield.maxparticles)"*
                                                            " has been reached")
    # elseif circulation<=0
    #     error("Got invalid circulation less or equal to zero! ($(circulation))")
    end

    # Fetch the index of the next empty particle in the field
    i_next = get_np(pfield)+1

    # Add particle to the field
    pfield.np += 1

    # Populate the empty particle
    set_X(pfield, i_next, X)
    set_Gamma(pfield, i_next, Gamma)
    set_sigma(pfield, i_next, sigma)
    set_vol(pfield, i_next, vol)
    set_circulation(pfield, i_next, circulation)
    set_C(pfield, i_next, C)
    set_static(pfield, i_next, Float64(static))

    return nothing
end

"""
  `add_particle(pfield::ParticleField, P)`

Add a copy of Particle `P` to the field.
"""
function add_particle(pfield::ParticleField, P)
    return add_particle(pfield, get_X(P), get_Gamma(P), get_sigma(P)[];
                        vol=get_vol(P)[], circulation=get_circulation(P)[],
                        C=get_C(P), static=is_static(P))
end

"""
    `get_np(pfield::ParticleField)`

    Returns current number of particles in the field.
"""
get_np(pfield::ParticleField) = pfield.np

"""
    `get_particle(pfield::ParticleField, i)`

    Returns the i-th particle in the field.
"""
function get_particle(pfield::ParticleField, i::Int; emptyparticle=false)
    if i<=0
        error("Requested invalid particle index $i")
    elseif !emptyparticle && i>get_np(pfield)
        error("Requested particle $i, but there is only $(get_np(pfield))"*
                                                    " particles in the field.")
    elseif emptyparticle && i!=(get_np(pfield)+1)
        error("Requested empty particle $i, but next empty particle is"*
                                                          " $(get_np(pfield)+1)")
    end

    return view(pfield.particles, :, i)
end

"Alias for `get_particleiterator`"
iterator(args...; optargs...) = get_particleiterator(args...; optargs...)

"Alias for `get_particleiterator`"
iterate(args...; optargs...) = get_particleiterator(args...; optargs...)

"Get functions for particles"
# This is (and should be) the only place that explicitly
# maps the indices of each particle's fields
get_X(P) = view(P, 1:3)
get_Gamma(P) = view(P, 4:6)
get_sigma(P) = view(P, 7)
get_vol(P) = view(P, 8)
get_circulation(P) = view(P, 9)
get_U(P) = view(P, 10:12)
get_vorticity(P) = view(P, 13:15)
get_J(P) = view(P, 16:24)
get_PSE(P) = view(P, 25:27)
get_M(P) = view(P, 28:36)
get_C(P) = view(P, 37:39)
get_SFS(P) = view(P, 40:42)
get_static(P) = view(P, 43)

is_static(P) = Bool(P[43])

# This extra function computes the vorticity using the cross-product
get_W(P) = (get_W1(P), get_W2(P), get_W3(P))

get_W1(P) = get_J(P)[6]-get_J(P)[8]
get_W2(P) = get_J(P)[7]-get_J(P)[3]
get_W3(P) = get_J(P)[2]-get_J(P)[4]

get_SFS1(P) = get_SFS(P)[1]
get_SFS2(P) = get_SFS(P)[2]
get_SFS3(P) = get_SFS(P)[3]

"Get functions for particles in ParticleField"
get_X(pfield::ParticleField, i::Int) = get_X(get_particle(pfield, i))
get_Gamma(pfield::ParticleField, i::Int) = get_Gamma(get_particle(pfield, i))
get_sigma(pfield::ParticleField, i::Int) = get_sigma(get_particle(pfield, i))
get_vol(pfield::ParticleField, i::Int) = get_vol(get_particle(pfield, i))
get_circulation(pfield::ParticleField, i::Int) = get_circulation(get_particle(pfield, i))
get_U(pfield::ParticleField, i::Int) = get_U(get_particle(pfield, i))
get_vorticity(pfield::ParticleField, i::Int) = get_vorticity(get_particle(pfield, i))
get_J(pfield::ParticleField, i::Int) = get_J(get_particle(pfield, i))
get_PSE(pfield::ParticleField, i::Int) = get_PSE(get_particle(pfield, i))
get_W(pfield::ParticleField, i::Int) = get_W(get_particle(pfield, i))
get_M(pfield::ParticleField, i::Int) = get_M(get_particle(pfield, i))
get_C(pfield::ParticleField, i::Int) = get_C(get_particle(pfield, i))
get_static(pfield::ParticleField, i::Int) = get_static(get_particle(pfield, i))

is_static(pfield::ParticleField, i::Int) = is_static(get_particle(pfield, i))

"Set functions for particles"
set_X(P, val) = get_X(P) .= val
set_Gamma(P, val) = get_Gamma(P) .= val
set_sigma(P, val) = get_sigma(P) .= val
set_vol(P, val) = get_vol(P) .= val
set_circulation(P, val) = get_circulation(P) .= val
set_U(P, val) = get_U(P) .= val
set_vorticity(P, val) = get_vorticity(P) .= val
set_J(P, val) = get_J(P) .= val
set_M(P, val) = get_M(P) .= val
set_C(P, val) = get_C(P) .= val
set_static(P, val) = get_static(P) .= val
set_PSE(P, val) = get_PSE(P) .= val
set_SFS(P, val) = get_SFS(P) .= val

"Set functions for particles in ParticleField"
set_X(pfield::ParticleField, i::Int, val) = set_X(get_particle(pfield, i), val)
set_Gamma(pfield::ParticleField, i::Int, val) = set_Gamma(get_particle(pfield, i), val)
set_sigma(pfield::ParticleField, i::Int, val) = set_sigma(get_particle(pfield, i), val)
set_vol(pfield::ParticleField, i::Int, val) = set_vol(get_particle(pfield, i), val)
set_circulation(pfield::ParticleField, i::Int, val) = set_circulation(get_particle(pfield, i), val)
set_U(pfield::ParticleField, i::Int, val) = set_U(get_particle(pfield, i), val)
set_vorticity(pfield::ParticleField, i::Int, val) = set_vorticity(get_particle(pfield, i), val)
set_J(pfield::ParticleField, i::Int, val) = set_J(get_particle(pfield, i), val)
set_M(pfield::ParticleField, i::Int, val) = set_M(get_particle(pfield, i), val)
set_C(pfield::ParticleField, i::Int, val) = set_C(get_particle(pfield, i), val)
set_static(pfield::ParticleField, i::Int, val) = set_static(get_particle(pfield, i), val)
set_PSE(pfield::ParticleField, i::Int, val) = set_PSE(get_particle(pfield, i), val)
set_SFS(pfield::ParticleField, i::Int, val) = set_SFS(get_particle(pfield, i), val)

"""
    `isinviscid(pfield::ParticleField)`

Returns true if particle field is inviscid.
"""
isinviscid(pfield::ParticleField) = isinviscid(pfield.viscous)


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

function _get_particleiterator(pfield::ParticleField; start_i::Int=1, end_i::Int=-1, reverse=false)
    if end_i > get_np(pfield)
        error("Requested end_i=$(end_i), but there is only $(get_np(pfield))"*
              " particles in the field.")
    end

    last_i = end_i==-1 ? get_np(pfield) : end_i

    if reverse
        i_particles = last_i : -1 : start_i
    else
        i_particles = start_i : last_i
    end

    return (view(pfield.particles, :, i) for i in i_particles)
end

"""
  `remove_particle(pfield::ParticleField, i)`

Remove the i-th particle in the field. This is done by moving the last particle
that entered the field into the memory slot of the target particle. To remove
particles sequentally, you will need to go from the last particle back to the
first one (see documentation of `get_particleiterator` for an example).
"""
function remove_particle(pfield::ParticleField, i::Int)
    if i<=0
        error("Requested removal of invalid particle index $i")
    elseif i>get_np(pfield)
        error("Requested removal of particle $i, but there is only"*
              " $(get_np(pfield)) particles in the field.")
    end

    if i != get_np(pfield)
        # Overwrite target particle with last particle in the field
        get_particle(pfield, i) .= get_particle(pfield, get_np(pfield))
    end

    # Remove last particle in the field
    _reset_particle(pfield, get_np(pfield))
    pfield.np -= 1

    return nothing
end

"""
  `nextstep(pfield::ParticleField, dt; relax=false)`

Steps the particle field in time by a step `dt`.
"""
function nextstep(pfield::ParticleField, dt::Real; optargs...)

    # Step in time
    if get_np(pfield)!=0
        pfield.integration(pfield, dt; optargs...)
    end

    # Updates time
    pfield.t += dt
    pfield.nt += 1
end


##### INTERNAL FUNCTIONS #######################################################
function _reset_particles(pfield::ParticleField)
    for particle in iterate(pfield)
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

function _reset_particle(pfield::ParticleField, i::Int)
    _reset_particle(get_particle(pfield, i))
end

function _reset_particles_sfs(pfield::ParticleField)
    for particle in iterate(pfield)
        _reset_particle_sfs(particle)
    end
end

function _reset_particles_sfs(pfield::ParticleField, i::Int)
    _reset_particle(get_particle(pfield, i))
end

function _reset_particle_sfs(particle)
    set_SFS(particle, zero(eltype(particle)))
end

##### END OF PARTICLE FIELD#####################################################
