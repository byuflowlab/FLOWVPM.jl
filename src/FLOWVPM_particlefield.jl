#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

const nfields = 43
const useGPU_default = 0

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
mutable struct FMM{TEPS}
  # Optional user inputs
  p::Int64                        # Multipole expansion order
  ncrit::Int64                    # Max number of particles per leaf
  theta::FLOAT_TYPE                  # Neighborhood criterion
  nonzero_sigma::Bool
  ε_tol::TEPS
end

function FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true, ε_tol=nothing)
    return FMM{typeof(ε_tol)}(p, ncrit, theta, nonzero_sigma, ε_tol)
end

################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleField{R, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::Matrix{R}                        # Array of particles
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme

    # Internal properties
    np::Int                                     # Number of particles in the field
    nt::Int                                     # Current time step number
    t::Real                                     # Current time

    # Solver setting
    kernel::Tkernel                             # Vortex particle kernel
    UJ::TUJ                                     # Particle-to-particle calculation

    # Optional inputs
    Uinf::TUinf # Uniform freestream function Uinf(t)
    SFS::S                                      # Subfilter-scale contributions scheme
    integration::Tintegration                   # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    relaxation::TRelaxation                              # Relaxation scheme
    fmm::FMM{TEPS}                                    # Fast-multipole settings
    useGPU::Int                                 # run on GPU if >0, CPU if 0

    # Internal memory for computation
    M::Array{R, 1} # uses particle type since this memory is used for particle-related computations.

    # switches for dispatch in the FMM
    toggle_rbf::Bool                            # if true, the FMM computes the vorticity field rather than velocity field
    toggle_sfs::Bool                            # if true, the FMM computes the stretching term for the SFS model
end

"""
    `ParticleField(maxparticles::Int, R=FLOAT_TYPE; <keyword arguments>)`

Create a new particle field with `maxparticles` particles. The particle field
is created with the default values for the other parameters.

# Arguments
- `maxparticles::Int`           : Maximum number of particles in the field.
- `R=FLOAT_TYPE`                : Type of the particle field. Default is `FLOAT_TYPE`.
- `formulation`                 : VPM formulation. Default is `rVPM`.
- `viscous::ViscousScheme`      : Viscous scheme. Default is `Inviscid()`. With `rVPM` formulation,
                                    a viscous scheme is not required for numerical stability.
- `np::Int`                     : Number of particles currently in the field. Default is 0. (user should not modify)
- `nt::Int`                     : Current time step number. Default is 0. (user should not modify)
- `t::Real`                     : Current time. Default is 0.
- `transposed::Bool`            : If true, the transposed scheme is recommended for stability.
                                Default is true. (user should not modify)
- `fmm::FMM{TEPS}`              : Fast-multipole settings. Default is `FMM()`.
- `M::Array{R, 1}`              : Memory for computations. Default is `zeros(R, 4)`. (user should not modify)
- `toggle_rbf::Bool`            : If true, the FMM computes the vorticity field rather than velocity field.
                                    This is used as an internal switch for the FMM.
                                    Default is false. (user should not modify)
- `toggle_sfs::Bool`            : If true, the FMM computes the stretching term for the SFS model.
                                    This is used as an internal switch for the FMM.
                                    Default is false. (user should not modify)
- `SFS::S`                      : Subfilter-scale contributions scheme. Default is `noSFS`.
- `kernel::Tkernel`             : Vortex particle kernel. Default is `gaussianerf`.
- `UJ::TUJ`                     : Particle-to-particle calculation. Default is `UJ_fmm`.
- `Uinf::TUinf`                 : Uniform freestream function Uinf(t). Default is no freestream.
- `relaxation::TR`              : Relaxation scheme. Default is `pedrizzetti`.
- `integration::Tintegration`   : Time integration scheme. Default is `rungekutta3`. The only other
                                    option is `euler`.
- `useGPU::Int`                 : Run on GPU if >0, CPU if 0. Default is 0. (Experimental and does not 
                                    accelerate SFS calculations)
"""
function ParticleField(maxparticles::Int, R=FLOAT_TYPE;
        formulation::F=formulation_default,
        viscous::V=Inviscid(),
        np=0, nt=0, t=zero(R),
        transposed=true,
        fmm::FMM{TEPS}=FMM(),
        M=zeros(R, 4),
        toggle_rbf=false, toggle_sfs=false,
        SFS::S=SFS_default, kernel::Tkernel=kernel_default,
        UJ::TUJ=UJ_fmm, Uinf::TUinf=Uinf_default,
        relaxation::TR=Relaxation(relax_pedrizzetti, 1, 0.3), # default relaxation has no type input, which is a problem for AD.
        integration::Tintegration=rungekutta3,
        useGPU=useGPU_default
    ) where {F, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel<:Kernel, TUJ, Tintegration, TR, TEPS}

    # create particle field
    # particles = [zero(Particle{R}) for _ in 1:maxparticles]
    particles = zeros(R, nfields, maxparticles)

    # Set index of each particle
    # for (i, P) in enumerate(particles)
    #     P.index[1] = i
    # end
    # Generate and return ParticleField
    return ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU, TEPS}(maxparticles, particles,
                                            formulation, viscous, np, nt, t,
                                            kernel, UJ, Uinf, SFS, integration,
                                            transposed, relaxation, fmm, useGPU,
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
  `add_particle(pfield::ParticleField, X, Gamma, sigma; <keyword arguments>)`

Add a particle to the field.

# Arguments
- `pfield::ParticleField`   : Particle field to add the particle to.
- `X`                      : Position of the particle.
- `Gamma`                  : Strength of the particle.
- `sigma`                 : Smoothing radius of the particle.
- `vol`                   : Volume of the particle. Default is 0.
- `circulation`           : Circulation of the particle. Default is 1.
- `C`                     : SFS parameter of the particle. Default is 0.
- `static`                : If true, the particle is static. Default is false.
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

const X_INDEX = 1:3
const GAMMA_INDEX = 4:6
const SIGMA_INDEX = 7
const VOL_INDEX = 8
const CIRCULATION_INDEX = 9
const U_INDEX = 10:12
const VORTICITY_INDEX = 13:15
const J_INDEX = 16:24
const PSE_INDEX = 25:27
const M_INDEX = 28:36
const C_INDEX = 37:39
const SFS_INDEX = 40:42
const STATIC_INDEX = 43

"Get functions for particles"
# This is (and should be) the only place that explicitly
# maps the indices of each particle's fields
get_X(P) = view(P, X_INDEX)
get_Gamma(P) = view(P, GAMMA_INDEX)
get_sigma(P) = view(P, SIGMA_INDEX)
get_vol(P) = view(P, VOL_INDEX)
get_circulation(P) = view(P, CIRCULATION_INDEX)
get_U(P) = view(P, U_INDEX)
get_vorticity(P) = view(P, VORTICITY_INDEX)
get_J(P) = view(P, J_INDEX)
get_PSE(P) = view(P, PSE_INDEX)
get_M(P) = view(P, M_INDEX)
get_C(P) = view(P, C_INDEX)
get_SFS(P) = view(P, SFS_INDEX)
get_static(P) = view(P, STATIC_INDEX)

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
get_X(pfield::ParticleField, i::Int) = view(pfield.particles, X_INDEX, i)
get_Gamma(pfield::ParticleField, i::Int) = view(pfield.particles, GAMMA_INDEX, i)
get_sigma(pfield::ParticleField, i::Int) = view(pfield.particles, SIGMA_INDEX, i)
get_vol(pfield::ParticleField, i::Int) = view(pfield.particles, VOL_INDEX, i)
get_circulation(pfield::ParticleField, i::Int) = view(pfield.particles, CIRCULATION_INDEX, i)
get_U(pfield::ParticleField, i::Int) = view(pfield.particles, U_INDEX, i)
get_vorticity(pfield::ParticleField, i::Int) = view(pfield.particles, VORTICITY_INDEX, i)
get_J(pfield::ParticleField, i::Int) = view(pfield.particles, J_INDEX, i)
get_PSE(pfield::ParticleField, i::Int) = view(pfield.particles, PSE_INDEX, i)
get_W(pfield::ParticleField, i::Int) = get_W(get_particle(pfield, i))
get_M(pfield::ParticleField, i::Int) = view(pfield.particles, M_INDEX, i)
get_C(pfield::ParticleField, i::Int) = view(pfield.particles, C_INDEX, i)
get_SFS(pfield::ParticleField, i::Int) = view(pfield.particles, SFS_INDEX, i)
get_static(pfield::ParticleField, i::Int) = Bool(pfield.particles[43, i])

"Set functions for particles"
function set_X(P, val) P[X_INDEX] .= val end
function set_Gamma(P, val) P[GAMMA_INDEX] .= val end
function set_sigma(P, val) P[SIGMA_INDEX] = val end
function set_vol(P, val) P[VOL_INDEX] = val end
function set_circulation(P, val) P[CIRCULATION_INDEX] = val end
function set_U(P, val) P[U_INDEX] .= val end
function set_vorticity(P, val) P[VORTICITY_INDEX] .= val end
function set_J(P, val) P[J_INDEX] .= val end
function set_M(P, val) P[M_INDEX] .= val end
function set_C(P, val) P[C_INDEX] .= val end
function set_static(P, val) P[STATIC_INDEX] = val end
function set_PSE(P, val) P[PSE_INDEX] .= val end
function set_SFS(P, val) P[SFS_INDEX] .= val end


"Set functions for particles in ParticleField"
function set_X(pfield::ParticleField, i::Int, val) pfield.particles[X_INDEX, i] .= val end
function set_Gamma(pfield::ParticleField, i::Int, val) pfield.particles[GAMMA_INDEX, i] .= val end
function set_sigma(pfield::ParticleField, i::Int, val) pfield.particles[SIGMA_INDEX, i] = val end
function set_vol(pfield::ParticleField, i::Int, val) pfield.particles[VOL_INDEX, i] = val end
function set_circulation(pfield::ParticleField, i::Int, val) pfield.particles[CIRCULATION_INDEX, i] = val end
function set_U(pfield::ParticleField, i::Int, val) pfield.particles[U_INDEX, i] .= val end
function set_vorticity(pfield::ParticleField, i::Int, val) pfield.particles[VORTICITY_INDEX, i] .= val end
function set_J(pfield::ParticleField, i::Int, val) pfield.particles[J_INDEX, i] .= val end
function set_M(pfield::ParticleField, i::Int, val) pfield.particles[M_INDEX, i] .= val end
function set_C(pfield::ParticleField, i::Int, val) pfield.particles[C_INDEX, i] .= val end
function set_static(pfield::ParticleField, i::Int, val) pfield.particles[STATIC_INDEX, i] = val end
function set_PSE(pfield::ParticleField, i::Int, val) pfield.particles[PSE_INDEX, i] .= val end
function set_SFS(pfield::ParticleField, i::Int, val) pfield.particles[SFS_INDEX, i] .= val end

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

Steps the particle field in time by a step `dt`. Modifies the pfield in place.

# Arguments
- `pfield::ParticleField`   : Particle field to step.
- `dt::Real`                : Time step to step the field.
- `relax::Bool`             : If true, the relaxation scheme is applied to the
                                particles. Default is false.

# Returns
- The time step number of the particle field.
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
    for i in 1:pfield.np
        _reset_particle(get_particle(pfield, i))
    end
end

function _reset_particle(particle)
    zeroVal = zero(eltype(particle))
    set_U(particle, zeroVal)
    set_vorticity(particle, zeroVal)
    set_J(particle, zeroVal)
    set_PSE(particle, zeroVal)
end

_reset_particle(pfield::ParticleField, i::Int) = _reset_particle(get_particle(pfield, i))

function _reset_particles_sfs(pfield::ParticleField)
    for i in 1:pfield.np
        _reset_particle_sfs(get_particle(pfield, i))
    end
end

function _reset_particles_sfs(pfield::ParticleField, i::Int)
    _reset_particle(get_particle(pfield, i))
end

function _reset_particle_sfs(particle)
    set_SFS(particle, zero(eltype(particle)))
end

##### END OF PARTICLE FIELD#####################################################
