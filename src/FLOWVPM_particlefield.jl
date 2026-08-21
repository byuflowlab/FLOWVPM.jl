#=##############################################################################
# DESCRIPTION
    Particle field struct definition.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

const nfields = 46
const useGPU_default = 0

################################################################################
# FMM STRUCT
################################################################################
"""
    `FMM(; p::Int=4, ncrit::Int=10, theta::Real=0.5, shrink_recenter::Bool=true,
          relative_tolerance::Real=1e-3, absolute_tolerance::Real=1e-6,
          autotune_p::Bool=true, autotune_ncrit::Bool=true,
          autotune_reg_error::Bool=true, default_rho_over_sigma::Real=1.0,
          min_ncrit::Int=50)`

Parameters for FMM solver.

# Arguments
* `p`       : Order of multipole expansion.
* `ncrit`   : Maximum number of particles per leaf.
* `theta`   : Neighborhood criterion. This criterion defines the distance
                where the far field starts. The criterion is that if θ*r < R1+R2
                the interaction between two cells is resolved through P2P, where
                r is the distance between cell centers, and R1 and R2 are each
                cell radius. This means that at θ=1, P2P is done only on cells
                that have overlap; at θ=0.5, P2P is done on cells that their
                distance is less than double R1+R2; at θ=0.25, P2P is done on
                cells that their distance is less than four times R1+R2; at
                θ=0, P2P is done on all cells.
* `shrink_recenter` : If true, shrink and recenter multipole expansions to account for nonzero particle radius.
* `relative_tolerance` : Relative error tolerance for FMM calls.
* `absolute_tolerance` : Absolute error tolerance fallback for FMM calls in case relative tolerance becomes too small.
* `autotune_p` : If true, automatically adjust p to optimize performance.
* `autotune_ncrit` : If true, automatically adjust ncrit to optimize performance.
* `autotune_reg_error` : If true, constrain regularization error in FMM calls.
* `default_rho_over_sigma` : Default value for ρ/σ in FMM calls (unused if `autotune_reg_error` is true).
* `min_ncrit` : Lower bound for auto-tuned leaf size (`ncrit`) when `autotune_ncrit=true`.

"""
struct FMM
  p::Int64                        # Multipole expansion order
  ncrit::Int64                    # Max number of particles per leaf
  theta::FLOAT_TYPE               # Neighborhood criterion
  shrink_recenter::Bool           # If true, shrink/recenter expansions to account for nonzero radius particles
  relative_tolerance::FLOAT_TYPE  # Relative error tolerance
  absolute_tolerance::FLOAT_TYPE  # Absolute error tolerance fallback in case relative tolerance becomes too small
  autotune_p::Bool                # If true, automatically adjust p to optimize performance
  autotune_ncrit::Bool            # If true, automatically adjust ncrit to optimize performance
  autotune_reg_error::Bool        # If true, automatically calculate rho/sigma to constrain regularization error
  default_rho_over_sigma::FLOAT_TYPE # Default value for ρ/σ in FMM calls (unused if `autotune_reg_error` is true)
  min_ncrit::Int64                 # Minimum number of particles per leaf (default to 50 for SFS robustness)
end

function FMM(; p=4, ncrit=10, theta=0.5, shrink_recenter=true, relative_tolerance=1e-3, absolute_tolerance=1e-3, autotune_p=true, autotune_ncrit=true, autotune_reg_error=true, default_rho_over_sigma=1.0, min_ncrit=50)
    return FMM(p, ncrit, theta, shrink_recenter, relative_tolerance, absolute_tolerance, autotune_p, autotune_ncrit, autotune_reg_error, default_rho_over_sigma, min_ncrit)
end

################################################################################
# MERGING WORKSPACE
################################################################################
# Scratch buffers reused across calls to `merge_particles!` to avoid heap
# allocations in the simulation hot loop. All Int — float accumulators are
# stack-local scalars in the merge routine.
mutable struct MergingWorkspace
    candidate_indices::Vector{Int}
    sorted_indices::Vector{Int}
    offsets::Vector{Int}
    counts::Vector{Int}
    keys::Vector{Int}
    parent::Vector{Int}
    rank::Vector{Int}
    root_count::Vector{Int}
    representative::Vector{Int}
    roots::Vector{Int}
    root_offset::Vector{Int}
    candidates_by_root::Vector{Int}
    to_remove::Vector{Int}
end

MergingWorkspace() = MergingWorkspace(
    Int[], Int[], Int[], Int[], Int[], Int[], Int[],
    Int[], Int[], Int[], Int[], Int[], Int[],
)

################################################################################
# SPLITTING STATE AND WORKSPACE
################################################################################
# Per-particle persistent state for splitting. All vectors are sized to
# maxparticles and indexed in lockstep with `pfield.particles` columns.
# add_particle initializes the new slot; remove_particle's swap-with-last
# semantics copy entries `i ← np` so that data tracks the particle that
# now occupies slot `i`.
mutable struct SplittingState{R}
    sigma_0::Vector{R}              # reference smoothing radius at creation
    H_chi::Vector{R}                # accumulated overlap-loss exposure
    hold_counter::Vector{Int}       # consecutive steps trigger has held
    cooldown_counter::Vector{Int}   # remaining steps a child cannot re-split
end

SplittingState{R}(maxparticles::Int) where {R} = SplittingState{R}(
    zeros(R, maxparticles),
    zeros(R, maxparticles),
    zeros(Int, maxparticles),
    zeros(Int, maxparticles),
)

# Scratch buffers reused across calls to `split_particles!` to avoid heap
# allocations in the simulation hot loop.
mutable struct SplittingWorkspace{R}
    candidate_indices::Vector{Int}
    severity::Vector{R}
    order::Vector{Int}
end

SplittingWorkspace{R}() where {R} = SplittingWorkspace{R}(Int[], R[], Int[])

################################################################################
# FILAMENT EDGE GRAPH (hybrid filament-edge refinement, Phase 2)
################################################################################
# Bounded directed edge graph: each particle has ≤2 upstream and ≤2 downstream
# active edges. Adjacency-only — no separate edge list. An edge from src→dst
# is represented by matching slots: `down_neighbor[k, src] == dst` and
# `up_neighbor[k′, dst] == src`. Per-edge state (coherence, score) lives
# once, on the canonical (downstream) side. Read an upstream-incident edge's
# state by following `up_neighbor[k, i]` back to src and looking up src's
# down_* slot that points at i. Slot value 0 = empty. `degree` packs both
# counts into one byte: low nibble = up_count, high nibble = down_count.
mutable struct FilamentEdgeGraph{R}
    up_neighbor::Matrix{Int}     # (2, maxparticles)
    down_neighbor::Matrix{Int}   # (2, maxparticles)
    down_coherent::BitMatrix     # (2, maxparticles)
    down_score::Matrix{R}        # (2, maxparticles)
    degree::Vector{UInt8}        # low nibble = up, high nibble = down
    filament_id::Vector{Int}     # optional explicit identity, 0 = none
end

FilamentEdgeGraph{R}(maxparticles::Int) where {R} = FilamentEdgeGraph{R}(
    zeros(Int, 2, maxparticles),
    zeros(Int, 2, maxparticles),
    falses(2, maxparticles),
    zeros(R, 2, maxparticles),
    zeros(UInt8, maxparticles),
    zeros(Int, maxparticles),
)

# Nibble-packed degree accessors. Counts are 0..2 so 4 bits each is ample.
@inline up_count(g::FilamentEdgeGraph, i::Int)   = Int(g.degree[i] & 0x0f)
@inline down_count(g::FilamentEdgeGraph, i::Int) = Int(g.degree[i] >> 4)
@inline function set_up_count!(g::FilamentEdgeGraph, i::Int, n::Integer)
    g.degree[i] = (g.degree[i] & 0xf0) | (UInt8(n) & 0x0f)
end
@inline function set_down_count!(g::FilamentEdgeGraph, i::Int, n::Integer)
    g.degree[i] = (g.degree[i] & 0x0f) | (UInt8(n) << 4)
end

# Slot lookups — bounded local scan over 2 entries.
@inline function find_down_slot(g::FilamentEdgeGraph, src::Int, dst::Int)
    g.down_neighbor[1, src] == dst && return 1
    g.down_neighbor[2, src] == dst && return 2
    return 0
end
@inline function find_up_slot(g::FilamentEdgeGraph, dst::Int, src::Int)
    g.up_neighbor[1, dst] == src && return 1
    g.up_neighbor[2, dst] == src && return 2
    return 0
end

# Scratch buffers for Phase 3 inference/validation. Cell-list buffers mirror
# `MergingWorkspace` so `_build_cell_list!` (FLOWVPM_merging.jl) can be reused.
# Resized on demand.
mutable struct FilamentEdgeWorkspace{R}
    candidates::Vector{Int}
    sorted_indices::Vector{Int}
    offsets::Vector{Int}
    counts::Vector{Int}
    keys::Vector{Int}
    axis_x::Vector{R}
    axis_y::Vector{R}
    axis_z::Vector{R}
    axis_ok::BitVector
    up_best::Matrix{Int}
    down_best::Matrix{Int}
    up_best_score::Matrix{R}
    down_best_score::Matrix{R}
    visits::Vector{Int}
    capped::BitVector
end

FilamentEdgeWorkspace{R}() where {R} = FilamentEdgeWorkspace{R}(
    Int[], Int[], Int[], Int[], Int[],
    R[], R[], R[],
    BitVector(),
    zeros(Int, 2, 0), zeros(Int, 2, 0),
    zeros(R, 2, 0), zeros(R, 2, 0),
    Int[], BitVector(),
)

FilamentEdgeWorkspace{R}(maxparticles::Int) where {R} = FilamentEdgeWorkspace{R}(
    Int[], Int[], Int[], Int[], Int[],
    R[], R[], R[],
    BitVector(),
    zeros(Int, 2, maxparticles), zeros(Int, 2, maxparticles),
    zeros(R, 2, maxparticles), zeros(R, 2, maxparticles),
    Int[], BitVector(),
)

################################################################################
# PARTICLE FIELD STRUCT
################################################################################
mutable struct ParticleField{R, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TRelaxation, AT<:AbstractMatrix{R}}
    # User inputs
    maxparticles::Int                           # Maximum number of particles
    particles::AT                               # Array of particles
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme
    scratch::AT                                 # Reusable scratch space for vectorized intermediate computations (see SCRATCH_INDEX)

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
    fmm::FMM                                    # Fast-multipole settings
    useGPU::Int                                 # run on GPU if >0, CPU if 0
    merging_workspace::MergingWorkspace         # Scratch buffers for merge_particles!
    splitting_state::SplittingState{R}          # Per-particle state for split_particles!
    splitting_workspace::SplittingWorkspace{R}  # Scratch buffers for split_particles!
    filament_edge_graph::FilamentEdgeGraph{R}            # Bounded 2-in/2-out edge graph
    filament_edge_workspace::FilamentEdgeWorkspace{R}    # Scratch for inference/validation
    track_H_chi::Bool                           # If true, accumulate H_chi each accepted step
    H_chi_axis::Symbol                          # :strength, :streamline, :strain1, or :max
    H_chi_clip_positive::Bool                   # If true, clip to max(0, λ_χ) before integrating
end

"""
    ParticleField(maxparticles::Int, R=FLOAT_TYPE; <keyword arguments>)

Create a new particle field with `maxparticles` particles. The particle field
is created with the default values for the other parameters.

# Arguments
- `maxparticles::Int`: Maximum number of particles in the field.
- `R=FLOAT_TYPE`: Type of the particle field. Default is `FLOAT_TYPE`.

# Keyword Arguments
- `formulation::Formulation=ReformulatedVPM{FLOAT_TYPE}(0, 1/5)`: VPM governing equations. Default is reformulated to conserve mass for a tube and angular momentum for a sphere.
- `viscous::ViscousScheme=Inviscid()`: Viscous scheme. Note that with `rVPM` formulation, artificial viscosity is not needed for numerical stability, as is common in VPM.
- `np::Int=0`: Number of particles currently in the field.
- `nt::Int=0`: Current time step number.
- `t::Real=0`: Current simulation time.
- `transposed::Bool=true`: If true, the transposed scheme of the stretching term is used (recommended for stability).
- `fmm::FMM`: Fast multipole method tuning and auto-tuning settings.
- `M::Array{R, 1}=zeros(R, 4)`: Auxilliary memory for computations. Should not be modified for most purposes.
- `SFS::SubFilterScale=NoSFS{FLOAT_TYPE}()`: Subfilter-scale turbulence model.
- `kernel::Kernel=Kernel(zeta_gauserf, g_gauserf, dgdr_gauserf, g_dgdr_gauserf)`: Regularization scheme. Default is Gaussian smoothing of the vorticity field.
- `UJ=UJ_fmm`: Method used to compute the \$N\$-body problem. Default uses the fast multipole method to achieve \$O(N)\$ complexity.
- `Uinf::Function=(t) -> [0.0,0.0,0.0]`: Uniform freestream velocity function Uinf(t).
- `relaxation::Relaxation=Relaxation(relax_pedrizzetti, 1, 0.3)`: Relaxation scheme to re-align the vorticity field to be divergence-free.
- `integration::Tintegration=rungekutta3`: Time integration scheme. Default is a Runge-Kutta 3rd order, low-memory scheme.
- `useGPU::Int`: Run on GPU if >0, CPU if 0. Default is 0. (Experimental and does not accelerate SFS calculations)
- `arraytype::Type{<:AbstractMatrix}=Matrix`: Array type backing `pfield.particles`. Default is `Matrix`
    (CPU). Pass `CuArray` (CUDA.jl) to allocate the particle field on GPU.
"""
function ParticleField(maxparticles::Int, R=FLOAT_TYPE;
        formulation::F=formulation_default,
        viscous::V=Inviscid(),
        np=0, nt=0, t=zero(R),
        transposed=true,
        fmm::FMM=FMM(),
        SFS::S=SFS_default, kernel::Tkernel=kernel_default,
        UJ::TUJ=UJ_fmm, Uinf::TUinf=Uinf_default,
        relaxation::TR=Relaxation(relax_correctedpedrizzetti, 1, 0.3), # default relaxation has no type input, which is a problem for AD.
        integration::Tintegration=rungekutta3,
        useGPU=useGPU_default,
        arraytype=Matrix
    ) where {F, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel<:Kernel, TUJ, Tintegration, TR}

    # create particle field
    particles = fill!(arraytype{R}(undef, nfields, maxparticles), zero(R))
    scratch = fill!(arraytype{R}(undef, N_SCRATCH, maxparticles), zero(R))

    # Generate and return ParticleField
    return ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, typeof(particles)}(maxparticles, particles,
                                            formulation, viscous, scratch, np, nt, t,
                                            kernel, UJ, Uinf, SFS, integration,
                                            transposed, relaxation, fmm, useGPU,
                                            MergingWorkspace(),
                                            SplittingState{R}(maxparticles),
                                            SplittingWorkspace{R}(),
                                            FilamentEdgeGraph{R}(maxparticles),
                                            FilamentEdgeWorkspace{R}(maxparticles),
                                            false, :strength, true)
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

    if pfield.particles isa Array
        # Populate the empty particle
        set_X(pfield, i_next, X)
        set_Gamma(pfield, i_next, Gamma)
        set_sigma(pfield, i_next, sigma)
        set_vol(pfield, i_next, vol)
        set_circulation(pfield, i_next, circulation)
        set_C(pfield, i_next, C)
        set_static(pfield, i_next, Float64(static))
    else
        _add_particle_broadcast!(pfield, i_next, X, Gamma, sigma, vol, circulation, C, static)
    end

    # Initialize per-particle splitting state for this slot
    R = eltype(pfield.particles)
    st = pfield.splitting_state
    st.sigma_0[i_next] = R(sigma)
    st.H_chi[i_next] = zero(R)
    st.hold_counter[i_next] = 0
    st.cooldown_counter[i_next] = 0

    # Initialize filament edge graph adjacency for this slot. Empty
    # (degree 0, all slots zero); edges are wired later by add_edge!.
    g = pfield.filament_edge_graph
    g.up_neighbor[1, i_next] = 0; g.up_neighbor[2, i_next] = 0
    g.down_neighbor[1, i_next] = 0; g.down_neighbor[2, i_next] = 0
    g.down_coherent[1, i_next] = false; g.down_coherent[2, i_next] = false
    g.down_score[1, i_next] = zero(R); g.down_score[2, i_next] = zero(R)
    g.degree[i_next] = UInt8(0)
    g.filament_id[i_next] = 0

    return nothing
end

"""
    `_add_particle_broadcast!(pfield, i, X, Gamma, sigma, vol, circulation, C, static)`

GPU-safe particle population used by `add_particle` when `pfield.particles`
is not a plain `Array` (e.g. `CuArray`). Two separate CUDA.jl restrictions
apply here, not just one:
  1. The scalar per-field setters (`set_sigma`, `set_vol`, `set_circulation`,
     `set_static`) assign into a single (row, column) index, which is
     disallowed scalar indexing on a `CuArray`. Indexing with a length-1
     range instead of a bare `Int` keeps every write a proper
     (kernel-dispatched) array broadcast.
  2. Broadcasting a host `Vector` (e.g. `X`/`Gamma`, as built by every caller
     in this codebase) directly onto a `CuArray` destination fails to
     compile ("passing non-bitstype argument") because a heap-allocated
     `Vector` isn't a valid GPU kernel argument. Converting it to a `Tuple`
     first (an immutable, stack-allocated bitstype) fixes this.
"""
_add_particle_gpu_bcast_val(v::AbstractArray) = Tuple(v)
_add_particle_gpu_bcast_val(v) = v

function _add_particle_broadcast!(pfield::ParticleField, i::Int, X, Gamma, sigma,
                                                    vol, circulation, C, static)
    view(pfield.particles, X_INDEX, i) .= _add_particle_gpu_bcast_val(X)
    view(pfield.particles, GAMMA_INDEX, i) .= _add_particle_gpu_bcast_val(Gamma)
    view(pfield.particles, SIGMA_INDEX:SIGMA_INDEX, i) .= sigma
    view(pfield.particles, VOL_INDEX:VOL_INDEX, i) .= vol
    view(pfield.particles, CIRCULATION_INDEX:CIRCULATION_INDEX, i) .= circulation
    view(pfield.particles, C_INDEX, i) .= _add_particle_gpu_bcast_val(C)
    view(pfield.particles, STATIC_INDEX:STATIC_INDEX, i) .= Float64(static)
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
const U_PREV_INDEX = 44

# Generic scratch rows: reusable per-particle storage for named intermediate
# quantities in vectorized (broadcast) hot-path computations (e.g. RK3's
# update_particle_states), so those intermediates write into a persistent
# buffer with `.=` instead of heap-allocating a fresh array every call. Row
# meaning is reused/contextual per call site (documented locally), the same
# way M_INDEX's rows already carry different meanings across integration
# schemes -- this is not new-field-per-quantity storage. Same fixed row count
# for every formulation (no per-formulation sizing); a formulation that needs
# fewer rows (e.g. ClassicVPM) simply leaves the rest unused.
#
# 11 is the true minimum for ReformulatedVPM's update_particle_states, achieved
# by aliasing rows whose value is provably dead before another quantity needs
# a row (see comments at each reuse site in FLOWVPM_timeintegration.jl) --
# every formula here is purely elementwise per-particle, so overwriting a row
# with `x .= f(x, ...)` (reading and writing the same row) is well-defined,
# same as the standard `x .= x .+ 1` pattern.
const N_SCRATCH = 11
const SCRATCH_INDEX = 1:N_SCRATCH

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
get_U_prev(P) = view(P, U_PREV_INDEX)

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
get_U_prev(pfield::ParticleField, i::Int) = view(pfield.particles, U_PREV_INDEX, i)

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
function set_U_prev(P, val) P[U_PREV_INDEX] = val end


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
function set_U_prev(pfield::ParticleField, i::Int, val) pfield.particles[U_PREV_INDEX, i] = val end


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

    np = get_np(pfield)
    st = pfield.splitting_state
    g = pfield.filament_edge_graph
    R = eltype(pfield.particles)

    # Phase A: drop slot i's incident edges. Each remove_edge! clears the
    # matching slot on the other endpoint via a size-2 mirror scan, so this
    # is bounded local work (≤4 edges total).
    @inbounds for k in 1:2
        d = g.down_neighbor[k, i]
        d != 0 && remove_edge!(g, i, d)
    end
    @inbounds for k in 1:2
        u = g.up_neighbor[k, i]
        u != 0 && remove_edge!(g, u, i)
    end

    if i != np
        # Overwrite target particle with last particle in the field
        get_particle(pfield, i) .= get_particle(pfield, np)
        # Mirror swap-with-last in the splitting side-buffers
        st.sigma_0[i] = st.sigma_0[np]
        st.H_chi[i] = st.H_chi[np]
        st.hold_counter[i] = st.hold_counter[np]
        st.cooldown_counter[i] = st.cooldown_counter[np]
        # Phase B: mirror swap-with-last in the edge-graph adjacency. The
        # neighbors of the moved particle still point at slot np; Phase C
        # below rewrites them to i.
        @inbounds for k in 1:2
            g.up_neighbor[k, i]   = g.up_neighbor[k, np]
            g.down_neighbor[k, i] = g.down_neighbor[k, np]
            g.down_coherent[k, i] = g.down_coherent[k, np]
            g.down_score[k, i]    = g.down_score[k, np]
        end
        g.degree[i]      = g.degree[np]
        g.filament_id[i] = g.filament_id[np]

        # Phase C: rewrite back-references. For each downstream neighbor of
        # the moved slot, find its up_neighbor entry pointing at np and
        # rewrite to i. Mirror for upstream neighbors.
        @inbounds for k in 1:2
            d = g.down_neighbor[k, i]
            if d != 0
                slot = find_up_slot(g, d, np)
                slot != 0 && (g.up_neighbor[slot, d] = i)
            end
        end
        @inbounds for k in 1:2
            u = g.up_neighbor[k, i]
            if u != 0
                slot = find_down_slot(g, u, np)
                slot != 0 && (g.down_neighbor[slot, u] = i)
            end
        end
    end

    # Remove last particle in the field
    _reset_particle(pfield, np)
    st.sigma_0[np] = zero(R)
    st.H_chi[np] = zero(R)
    st.hold_counter[np] = 0
    st.cooldown_counter[np] = 0
    @inbounds for k in 1:2
        g.up_neighbor[k, np]   = 0
        g.down_neighbor[k, np] = 0
        g.down_coherent[k, np] = false
        g.down_score[k, np]    = zero(R)
    end
    g.degree[np]      = UInt8(0)
    g.filament_id[np] = 0
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
function nextstep(pfield::ParticleField, dt::Real; update_U_prev=true, optargs...)

    # Step in time
    if get_np(pfield)!=0
        pfield.integration(pfield, dt; optargs...)
    end

    # update U_prev
    if update_U_prev
        if pfield.np > MIN_MT_NP
            Threads.@threads for i in 1:pfield.np
                Ux, Uy, Uz = get_U(pfield, i)
                set_U_prev(pfield, i, sqrt(Ux*Ux + Uy*Uy + Uz*Uz))
            end
        else
            for i in 1:pfield.np
                Ux, Uy, Uz = get_U(pfield, i)
                set_U_prev(pfield, i, sqrt(Ux*Ux + Uy*Uy + Uz*Uz))
            end
        end
    end

    # Accumulate H_chi exposure for split triggers (no-op unless
    # pfield.track_H_chi is set by a SeparationTrigger)
    accumulate_H_chi!(pfield, dt)

    # Updates time
    pfield.t += dt
    pfield.nt += 1
end


##### INTERNAL FUNCTIONS #######################################################
function _reset_particles(pfield::ParticleField)
    if pfield.particles isa Array
        zeroVal = zero(eltype(pfield.particles))
        if pfield.np > MIN_MT_NP
            Threads.@threads for i in 1:pfield.np
                (pfield.particles[STATIC_INDEX, i] == 0) && _reset_particle(pfield, i; zeroVal)
            end
        else
            for i in 1:pfield.np
                (pfield.particles[STATIC_INDEX, i] == 0) && _reset_particle(pfield, i; zeroVal)
            end
        end
    else
        _reset_particles_broadcast!(pfield)
    end
end

"""
    `_reset_particles_broadcast!(pfield)`

GPU-safe equivalent of `_reset_particles`'s per-particle loop, used when
`pfield.particles` is not a plain `Array`. The scalar `pfield.particles[STATIC_INDEX, i]
== 0` check in the loop version is disallowed indexing on a `CuArray`;
multiplying by the (0/1) static mask gets the same "leave static particles
untouched, zero everything else" behavior as a whole-array broadcast.
"""
function _reset_particles_broadcast!(pfield::ParticleField)
    np = pfield.np
    is_static = view(pfield.particles, STATIC_INDEX:STATIC_INDEX, 1:np)
    view(pfield.particles, U_INDEX, 1:np) .*= is_static
    view(pfield.particles, VORTICITY_INDEX, 1:np) .*= is_static
    view(pfield.particles, J_INDEX, 1:np) .*= is_static
    view(pfield.particles, PSE_INDEX, 1:np) .*= is_static
    return nothing
end

function _reset_particle(particle)
    zeroVal = zero(eltype(particle))
    set_U(particle, zeroVal)
    set_vorticity(particle, zeroVal)
    set_J(particle, zeroVal)
    set_PSE(particle, zeroVal)
end

function _reset_particle(pfield::ParticleField, i::Int; zeroVal=zero(eltype(pfield.particles)))
    pfield.particles[U_INDEX, i] .= zeroVal
    pfield.particles[VORTICITY_INDEX, i] .= zeroVal
    pfield.particles[J_INDEX, i] .= zeroVal
    pfield.particles[PSE_INDEX, i] .= zeroVal
end

function _reset_particles_sfs(pfield::ParticleField)
    if pfield.particles isa Array
        zeroVal = zero(eltype(pfield.particles))
        if pfield.np > MIN_MT_NP
            Threads.@threads for i in 1:pfield.np
                (pfield.particles[STATIC_INDEX, i] == 0) && _reset_particle_sfs(pfield, i; zeroVal)
            end
        else
            for i in 1:pfield.np
                (pfield.particles[STATIC_INDEX, i] == 0) && _reset_particle_sfs(pfield, i; zeroVal)
            end
        end
    else
        _reset_particles_sfs_broadcast!(pfield)
    end
end

"""
    `_reset_particles_sfs_broadcast!(pfield)`

GPU-safe equivalent of `_reset_particles_sfs`'s per-particle loop, used when
`pfield.particles` is not a plain `Array`. Same static-particle-mask
broadcast technique as `_reset_particles_broadcast!`, applied to `SFS_INDEX`.
"""
function _reset_particles_sfs_broadcast!(pfield::ParticleField)
    np = pfield.np
    is_static = view(pfield.particles, STATIC_INDEX:STATIC_INDEX, 1:np)
    view(pfield.particles, SFS_INDEX, 1:np) .*= is_static
    return nothing
end

function _reset_particle_sfs(pfield::ParticleField, i::Int; zeroVal=zero(eltype(pfield.particles)))
    pfield.particles[SFS_INDEX, i] .= zeroVal
end

function _reset_particle_sfs(particle)
    set_SFS(particle, zero(eltype(particle)))
end

##### END OF PARTICLE FIELD#####################################################
