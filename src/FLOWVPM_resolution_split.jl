#=##############################################################################
# DESCRIPTION
    Resolution-preserving particle splitting (BRAINSTORM 026, Phase 2).

    Fully independent of the experimental splitting machinery in
    `FLOWVPM_splitting.jl` (`SplittingState`/`SplitOptions`/`SplitTrigger`) per
    Ryan's ruling 2026-09-07: no shared structs, no trigger trees. Entry point
    is a new method `split_particles!(pfield, ::ResolutionSplitOpts)`.

    Two mechanisms (both OFF by default):
    * Viscous (Mechanism A, spec §3a): isotropic growth from core spreading —
      4-child tetrahedron, `σ_c = σ_p·4^(-1/3)`.
    * Stretch (Mechanism B, spec §3b + Ryan 2026-09-07 two-regime ruling):
      - compression regime (grow side): 3-child triangle in the plane normal
        to the averaged stretch axis, `σ_c = σ_p/√3` (W5 mass-per-length rule);
      - elongation regime (shrink side): 2 in-line children along the stretch
        axis with `σ_c = σ_p` (re-discretizes length; cross-section unchanged).

    The split plane/axis uses the sign-invariant running average of the
    stretching vector S accumulated since the particle's last split (reset on
    split/merge), falling back to Γ̂ when incoherent (D1 closed 2026-09-07).

    NOTE on include order: this file must be included BEFORE
    `FLOWVPM_particlefield.jl` because `ParticleField` carries a
    `resolution_split::Union{Nothing,ResolutionSplitState{R}}` field. Functions
    here therefore leave their `pfield` argument unannotated (they dispatch on
    the opts/state types); all particle accessors resolve at call time.
=###############################################################################

################################################################################
# STATE
################################################################################
"""
    ResolutionSplitState{R}(maxparticles)

Per-particle state for resolution-preserving splitting. All arrays are sized to
`maxparticles` and indexed in lockstep with `pfield.particles` columns:
`add_particle` initializes the new slot, `remove_particle`'s swap-with-last
semantics mirror `i ← np` (see `_rsplit_swap!`/`_rsplit_zero!`).

Semantics: mean stretch axis = `normalize(axis[:,i])`; coherence =
`|axis[:,i]|/weight[i] ∈ (0,1]`. Each sample `s = dt·S` is sign-aligned to the
current sum (`dot(s, axis) < 0 ⇒ s ← −s`) before adding, so anti-parallel
flapping accumulates instead of cancelling. `exposure` is the accumulated
separation exposure `Σ dt·λ` with `λ = (Γ·S)/|Γ|²` (log length-stretch along
Γ̂). All of axis/weight/exposure/dvisc/drvpm reset on split and on merge (the
merged representative is a new entity).

Anti-refire WITHOUT cooldown — each trigger is self-limiting via its own reset:
* ratio trigger: children/merged rep get fresh `sigma_0` → ratio restarts at 1;
* abs cap: grow-side kernels (tetra4, tri3) birth children at `σ_c < σ_p ≈ cap`
  → must regrow to refire (pair2's `σ_c = σ_p` only fires on shrink events,
  far from the cap);
* exposure: reset to 0 on split → must re-accumulate `log_stretch_max`;
* floor: fires only when `sigma_0 > floor` — children born at/below the floor
  can never refire it (floor trigger permanently disarmed for that lineage;
  the exposure trigger still covers them).
"""
mutable struct ResolutionSplitState{R}
    # per-particle arrays, lockstep with pfield.particles columns — that's ALL
    # of it: no cooldown, no scratch (the main pass is a single serial loop)
    sigma_0::Vector{R}     # σ reference at creation / last split / last merge
    axis::Matrix{R}        # (3, maxp) sign-aligned running Σ dt·S since last split
    weight::Vector{R}      # Σ dt·|S| since last split (coherence denominator)
    exposure::Vector{R}    # Σ dt·λ along Γ̂ since last split (shrink separation exposure)
    dvisc::Vector{R}       # Σ Δσ² from viscous spreading   (attribution → viscous mech)
    drvpm::Vector{R}       # Σ Δσ² from rVPM compression    (attribution → stretch mech)
end

ResolutionSplitState{R}(maxparticles::Int) where {R} = ResolutionSplitState{R}(
    zeros(R, maxparticles),
    zeros(R, 3, maxparticles),
    zeros(R, maxparticles),
    zeros(R, maxparticles),
    zeros(R, maxparticles),
    zeros(R, maxparticles),
)

"""
    enable_resolution_split!(pfield)

Lazily attach a `ResolutionSplitState` to `pfield` (no-op if already attached).
Existing particles get `sigma_0` seeded from their current σ; all accumulators
start at zero. While `pfield.resolution_split === nothing` every hook in this
file is a no-op (one branch), so the feature costs nothing when off.
"""
function enable_resolution_split!(pfield)
    if pfield.resolution_split === nothing
        R = eltype(pfield.particles)
        rs = ResolutionSplitState{R}(pfield.maxparticles)
        for i in 1:get_np(pfield)
            rs.sigma_0[i] = get_sigma(pfield, i)[]
        end
        pfield.resolution_split = rs
    end
    return pfield.resolution_split
end

################################################################################
# OPTIONS
################################################################################
"""
    ResolutionSplitOpts{R}(; kwargs...)

Flat options for `split_particles!(pfield, opts::ResolutionSplitOpts)`. Each
trigger threshold is independently disabled by `NaN` (the default); a particle
splits when its quantity crosses an armed threshold AND the routed mechanism is
enabled (disabled mechanism ⇒ skip + count, never reroute).

Naming: `viscous_*` is the isotropic viscous mechanism (spec §3a); the stretch
mechanism's two regimes are `compress_*` (negative stretch → 3-child triangle)
and `elongate_*` (positive stretch → 2 in-line children). Both stretch regimes
share `enable_stretch_split`.
"""
struct ResolutionSplitOpts{R}
    # triggers — NaN disables each independently
    sigma_max::R              # grow:   σ > sigma_max (absolute cap)
    sigma_growth_ratio_max::R # grow:   σ/σ₀ > this
    log_stretch_max::R        # shrink: accumulated log length-stretch (exposure) > this
    sigma_floor::R            # shrink: σ pinned on the floor (σ ≤ floor·(1+1e-6))
    # mechanisms
    enable_viscous_split::Bool # §3a: isotropic 4-child tetrahedron
    enable_stretch_split::Bool # §3b + 2026-09-07 two-regime ruling
    # child placement — child offset from parent center, as a ratio of parent σ
    viscous_offset_ratio::R   # tetra vertex radius / σ_p (default: second-moment match)
    compress_offset_ratio::R  # tri3 ring radius / σ_p (compression; default spacing 1.8 σ_c)
    elongate_offset_ratio::R  # pair2 half-spacing / σ_p (elongation; default spacing 1.0 σ_p)
    # split-plane orientation for the stretch mechanism
    use_stretch_axis::Bool    # false → split plane ⟂ Γ̂ always (comparison arm)
    axis_coherence_min::R     # below this coherence → fall back to Γ̂ (still splits)
end

function ResolutionSplitOpts{R}(;
            sigma_max              = R(NaN),
            sigma_growth_ratio_max = R(NaN),
            log_stretch_max        = R(NaN),
            sigma_floor            = R(NaN),
            enable_viscous_split::Bool = false,
            enable_stretch_split::Bool = false,
            viscous_offset_ratio   = R(1.3503),
            compress_offset_ratio  = R(0.6),
            elongate_offset_ratio  = R(0.5),
            use_stretch_axis::Bool = true,
            axis_coherence_min     = R(0.5),
        ) where {R}
    return ResolutionSplitOpts{R}(R(sigma_max), R(sigma_growth_ratio_max),
                R(log_stretch_max), R(sigma_floor),
                enable_viscous_split, enable_stretch_split,
                R(viscous_offset_ratio), R(compress_offset_ratio),
                R(elongate_offset_ratio),
                use_stretch_axis, R(axis_coherence_min))
end

ResolutionSplitOpts(; kwargs...) = ResolutionSplitOpts{FLOAT_TYPE}(; kwargs...)

################################################################################
# LOCKSTEP LIFECYCLE (called from add_particle/remove_particle when enabled)
################################################################################
"Initialize slot `i` for a newly created particle with smoothing radius `sigma`."
@inline function _rsplit_init_slot!(rs::ResolutionSplitState, i::Int, sigma)
    rs.sigma_0[i] = sigma
    rs.axis[1, i] = 0; rs.axis[2, i] = 0; rs.axis[3, i] = 0
    rs.weight[i] = 0
    rs.exposure[i] = 0
    rs.dvisc[i] = 0
    rs.drvpm[i] = 0
    return nothing
end

"""
Reset slot `i` to a fresh entity with reference radius `sigma` (used on split
children in-place and by the merge `on_representative` hook — the merged
particle is a new entity, W3).
"""
@inline _rsplit_reset_slot!(rs::ResolutionSplitState, i::Int, sigma) =
    _rsplit_init_slot!(rs, i, sigma)

"Mirror `remove_particle`'s swap-with-last: copy slot `np`'s state into slot `i`."
@inline function _rsplit_swap!(rs::ResolutionSplitState, i::Int, np::Int)
    rs.sigma_0[i] = rs.sigma_0[np]
    rs.axis[1, i] = rs.axis[1, np]
    rs.axis[2, i] = rs.axis[2, np]
    rs.axis[3, i] = rs.axis[3, np]
    rs.weight[i] = rs.weight[np]
    rs.exposure[i] = rs.exposure[np]
    rs.dvisc[i] = rs.dvisc[np]
    rs.drvpm[i] = rs.drvpm[np]
    return nothing
end

"Zero the vacated tail slot `np` after a removal."
@inline function _rsplit_zero!(rs::ResolutionSplitState, np::Int)
    rs.sigma_0[np] = 0
    rs.axis[1, np] = 0; rs.axis[2, np] = 0; rs.axis[3, np] = 0
    rs.weight[np] = 0
    rs.exposure[np] = 0
    rs.dvisc[np] = 0
    rs.drvpm[np] = 0
    return nothing
end

################################################################################
# ACCUMULATION (integrator-inline; caller guards on `rs === nothing`)
################################################################################
"""
    _rsplit_accumulate!(rs, i, dt, sx, sy, sz, gx, gy, gz)

Accumulate one stretching sample `S = (sx,sy,sz)` (and strength `Γ = (gx,gy,gz)`)
for particle `i` over time weight `dt`: sign-invariant axis average, coherence
weight, and separation exposure `λ = (Γ·S)/|Γ|²` — all in a single pass, called
where S is already in hand inside the integrator (~20 flops).
"""
@inline function _rsplit_accumulate!(rs::ResolutionSplitState, i::Int, dt,
                                     sx, sy, sz, gx, gy, gz)
    n = sqrt(sx*sx + sy*sy + sz*sz)
    n > 0 || return nothing
    ax = rs.axis
    d = ax[1, i]*sx + ax[2, i]*sy + ax[3, i]*sz
    sgn = d < 0 ? -one(n) : one(n)
    ax[1, i] += sgn*dt*sx; ax[2, i] += sgn*dt*sy; ax[3, i] += sgn*dt*sz
    rs.weight[i] += dt*n
    g2 = gx*gx + gy*gy + gz*gz
    g2 > 0 && (rs.exposure[i] += dt*(gx*sx + gy*sy + gz*sz)/g2)   # λ = Γ·S/|Γ|²
    return nothing
end

"""
    _rsplit_accumulate_dsigma2!(rs, i, dv, dr)

Mirror-write the applied Δσ² attribution for particle `i`: `dv` from viscous
spreading, `dr` from rVPM compression. Colocated with the landed
`SplittingState.dsigma2_visc/rvpm` writes (which stay untouched); routes
grow-side split events to the dominant mechanism.
"""
@inline function _rsplit_accumulate_dsigma2!(rs::ResolutionSplitState, i::Int, dv, dr)
    rs.dvisc[i] += dv
    rs.drvpm[i] += dr
    return nothing
end

################################################################################
# DIRECTION RESOLUTION
################################################################################
"""
    _rsplit_direction(rs, i, opts, pfield) -> (ex, ey, ez)

Unit direction for the stretch mechanism's split geometry: the averaged stretch
axis when armed and coherent (`|axis|/weight ≥ axis_coherence_min`), else Γ̂
(STRENGTH fallback; D1 closed 2026-09-07). Always returns a direction —
coherence never refuses a split.
"""
@inline function _rsplit_direction(rs::ResolutionSplitState, i::Int,
                                   opts::ResolutionSplitOpts, pfield)
    if opts.use_stretch_axis && rs.weight[i] > 0
        ax, ay, az = rs.axis[1, i], rs.axis[2, i], rs.axis[3, i]
        n = sqrt(ax*ax + ay*ay + az*az)
        if n > 0 && n/rs.weight[i] >= opts.axis_coherence_min
            return (ax/n, ay/n, az/n)
        end
    end
    G = get_Gamma(pfield, i)
    gn = sqrt(G[1]*G[1] + G[2]*G[2] + G[3]*G[3])
    gn > 0 && return (G[1]/gn, G[2]/gn, G[3]/gn)
    return (one(gn), zero(gn), zero(gn))  # Γ = 0 degenerate: fixed arbitrary axis
end
