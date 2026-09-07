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
################################################################################
# ORIENTATION DRAWS (fully random per Ryan 2026-09-05 — no lineage/seed state;
# NOT reproducible across warm starts, accepted for now. REVISIT if bitwise
# A/B across restarts becomes needed.)
################################################################################
"""
    _random_rotation() -> SMatrix{3,3,Float64}

Uniform random rotation on SO(3) via the Shoemake quaternion construction
from three `rand()` draws. Called from the serial split pass — the default
task-local RNG is safe.
"""
function _random_rotation()
    u1, u2, u3 = rand(), rand(), rand()
    s1 = sqrt(1 - u1); s2 = sqrt(u1)
    a1 = 2pi*u2; a2 = 2pi*u3
    qw = s1*sin(a1); qx = s1*cos(a1); qy = s2*sin(a2); qz = s2*cos(a2)
    return SMatrix{3,3,Float64}(
        1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qz*qw),     2*(qx*qz - qy*qw),
        2*(qx*qy - qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qx*qw),
        2*(qx*qz + qy*qw),     2*(qy*qz - qx*qw),     1 - 2*(qx*qx + qy*qy))
end

"Uniform in-plane angle ψ ∈ [0, 2π) for the tri3 kernel's triangle orientation."
_inplane_angle() = 2pi*rand()

################################################################################
# SHARED CHILD EMISSION
################################################################################
"""
    _rsplit_emit_child!(pfield, rs, slot, x, y, z, gx, gy, gz, sigma_c, circ,
                        is_stat)

Emit one split child. `slot > 0` overwrites that existing column in place
(child 1 reuses the parent's slot); `slot == 0` appends via `add_particle`
(caller guarantees headroom). Either way the child gets:

* `vol = (4/3)π σ_c³` — W5 σ³-consistent hygiene rule, NOT `vol_p/m`. This
  intentionally breaks merge inversion (merging's `σ = cbrt(Σσ³)` no longer
  reproduces the parent) and, for the pair2 kernel's `σ_c = σ_p`, doubles the
  σ-implied total volume — accepted, `vol` feeds no dynamics.
* zeroed U, vorticity, J, PSE, M, C, SFS, U_prev — maintenance runs
  post-convection and UJ re-evaluates next step; zeroing M also clears
  euler_exp's M[9] Zeff stash, which is correct for a fresh child.
* fresh ResolutionSplitState (`sigma_0 = σ_c`, accumulators zero) and, for
  the in-place slot, an equally fresh legacy `SplittingState` slot (append
  slots get that from `add_particle`) so lockstep bookkeeping stays coherent
  even though the two split policies must never be active together.
"""
function _rsplit_emit_child!(pfield, rs::ResolutionSplitState, slot::Int,
                             x, y, z, gx, gy, gz, sigma_c, circ, is_stat::Bool)
    R = eltype(pfield.particles)
    vol_c = R(4)/R(3) * pi * sigma_c^3
    if slot == 0
        add_particle(pfield, (x, y, z), (gx, gy, gz), sigma_c;
                     vol=vol_c, circulation=circ, C=zero(R), static=is_stat)
        # add_particle's rs hook already set sigma_0 = sigma_c and zeroed the
        # accumulators; nothing more to do.
    else
        set_X(pfield, slot, (x, y, z))
        set_Gamma(pfield, slot, (gx, gy, gz))
        set_sigma(pfield, slot, sigma_c)
        set_vol(pfield, slot, vol_c)
        set_circulation(pfield, slot, circ)
        zeroR = zero(R)
        set_U(pfield, slot, zeroR)
        set_vorticity(pfield, slot, zeroR)
        set_J(pfield, slot, zeroR)
        set_PSE(pfield, slot, zeroR)
        set_M(pfield, slot, zeroR)
        set_C(pfield, slot, zeroR)
        set_SFS(pfield, slot, zeroR)
        set_U_prev(pfield, slot, zeroR)
        set_static(pfield, slot, Float64(is_stat))
        _rsplit_reset_slot!(rs, slot, sigma_c)
        # keep the (inactive) legacy SplittingState slot equally fresh
        st = pfield.splitting_state
        st.sigma_0[slot] = R(sigma_c)
        st.H_chi[slot] = zeroR
        st.hold_counter[slot] = 0
        st.cooldown_counter[slot] = 0
        st.dsigma2_visc[slot] = zeroR
        st.dsigma2_rvpm[slot] = zeroR
    end
    return nothing
end

"Snapshot the parent quantities a kernel needs before child 1 overwrites slot i."
@inline function _rsplit_parent(pfield, i::Int)
    X = get_X(pfield, i); G = get_Gamma(pfield, i)
    return (X[1], X[2], X[3], G[1], G[2], G[3], get_sigma(pfield, i)[],
            get_circulation(pfield, i)[], get_static(pfield, i))
end

################################################################################
# KERNELS
################################################################################
"""
    _split_viscous_tetra4!(pfield, rs, i, offset_ratio)

Viscous mechanism (spec §3a): isotropic 4-child split of particle `i` on the
vertices of a randomly oriented regular tetrahedron centered on the parent.
`σ_c = σ_p·4^(-1/3)` (volume rule = merge inverse), vertex radius
`a = offset_ratio·σ_p` (default 1.3503 = per-axis second-moment match,
spec §3a; edge = a·√(8/3)), all children `Γ_p/4 ∥ Γ_p`. Child 1 overwrites
slot `i`; 3 children appended (caller guarantees headroom).
"""
function _split_viscous_tetra4!(pfield, rs::ResolutionSplitState, i::Int,
                                offset_ratio)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    sigma_c = sigma_p * 4.0^(-1/3)
    a = offset_ratio * sigma_p
    Q = _random_rotation()
    cgx, cgy, cgz = gx/4, gy/4, gz/4
    # canonical unit tetrahedron vertices: (±1,±1,±1)-family / √3
    s3 = sqrt(3)
    for (k, v) in enumerate(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
        d = Q * SVector{3,Float64}(v[1]/s3, v[2]/s3, v[3]/s3)
        _rsplit_emit_child!(pfield, rs, k == 1 ? i : 0,
                            x0 + a*d[1], y0 + a*d[2], z0 + a*d[3],
                            cgx, cgy, cgz, sigma_c, circ, is_stat)
    end
    return nothing
end

"""
    _split_compress_tri3!(pfield, rs, i, ex, ey, ez, offset_ratio)

Stretch mechanism, COMPRESSION regime (negative stretch: the tube shortens
and fattens — re-discretize the cross-section; doc §3b geometry): 3 children
on the vertices of an equilateral triangle in the plane with unit normal
`(ex,ey,ez)` (averaged stretch axis or Γ̂ fallback), centered on the parent,
random in-plane orientation. `σ_c = σ_p/√3` (W5 mass-per-length rule),
ring radius `a = offset_ratio·σ_p` ⇒ triangle side `a√3`, i.e.
`spacing/σ_c = 3·offset_ratio` (default 0.6 → spacing 1.8 σ_c; the
moment-match value 1.155 is available by knob [D2]). Children `Γ_p/3 ∥ Γ_p`
— parallel to the parent Γ, NOT forced along the axis. Child 1 overwrites
slot `i`; 2 appended.
"""
function _split_compress_tri3!(pfield, rs::ResolutionSplitState, i::Int,
                               ex, ey, ez, offset_ratio)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    sigma_c = sigma_p / sqrt(3)
    a = offset_ratio * sigma_p
    cgx, cgy, cgz = gx/3, gy/3, gz/3
    # stable in-plane basis: cross the normal with its least-aligned
    # coordinate axis, then complete the right-handed triad
    ax_, ay_, az_ = abs(ex), abs(ey), abs(ez)
    ux, uy, uz = ax_ <= ay_ && ax_ <= az_ ? (1.0, 0.0, 0.0) :
                 (ay_ <= az_ ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0))
    t1x = ey*uz - ez*uy; t1y = ez*ux - ex*uz; t1z = ex*uy - ey*ux
    t1n = sqrt(t1x*t1x + t1y*t1y + t1z*t1z)
    t1x /= t1n; t1y /= t1n; t1z /= t1n
    t2x = ey*t1z - ez*t1y; t2y = ez*t1x - ex*t1z; t2z = ex*t1y - ey*t1x
    psi0 = _inplane_angle()
    for k in 0:2
        psi = psi0 + k*2pi/3
        c, s = cos(psi), sin(psi)
        dx = a*(c*t1x + s*t2x); dy = a*(c*t1y + s*t2y); dz = a*(c*t1z + s*t2z)
        _rsplit_emit_child!(pfield, rs, k == 0 ? i : 0,
                            x0 + dx, y0 + dy, z0 + dz,
                            cgx, cgy, cgz, sigma_c, circ, is_stat)
    end
    return nothing
end

"""
    _split_elongate_pair2!(pfield, rs, i, ex, ey, ez, offset_ratio)

Stretch mechanism, ELONGATION regime (positive stretch: the tube lengthens
and thins — the cross-section is fine, re-discretize the LENGTH; Ryan
2026-09-07 third ruling, superseding doc §3b's routing of shrink events to
the triangle): 2 children on the line through the parent along `(ex,ey,ez)`
(averaged stretch axis or Γ̂ fallback) at `±b`, `b = offset_ratio·σ_p`
(default 0.5 ⇒ spacing 1.0 σ_p — children stay well-overlapped), each child
`Γ_p/2 ∥ Γ_p`, and **`σ_c = σ_p`** — core radius unchanged, each child
represents half the segment length. Total Γ, centroid, linear impulse exact;
angular impulse exact by the ± symmetry. Floor consistency: children inherit
`sigma_0 = σ_c = σ_p` (≤ floor at a floor event) so the floor trigger's
`sigma_0 > floor` guard self-disarms for them (exposure resets to 0 and must
re-accumulate). No random draw — the axis is state, the offsets symmetric.
"""
function _split_elongate_pair2!(pfield, rs::ResolutionSplitState, i::Int,
                                ex, ey, ez, offset_ratio)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    b = offset_ratio * sigma_p
    cgx, cgy, cgz = gx/2, gy/2, gz/2
    _rsplit_emit_child!(pfield, rs, i,
                        x0 - b*ex, y0 - b*ey, z0 - b*ez,
                        cgx, cgy, cgz, sigma_p, circ, is_stat)
    _rsplit_emit_child!(pfield, rs, 0,
                        x0 + b*ex, y0 + b*ey, z0 + b*ez,
                        cgx, cgy, cgz, sigma_p, circ, is_stat)
    return nothing
end
