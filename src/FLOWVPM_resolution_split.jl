#=##############################################################################
# DESCRIPTION
    Resolution-preserving particle splitting (BRAINSTORM 026, Phase 2).

    THE particle-splitting system of FLOWVPM (the legacy experimental
    splitting path was removed in its favor, Ryan authorization 2026-09-08).
    Entry point is `split_particles!(pfield, ::ResolutionSplitOpts)`; wire
    into a time march via `run_vpm!`'s `split_every`/`split_opts` kwargs.

    Two mechanisms (both OFF by default):
    * Viscous (Mechanism A, spec §3a): isotropic growth from core spreading —
      4-child tetrahedron, `σ_c = σ_p·4^(-1/3)`.
    * Stretch (Mechanism B, spec §3b + Ryan 2026-09-07 two-regime ruling):
      - compression regime (grow side): 3-child triangle in the plane normal
        to the averaged stretch axis, `σ_c = σ_p/√3` (W5 mass-per-length rule);
      - elongation regime (shrink side): in-line children along the stretch
        axis with `σ_c = σ_p` (re-discretizes length; cross-section
        unchanged). With `elongate_overlap` set (the driver default), child
        count and spacing are ADAPTIVE — m = clamp(round(λ_att·σ₀/σ_c), 2,
        elongate_m_max) children tile the accumulated stretch at the target
        overlap (theory doc §3, Ryan 2026-09-09); with `elongate_overlap =
        NaN` the legacy fixed 2-child kernel at `elongate_offset_ratio`
        spacing is used.

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
flapping accumulates instead of cancelling. `dvisc`/`drvpm` are per-mechanism
ATTEMPTED Δσ² accumulators (Ryan 2026-09-08 fractional-gating ruling): each
records the pre-clamp σ² change its mechanism tried to apply since the
particle's last split, so a particle pinned at the sigma_guard floor/ceiling
keeps accruing credit and its trigger still fires. `drvpm` is a signed net
(compression + and elongation − cancel — a particle that compresses then
relaxes back never splits); `dvisc` is nonnegative by physics. All of
axis/weight/dvisc/drvpm reset on split and on merge (the merged representative
is a new entity).

Anti-refire WITHOUT cooldown — every trigger reads only `sigma_0` and the
accumulators, and `_rsplit_reset_slot!` restamps `sigma_0 = σ_c` and zeroes
the accumulators on each child (and on merged representatives), so a fresh
child — even one still pinned at a clamp (pair2's `σ_c = σ_p` included) —
re-arms only after accruing fresh attempted deformation.
"""
mutable struct ResolutionSplitState{R, TV<:AbstractVector{R}, TM<:AbstractMatrix{R}}
    # per-particle arrays, lockstep with pfield.particles columns — that's ALL
    # of it: no cooldown, no scratch (the main pass is a single serial loop).
    # Backing storage matches the owning pfield's array type (host Vector/
    # Matrix on Array-backed fields, device arrays on GPU-backed fields) so
    # the broadcast integrator twins can accumulate in place on either
    # backend; the scalar hooks below are only ever called from host-side
    # code paths (add/remove/split/merge run on Array-backed fields or the
    # host mirror of a device-backed wake).
    sigma_0::TV            # σ reference at creation / last split / last merge
    axis::TM               # (3, maxp) sign-aligned running Σ dt·S since last split
    weight::TV             # Σ dt·|S| since last split (coherence denominator)
    dvisc::TV              # Σ attempted Δσ² from viscous spreading (≥ 0)
    drvpm::TV              # Σ attempted Δσ² from rVPM area evolution (signed net)
end

ResolutionSplitState{R}(maxparticles::Int) where {R} = ResolutionSplitState(
    zeros(R, maxparticles),
    zeros(R, 3, maxparticles),
    zeros(R, maxparticles),
    zeros(R, maxparticles),
    zeros(R, maxparticles),
)

"Allocate a ResolutionSplitState whose arrays match `template`'s storage type
(e.g. CuArray for a device-backed particle matrix), zero-initialized."
function ResolutionSplitState(template::AbstractMatrix{R}, maxparticles::Int) where {R}
    v() = fill!(similar(template, R, maxparticles), zero(R))
    m() = fill!(similar(template, R, (3, maxparticles)), zero(R))
    return ResolutionSplitState(v(), m(), v(), v(), v())
end

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
        if pfield.particles isa Array
            rs = ResolutionSplitState{R}(pfield.maxparticles)
            for i in 1:get_np(pfield)
                rs.sigma_0[i] = get_sigma(pfield, i)[]
            end
        else
            # device-backed field: allocate matching device arrays and seed
            # sigma_0 by broadcast (no scalar indexing on device storage)
            rs = ResolutionSplitState(pfield.particles, pfield.maxparticles)
            np = get_np(pfield)
            if np > 0
                view(rs.sigma_0, 1:np) .= view(pfield.particles, SIGMA_INDEX, 1:np)
            end
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

Flat options for `split_particles!(pfield, opts::ResolutionSplitOpts)`.
Triggers are PER-MECHANISM growth fractions relative to the particle's own
`sigma_0` (Ryan 2026-09-08 ruling), each independently disabled by `NaN` (the
default): mechanism k fires when its own attempted Δσ² accumulator crosses its
fraction, i.e. `sqrt(σ₀² + Δσ²_k)/σ₀` leaves `[1 − f_elong, 1 + f_k]` on the
corresponding side. The trigger IS the mechanism — there is no separate
routing step. A firing mechanism that is not enabled skips + counts, never
reroutes.

`sigma_min`/`sigma_max` are CLAMPS, not triggers: emitted children are clamped
into `[sigma_min, sigma_max]` (NaN = unbounded on that side). In-step σ is
clamped separately by the integrator's `sigma_guard` (052c floor/ceil — a
permanent partner of splitting, no longer a band-aid); accumulation is
pre-clamp, so clamped particles keep firing.

Naming: `viscous_*` is the isotropic viscous mechanism (spec §3a); the stretch
mechanism's two regimes are `compress_*` (net rVPM compression → 3-child
triangle) and `elongate_*` (net rVPM elongation → 2 in-line children). Both
stretch regimes share `enable_stretch_split`.
"""
struct ResolutionSplitOpts{R}
    # per-mechanism fractional triggers — NaN disables each independently
    f_visc::R                 # viscous:  sqrt(σ₀² + dvisc)/σ₀ > 1 + f_visc
    f_comp::R                 # compress: sqrt(σ₀² + drvpm)/σ₀ > 1 + f_comp (drvpm > 0)
    f_elong::R                # elongate: sqrt(max(σ₀² + drvpm, 0))/σ₀ < 1 − f_elong
    # σ bounds — clamps on emitted children, NaN = unbounded
    sigma_min::R
    sigma_max::R
    # mechanisms
    enable_viscous_split::Bool # §3a: isotropic 4-child tetrahedron
    enable_stretch_split::Bool # §3b + 2026-09-07 two-regime ruling
    # child placement — child offset from parent center, as a ratio of parent σ
    viscous_offset_ratio::R   # tetra vertex radius / σ_p (default: second-moment match)
    compress_offset_ratio::R  # tri3 ring radius / σ_p (compression; default spacing 1.8 σ_c)
    elongate_offset_ratio::R  # LEGACY pair2 half-spacing / σ_p (used only when elongate_overlap is NaN)
    # adaptive in-line elongation (theory doc §3): child count m = clamp(
    # round(λ_att·σ₀/σ_c), 2, elongate_m_max) tiles the stretched length
    # λ_att·σ₀/elongate_overlap at spacing ≈ σ_c/elongate_overlap
    elongate_overlap::R       # target child overlap Φ_t = σ_c/spacing; NaN → legacy fixed pair2
    elongate_m_max::Int       # cap on children per elongation split (≥ 2)
    # split-plane orientation for the stretch mechanism
    use_stretch_axis::Bool    # false → split plane ⟂ Γ̂ always (comparison arm)
    axis_coherence_min::R     # below this coherence → fall back to Γ̂ (still splits)
end

function ResolutionSplitOpts{R}(;
            f_visc                 = R(NaN),
            f_comp                 = R(NaN),
            f_elong                = R(NaN),
            sigma_min              = R(NaN),
            sigma_max              = R(NaN),
            enable_viscous_split::Bool = false,
            enable_stretch_split::Bool = false,
            viscous_offset_ratio   = R(1.3503),
            compress_offset_ratio  = R(0.6),
            elongate_offset_ratio  = R(0.5),
            elongate_overlap       = R(NaN),
            elongate_m_max::Int    = 4,
            use_stretch_axis::Bool = true,
            axis_coherence_min     = R(0.5),
        ) where {R}
    isnan(f_visc) || f_visc > 0 || throw(ArgumentError(
        "f_visc must be positive (or NaN to disable), got $(f_visc)"))
    isnan(f_comp) || f_comp > 0 || throw(ArgumentError(
        "f_comp must be positive (or NaN to disable), got $(f_comp)"))
    isnan(f_elong) || 0 < f_elong < 1 || throw(ArgumentError(
        "f_elong must be in (0, 1) (or NaN to disable), got $(f_elong)"))
    isnan(elongate_overlap) || elongate_overlap > 0 || throw(ArgumentError(
        "elongate_overlap must be positive (or NaN for the legacy fixed " *
        "pair2 kernel), got $(elongate_overlap)"))
    elongate_m_max >= 2 || throw(ArgumentError(
        "elongate_m_max must be at least 2, got $(elongate_m_max)"))
    return ResolutionSplitOpts{R}(R(f_visc), R(f_comp), R(f_elong),
                R(sigma_min), R(sigma_max),
                enable_viscous_split, enable_stretch_split,
                R(viscous_offset_ratio), R(compress_offset_ratio),
                R(elongate_offset_ratio),
                R(elongate_overlap), elongate_m_max,
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
    rs.dvisc[i] = rs.dvisc[np]
    rs.drvpm[i] = rs.drvpm[np]
    return nothing
end

"Zero the vacated tail slot `np` after a removal."
@inline function _rsplit_zero!(rs::ResolutionSplitState, np::Int)
    rs.sigma_0[np] = 0
    rs.axis[1, np] = 0; rs.axis[2, np] = 0; rs.axis[3, np] = 0
    rs.weight[np] = 0
    rs.dvisc[np] = 0
    rs.drvpm[np] = 0
    return nothing
end

################################################################################
# ACCUMULATION (integrator-inline; caller guards on `rs === nothing`)
################################################################################
"""
    _rsplit_accumulate!(rs, i, dt, sx, sy, sz)

Accumulate one stretching sample `S = (sx,sy,sz)` for particle `i` over time
weight `dt`: sign-invariant axis average and coherence weight (split DIRECTION
state only — trigger state lives in the Δσ² accumulators), called where S is
already in hand inside the integrator (~15 flops).
"""
@inline function _rsplit_accumulate!(rs::ResolutionSplitState, i::Int, dt,
                                     sx, sy, sz)
    n = sqrt(sx*sx + sy*sy + sz*sz)
    n > 0 || return nothing
    ax = rs.axis
    d = ax[1, i]*sx + ax[2, i]*sy + ax[3, i]*sz
    sgn = d < 0 ? -one(n) : one(n)
    ax[1, i] += sgn*dt*sx; ax[2, i] += sgn*dt*sy; ax[3, i] += sgn*dt*sz
    rs.weight[i] += dt*n
    return nothing
end

"""
    _rsplit_accumulate_dsigma2!(rs, i, dv, dr)

Accumulate the ATTEMPTED (pre-clamp) Δσ² attribution for particle `i`: `dv`
from viscous spreading, `dr` from the rVPM area evolution (signed: compression
+, elongation −) — written inline at each site where the integrator/viscous
scheme computes a σ update, BEFORE any sigma_guard floor/ceil clamp, so
clamp-pinned particles keep accruing and their per-mechanism fractional
triggers still fire (Ryan 2026-09-08).
"""
@inline function _rsplit_accumulate_dsigma2!(rs::ResolutionSplitState, i::Int, dv, dr)
    rs.dvisc[i] += dv
    rs.drvpm[i] += dr
    return nothing
end

################################################################################
# BROADCAST ACCUMULATION (device-compatible twins of the two hooks above,
# called from the broadcast integrator paths; work on plain Arrays too)
################################################################################
"""
    _rsplit_accumulate_broadcast!(rs, dt, S1, S2, S3, mask)

Broadcast twin of `_rsplit_accumulate!`: accumulate one stretching sample per
particle from full-length row vectors `S1,S2,S3` (length maxparticles, dead
tail columns hold S = 0 and are masked out by the |S| > 0 gate). `mask` is the
active-particle mask (1 active, 0 static). Allocates small temporaries — the
broadcast paths allocate by design (see `_euler_broadcast_reformulated!`).
"""
function _rsplit_accumulate_broadcast!(rs::ResolutionSplitState, dt, S1, S2, S3, mask)
    ax1 = view(rs.axis, 1, :); ax2 = view(rs.axis, 2, :); ax3 = view(rs.axis, 3, :)
    R = eltype(rs.weight)
    n = sqrt.(S1.^2 .+ S2.^2 .+ S3.^2)
    # gate: active AND |S| > 0 (also excludes non-finite dead-tail lanes).
    # ifelse (not a mask product) throughout so a NaN in an excluded lane
    # cannot leak through as NaN*0.
    m = (mask .> 0) .& (n .> 0)
    # sign-invariant axis add: flip the sample toward the running axis
    d = ax1 .* S1 .+ ax2 .* S2 .+ ax3 .* S3
    sgndt = ifelse.(d .< 0, -R(dt), R(dt))
    ax1 .+= ifelse.(m, sgndt .* S1, zero(R))
    ax2 .+= ifelse.(m, sgndt .* S2, zero(R))
    ax3 .+= ifelse.(m, sgndt .* S3, zero(R))
    rs.weight .+= ifelse.(m, R(dt) .* n, zero(R))
    return nothing
end

"""
    _rsplit_accumulate_dsigma2_broadcast!(rs, dv, dr, mask)

Broadcast twin of `_rsplit_accumulate_dsigma2!`: `dv`/`dr` are per-particle
attempted Δσ² attributions (scalars or full-length row vectors), masked by
the active mask.
"""
function _rsplit_accumulate_dsigma2_broadcast!(rs::ResolutionSplitState, dv, dr, mask)
    R = eltype(rs.dvisc)
    # ifelse (not a mask product) so NaN in an excluded lane cannot leak
    rs.dvisc .+= ifelse.(mask .> 0, dv, zero(R))
    rs.drvpm .+= ifelse.(mask .> 0, dr, zero(R))
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
"NaN-tolerant σ clamp: NaN bound = unbounded on that side."
@inline function _rsplit_clamp_sigma(s, smin, smax)
    isnan(smax) || (s = min(s, smax))
    isnan(smin) || (s = max(s, smin))
    return s
end

"""
    _rsplit_emit_child!(pfield, rs, slot, x, y, z, gx, gy, gz, sigma_c, circ,
                        is_stat, opts)

Emit one split child. `slot > 0` overwrites that existing column in place
(child 1 reuses the parent's slot); `slot == 0` appends via `add_particle`
(caller guarantees headroom). Either way the child gets:

* `σ_c` clamped into `[opts.sigma_min, opts.sigma_max]` (NaN = unbounded) —
  the σ bounds are clamp operations, never triggers (Ryan 2026-09-08);
* `vol = (4/3)π σ_c³` from the CLAMPED σ_c — W5 σ³-consistent hygiene rule,
  NOT `vol_p/m`. This
  intentionally breaks merge inversion (merging's `σ = cbrt(Σσ³)` no longer
  reproduces the parent) and, for the pair2 kernel's `σ_c = σ_p`, doubles the
  σ-implied total volume — accepted, `vol` feeds no dynamics.
* zeroed U, vorticity, J, PSE, M, C, SFS, U_prev — maintenance runs
  post-convection and UJ re-evaluates next step; zeroing M also clears
  euler_exp's M[9] Zeff stash, which is correct for a fresh child.
* fresh ResolutionSplitState (`sigma_0 = σ_c`, accumulators zero).
"""
function _rsplit_emit_child!(pfield, rs::ResolutionSplitState, slot::Int,
                             x, y, z, gx, gy, gz, sigma_c, circ, is_stat::Bool,
                             opts::ResolutionSplitOpts)
    R = eltype(pfield.particles)
    sigma_c = _rsplit_clamp_sigma(sigma_c, opts.sigma_min, opts.sigma_max)
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
    _split_viscous_tetra4!(pfield, rs, i, opts)

Viscous mechanism (spec §3a): isotropic 4-child split of particle `i` on the
vertices of a randomly oriented regular tetrahedron centered on the parent.
`σ_c = σ_p·4^(-1/3)` (volume rule = merge inverse), vertex radius
`a = offset_ratio·σ_p` (default 1.3503 = per-axis second-moment match,
spec §3a; edge = a·√(8/3)), all children `Γ_p/4 ∥ Γ_p`. Child 1 overwrites
slot `i`; 3 children appended (caller guarantees headroom).
"""
function _split_viscous_tetra4!(pfield, rs::ResolutionSplitState, i::Int,
                                opts::ResolutionSplitOpts)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    sigma_c = sigma_p * 4.0^(-1/3)
    a = opts.viscous_offset_ratio * sigma_p
    Q = _random_rotation()
    cgx, cgy, cgz = gx/4, gy/4, gz/4
    # lengthwise bundle division: each of the 4 filaments carries 1/4 of the
    # parent's vorticity flux (circulation is diagnostic-only today)
    circ = circ/4
    # canonical unit tetrahedron vertices: (±1,±1,±1)-family / √3
    s3 = sqrt(3)
    for (k, v) in enumerate(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
        d = Q * SVector{3,Float64}(v[1]/s3, v[2]/s3, v[3]/s3)
        _rsplit_emit_child!(pfield, rs, k == 1 ? i : 0,
                            x0 + a*d[1], y0 + a*d[2], z0 + a*d[3],
                            cgx, cgy, cgz, sigma_c, circ, is_stat, opts)
    end
    return nothing
end

"""
    _split_compress_tri3!(pfield, rs, i, ex, ey, ez, opts)

Stretch mechanism, COMPRESSION regime (negative stretch: the tube shortens
and fattens — re-discretize the cross-section; doc §3b geometry): 3 children
on the vertices of an equilateral triangle in the plane with unit normal
`(ex,ey,ez)` (averaged stretch axis or Γ̂ fallback), centered on the parent,
random in-plane orientation. `σ_c = σ_p/√3` (W5 mass-per-length rule),
ring radius `a = offset_ratio·σ_p` ⇒ triangle side `a√3`, i.e.
`spacing/σ_c = 3·offset_ratio` (default 0.6 → spacing 1.8 σ_c; the
moment-match value 1.155 is available by knob [D2]). Children `Γ_p/3 ∥ Γ_p`
— parallel to the parent Γ, NOT forced along the axis — each carrying
`circulation/3` (lengthwise bundle division splits the vorticity flux).

Child count is FIXED at m = 3 (Ryan 2026-09-09): re-discretizing the
fattened cross-section into birth-sized cores wants `m ≈ (1+f_comp)²`
filaments, so the triangle is matched to `f_comp ≈ √3−1 ≈ 0.73`; m = 2
would impose an artificial transverse anisotropy on an axisymmetric
fattening, so the triangle (smallest in-plane-isotropic arrangement) is
kept even at smaller f_comp. Larger accumulated compression could be
matched by higher m with a more complicated child shape (ring + center,
two rings) — deliberately not pursued yet (theory doc §5). Child 1
overwrites slot `i`; 2 appended.
"""
function _split_compress_tri3!(pfield, rs::ResolutionSplitState, i::Int,
                               ex, ey, ez, opts::ResolutionSplitOpts)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    sigma_c = sigma_p / sqrt(3)
    a = opts.compress_offset_ratio * sigma_p
    cgx, cgy, cgz = gx/3, gy/3, gz/3
    # lengthwise bundle division: each of the 3 filaments carries 1/3 of the
    # parent's vorticity flux (circulation is diagnostic-only today); the
    # crosswise elongation cut, by contrast, keeps circulation unchanged
    circ = circ/3
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
                            cgx, cgy, cgz, sigma_c, circ, is_stat, opts)
    end
    return nothing
end

"""
    _split_elongate_pair2!(pfield, rs, i, ex, ey, ez, opts)

Stretch mechanism, ELONGATION regime — LEGACY fixed 2-child kernel, used
only when `opts.elongate_overlap` is NaN (the adaptive
`_split_elongate_line!` replaced it as the default, Ryan 2026-09-09; kept
as the comparison arm because its fixed spacing is dimensionally arbitrary —
over/under-coverage ratio `2Φ(1−f_elong)³`, see theory doc §3): 2 children
on the line through the parent along `(ex,ey,ez)`
(averaged stretch axis or Γ̂ fallback) at `±b`, `b = offset_ratio·σ_p`
(default 0.5 ⇒ spacing 1.0 σ_p — children stay well-overlapped), each child
`Γ_p/2 ∥ Γ_p`, and **`σ_c = σ_p`** — core radius unchanged, each child
represents half the segment length. Total Γ, centroid, linear impulse exact;
angular impulse exact by the ± symmetry. Children inherit `sigma_0 = σ_c =
σ_p` with zeroed accumulators, so even a still-floor-pinned child re-arms
only after accruing fresh attempted shrink (no structural guard needed). No
random draw — the axis is state, the offsets symmetric.
"""
function _split_elongate_pair2!(pfield, rs::ResolutionSplitState, i::Int,
                                ex, ey, ez, opts::ResolutionSplitOpts)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    b = opts.elongate_offset_ratio * sigma_p
    cgx, cgy, cgz = gx/2, gy/2, gz/2
    _rsplit_emit_child!(pfield, rs, i,
                        x0 - b*ex, y0 - b*ey, z0 - b*ez,
                        cgx, cgy, cgz, sigma_p, circ, is_stat, opts)
    _rsplit_emit_child!(pfield, rs, 0,
                        x0 + b*ex, y0 + b*ey, z0 + b*ez,
                        cgx, cgy, cgz, sigma_p, circ, is_stat, opts)
    return nothing
end

"""
    _elongate_plan(rs, i, opts, sigma_p) -> (m, s)

Child count and spacing for the ADAPTIVE in-line elongation kernel (theory
doc §3; requires `opts.elongate_overlap` finite). The rVPM channel conserves
σ²·L exactly, so the attempted accumulator gives the represented length
stretch directly, `λ_att = σ₀²/(σ₀² + drvpm)` (`drvpm < 0` here). Children of
core `σ_c = σ_p` tile the stretched length `λ_att·ℓ₀` (birth tile
`ℓ₀ = σ₀/Φ_t`) at spacing `≈ σ_c/Φ_t`:

    m = clamp(round(λ_att·σ₀/σ_p), 2, elongate_m_max),   s = λ_att·ℓ₀/m.

Unclamped this is `m* = (1−f_elong)⁻³` with `m·σ_c³ = σ₀³` (the volume
rule / merge inverse), is self-consistent without per-particle L₀ state,
and composes exactly across repeated splits (up to integer rounding).
Saturation: when `m_ideal > elongate_m_max` (or `σ₀² + drvpm ≤ 0`,
attempted total collapse), spacing falls back to the target `σ_p/Φ_t` —
children retain overlap and the event under-tiles rather than spraying
`m_max` children over the full stretched length.
"""
@inline function _elongate_plan(rs::ResolutionSplitState, i::Int,
                                opts::ResolutionSplitOpts, sigma_p)
    s0 = rs.sigma_0[i]
    s02 = s0*s0
    den = s02 + rs.drvpm[i]
    s_target = sigma_p / opts.elongate_overlap
    den > 0 || return (opts.elongate_m_max, s_target)
    lam = s02 / den
    m_ideal = lam * s0 / sigma_p
    m_ideal > opts.elongate_m_max && return (opts.elongate_m_max, s_target)
    m = max(2, round(Int, m_ideal))
    s = lam * (s0 / opts.elongate_overlap) / m
    return (m, s)
end

"""
    _split_elongate_line!(pfield, rs, i, ex, ey, ez, m, s, opts)

Stretch mechanism, ELONGATION regime, adaptive in-line kernel (theory doc
§3): `m` children on the line through the parent along `(ex,ey,ez)` at
spacing `s` (from `_elongate_plan`), centroid-symmetric offsets
`(k − (m+1)/2)·s`, each child `Γ_p/m ∥ Γ_p`, `σ_c = σ_p` (cross-section
untouched), `circulation` unchanged (a crosswise cut preserves each
segment's circulation). Total Γ, centroid, linear impulse exact for any m
(symmetric offsets cancel pairwise; odd m leaves the middle child at the
parent position); angular impulse error is quadratic in the offsets,
`|ΔA| ≤ a²|Γ|/3` with `a = (m−1)s/2` the outermost offset, vanishing when
the split axis is parallel to Γ (t2's bound, same as the other kernels).
Child 1 overwrites slot `i`; `m − 1` appended (caller guarantees headroom).
"""
function _split_elongate_line!(pfield, rs::ResolutionSplitState, i::Int,
                               ex, ey, ez, m::Int, s,
                               opts::ResolutionSplitOpts)
    x0, y0, z0, gx, gy, gz, sigma_p, circ, is_stat = _rsplit_parent(pfield, i)
    cgx, cgy, cgz = gx/m, gy/m, gz/m
    half = (m + 1) / 2
    for k in 1:m
        off = (k - half) * s
        _rsplit_emit_child!(pfield, rs, k == 1 ? i : 0,
                            x0 + off*ex, y0 + off*ey, z0 + off*ez,
                            cgx, cgy, cgz, sigma_p, circ, is_stat, opts)
    end
    return nothing
end

################################################################################
# TRIGGER CHECK + ROUTING + MAIN PASS
################################################################################
"""
    _rsplit_check(opts, rs, pfield, i) -> :none | :viscous | :compress | :elongate

Per-mechanism fractional trigger check (Ryan 2026-09-08 ruling) — the trigger
IS the mechanism, so this returns the kernel to route to directly. With
`σ₀ = sigma_0[i]` and the attempted per-mechanism accumulators:

* `:viscous`  — `dvisc > ((1 + f_visc)² − 1)·σ₀²`
* `:compress` — `drvpm > ((1 + f_comp)² − 1)·σ₀²`
* `:elongate` — `drvpm < ((1 − f_elong)² − 1)·σ₀²` (negative bound; a net
  attempted collapse past `σ₀² + drvpm ≤ 0` fires for any armed f_elong)

Every fraction is NaN-disabled (NaN comparisons are false). All comparisons
are in Δσ² space (no sqrt on the hot path). `:compress`/`:elongate` are
mutually exclusive (one signed accumulator); when `:viscous` fires together
with a stretch regime, the larger ratio excess wins, with `:viscous` winning
ties and beating `:elongate` (grow-wins convention, unchanged from the
absolute-threshold design). Realized σ and the sigma_guard clamps never enter
the check — a clamp-pinned particle keeps firing (accumulation is pre-clamp).
"""
@inline function _rsplit_check(opts::ResolutionSplitOpts, rs::ResolutionSplitState,
                               pfield, i::Int)
    s0 = rs.sigma_0[i]
    s0 > 0 || return :none
    s02 = s0*s0
    dvisc = rs.dvisc[i]; drvpm = rs.drvpm[i]
    fires_visc = dvisc > ((1 + opts.f_visc)^2 - 1)*s02
    fires_comp = drvpm > ((1 + opts.f_comp)^2 - 1)*s02
    fires_elong = drvpm < ((1 - opts.f_elong)^2 - 1)*s02
    if fires_visc
        # rare double fire: compare ratio excesses (sqrt off the hot path)
        if fires_comp
            excess_v = sqrt(1 + dvisc/s02) - (1 + opts.f_visc)
            excess_c = sqrt(1 + drvpm/s02) - (1 + opts.f_comp)
            return excess_v >= excess_c ? :viscous : :compress
        end
        return :viscous   # grow wins over :elongate
    end
    fires_comp && return :compress
    fires_elong && return :elongate
    return :none
end

"""
    split_particles!(pfield, opts::ResolutionSplitOpts; verbose=false, dt=nothing)

Resolution-preserving splitting pass (BRAINSTORM 026 Phase 2). This is THE
particle-splitting system of FLOWVPM (the legacy experimental splitting
path was removed in its favor, Ryan authorization 2026-09-08). Use standalone,
or wire into a time march via `run_vpm!`'s `split_every`/`split_opts` kwargs
(applied after merging; merged representatives get a fresh state slot).

One serial loop over the pre-pass particles (`1:np0`), no scratch, no
ranking: appending is safe while iterating because children land either in
slot `i` (already visited) or in slots `> np0` (never visited this call).
`_rsplit_check` returns the firing mechanism directly (per-mechanism
fractional triggers — no separate routing step): `:viscous` → tetra4,
`:compress` → tri3, `:elongate` → pair2. A firing-but-disabled mechanism
skips and counts — never reroutes (W6 keeps the viscous mechanism gated by
evidence). Emitted children are clamped into
`[opts.sigma_min, opts.sigma_max]`. The only hard guard is maxparticles
headroom (a correctness requirement, not a feature).

`dt` is accepted for the caller's policy seam but unused — all state
accumulation happens inline in the integrator, not here.

Returns `(; n_split_viscous, n_split_compress, n_split_elongate,
n_children_elongate, n_skipped_capacity, n_skipped_mech_disabled)`.
`n_children_elongate` totals the children emitted by elongation events (the
adaptive kernel emits a variable 2..`elongate_m_max` per event; the other
mechanisms stay fixed at 4 and 3 children).
"""
function split_particles!(pfield, opts::ResolutionSplitOpts;
                          verbose::Bool=false, dt=nothing)
    rs = enable_resolution_split!(pfield)
    n_split_viscous = 0; n_split_compress = 0; n_split_elongate = 0
    n_children_elongate = 0
    n_skipped_capacity = 0; n_skipped_mech_disabled = 0
    np0 = get_np(pfield)
    maxp = pfield.maxparticles
    for i in 1:np0
        mech = _rsplit_check(opts, rs, pfield, i)
        mech === :none && continue
        if mech === :elongate
            if !opts.enable_stretch_split
                n_skipped_mech_disabled += 1; continue
            end
            if isnan(opts.elongate_overlap)
                # legacy fixed pair2 (elongate_offset_ratio spacing)
                if get_np(pfield) + 1 > maxp
                    n_skipped_capacity += 1; break
                end
                ex, ey, ez = _rsplit_direction(rs, i, opts, pfield)
                _split_elongate_pair2!(pfield, rs, i, ex, ey, ez, opts)
                n_split_elongate += 1
                n_children_elongate += 2
            else
                # adaptive in-line kernel (theory doc §3): the plan must be
                # computed BEFORE the capacity check (m − 1 appended slots)
                m, s = _elongate_plan(rs, i, opts, get_sigma(pfield, i)[])
                if get_np(pfield) + (m - 1) > maxp
                    n_skipped_capacity += 1; break
                end
                ex, ey, ez = _rsplit_direction(rs, i, opts, pfield)
                _split_elongate_line!(pfield, rs, i, ex, ey, ez, m, s, opts)
                n_split_elongate += 1
                n_children_elongate += m
            end
        elseif mech === :viscous
            if !opts.enable_viscous_split
                n_skipped_mech_disabled += 1; continue
            end
            if get_np(pfield) + 3 > maxp
                n_skipped_capacity += 1; break
            end
            _split_viscous_tetra4!(pfield, rs, i, opts)
            n_split_viscous += 1
        else
            if !opts.enable_stretch_split
                n_skipped_mech_disabled += 1; continue
            end
            if get_np(pfield) + 2 > maxp
                n_skipped_capacity += 1; break
            end
            ex, ey, ez = _rsplit_direction(rs, i, opts, pfield)
            _split_compress_tri3!(pfield, rs, i, ex, ey, ez, opts)
            n_split_compress += 1
        end
    end
    if verbose && (n_split_viscous + n_split_compress + n_split_elongate +
                   n_skipped_capacity + n_skipped_mech_disabled) > 0
        println("split_particles! (resolution): viscous=$(n_split_viscous)"*
                " compress=$(n_split_compress) elongate=$(n_split_elongate)"*
                " (children=$(n_children_elongate))"*
                " | skipped: capacity=$(n_skipped_capacity)"*
                " mech_disabled=$(n_skipped_mech_disabled)")
    end
    return (; n_split_viscous, n_split_compress, n_split_elongate,
              n_children_elongate,
              n_skipped_capacity, n_skipped_mech_disabled)
end
