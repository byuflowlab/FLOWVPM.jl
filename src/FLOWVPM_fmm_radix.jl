#=##############################################################################
# DESCRIPTION
    Coupling of ParticleField to FastMultipole's device-resident radix FMM
    lifecycle (FastMultipole `matrix-ops` interface, FLOWVPM task 034).

    A `CuArray`-backed ParticleField drives the resident GPU lifecycle
    device-to-device: one `RadixFMMCache` is built lazily at first use, sized
    to `pfield.maxparticles`, and reused across every UJ evaluation (all RK3
    substeps and time steps) with zero per-step body host/device transfer
    (the task 023 counter contract: `body_uploads == 0`,
    `expansion_host_copies == 0`). A `Matrix`-backed ParticleField can use the
    same machinery through FastMultipole's transfer-based host-resident path
    (used by the CPU-side correctness tests); the production CPU path remains
    the legacy octree `fmm.fmm!` call in `UJ_fmm`, unchanged.

    This file is a no-op (defining only a loud error stub) when the installed
    FastMultipole does not provide the radix device interface — downstream
    CPU-only consumers on registry FastMultipole are unaffected.

# AUTHORSHIP
  * gpu-full branch, task 034 (2026-08-06)
=###############################################################################

# The device-resident radix interface shipped with FastMultipole task 032
# (branch matrix-ops). Registry releases without it simply skip this coupling.
const _FMM_HAS_RADIX = isdefined(fmm, :RadixFMMCache) &&
                       isdefined(fmm, :RegularizedVortex) &&
                       isdefined(fmm, :DeviceResident)

"""
    is_gaussianerf(kernel::Kernel)

True if `kernel` is the `gaussianerf` regularization (the only kernel
supported by the radix/GPU FMM path — FastMultipole's `RegularizedVortex`
nearfield implements exactly this regularization).
"""
is_gaussianerf(kernel::Kernel) = kernel.g_dgdr === g_dgdr_gauserf

if _FMM_HAS_RADIX

################################################################################
# FastMultipole system traits (radix path only; the legacy octree path never
# consults these, so CPU behavior through `UJ_fmm` is unchanged)
################################################################################

fmm.body_type(::ParticleField) = fmm.Point{fmm.Vortex}

# The residency trait selects the transfer-based host path for Matrix-backed
# fields and the zero-transfer device-resident path for CuArray-backed fields.
# The CPU/GPU switch is `pfield.particles isa Array`, per repo convention.
fmm.residency(pfield::ParticleField) =
    pfield.particles isa Array ? fmm.HostResident() : fmm.DeviceResident()

# Nearfield kernel: gaussianerf Biot-Savart with the raw smoothing radius
# sigma in packed extra-state row 8 (the same row the legacy
# `source_system_to_buffer!` writes). Row 4 stays the inflated
# rho_sigma*sigma MAC radius, distinct from sigma by convention.
# The kernel strategy is selectable through `RadixFMMSettings.direct_kernel`
# (task 035 tuning surface); the shipped default is `PartitionedVortex` at
# `rho_t = 4.789` (048 production selection, 2026-08-22).
function fmm.direct_kernel(pfield::ParticleField)
    is_gaussianerf(pfield.kernel) || error(
        "the radix/GPU FMM path supports only the `gaussianerf` kernel " *
        "(FLOWVPM default; sole CoreSpreading-compatible kernel). " *
        "Got a different `pfield.kernel`.")
    settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
    return _radix_direct_kernel(settings)
end

# Coupling default smoothing cutoff for the partitioned kernel (task 048
# production selection, user-approved 2026-08-22): the conservative
# Jacobian-per-pair cutoff. Together with expansion_order = 6 and derived
# near-shell geometry it passes the strict 5e-4 F64 delivered-E_str gate on
# the p018 210k production field (e_sfs 1.46e-4, +3% U/J cost vs the P4
# baseline; H200 job 13303399 sweep, `fm048_sweep_13303399.csv`). Supersedes
# the 035-cycle-3 velocity-RMS cutoff 3.668 (superseded value validated
# under the sum gate < 1e-3 at cube/wake n = 1e5/1e6). Applies only to
# :partitioned; :regularized/:twopass keep their constructor defaults.
const _PARTITIONED_RHO_T_DEFAULT = 4.789

"Resolve the FastMultipole nearfield direct-kernel functor from settings."
function _radix_direct_kernel(settings)
    sym = settings.direct_kernel
    rho_t = settings.rho_t
    rho_c = settings.rho_c
    if sym === :regularized
        rho_c === nothing || error(
            "RadixFMMSettings.rho_c applies only to direct_kernel=:twopass")
        return rho_t === nothing ? fmm.RegularizedVortex(; sigma_row=8) :
            fmm.RegularizedVortex(; sigma_row=8, rho_t)
    elseif sym === :partitioned
        rho_c === nothing || error(
            "RadixFMMSettings.rho_c applies only to direct_kernel=:twopass")
        return fmm.PartitionedVortex(; sigma_row=8,
            rho_t=something(rho_t, _PARTITIONED_RHO_T_DEFAULT))
    elseif sym === :twopass
        rt = something(rho_t, fmm.TwoPassVortex(; sigma_row=8).rho_t)
        rc = something(rho_c, 2.0)
        return fmm.TwoPassVortex(; sigma_row=8, rho_t=rt, rho_c=rc)
    end
    error("RadixFMMSettings.direct_kernel must be :regularized, :partitioned, " *
        "or :twopass; got $(repr(sym))")
end

"Resolve the (m2l_strategy, operator) pair from settings (task 035)."
function _radix_m2l_strategy(settings)
    sym = settings.m2l_strategy
    sym === :concat &&
        return (fmm.ConcatenatedFixedZM2L(), fmm.MaterializedYRotationM2L())
    sym === :dense &&
        return (fmm.DenseTranslationM2L(), fmm.MaterializedYRotationM2L())
    sym === :precomputed_y &&
        return (fmm.PrecomputedFactoredYM2L(), fmm.FactoredRotationM2L())
    error("RadixFMMSettings.m2l_strategy must be :concat, :dense, or " *
        ":precomputed_y; got $(repr(sym))")
end

################################################################################
# Settings and cache registry
################################################################################

"""
    RadixFMMSettings(; kwargs...)

Per-`ParticleField` overrides for the radix FMM coupling. All fields default
to automatic derivation:

- `expansion_order`: defaults to `4` (literature `P = 5`), the task-035
  cycle-3 measured winner: paired with the smaller `rho_t = 3.668` direct
  shell it is both faster (1.06-1.69x across cube/wake at n = 1e5/1e6) and
  more accurate than the previous literature-P4 defaults. Pass
  `expansion_order = nothing` to derive `pfield.fmm.p - 1` (literature
  `P = p`) as before.
- `ell`: radix tree depth; defaults to the deepest depth passing the
  margin-guarded near-set inequality
  `g_min(q)*h_leaf >= accuracy_margin*rho_t*sigma_max` for some supported
  leaf `q >= near_radius2`, capped by an occupancy heuristic of `~n^(1/3)`
  cells per side (task 035 cycle 1 joint (ell, q) rule). Passing `ell`
  explicitly uses `near_radius2` as the leaf radius verbatim.
- `near_radius2`: leaf direct-stencil ball radius squared (floor for the
  auto rule; 6 is the smallest shell measured gate-passing at the
  literature-P5 / `rho_t = 3.668` defaults — task 035 cycle 3. At the old
  literature-P4 / `rho_t = 4.252` settings the validated floor was 16;
  restore it when overriding those fields).
- `window_classes`: M2L window classes; `nothing` = 256 on device, framework
  default on host.
- `padding`: per-face padding fraction applied to derived domain bounds
  (construction and automatic `recenter!`).
- `bounds`: explicit `(x_min::SVector{3}, box_size)` domain box, where
  `box_size` may be a scalar (cubic) or a 3-vector (rectangular, FastMultipole
  task 037) and passes through to the cache as-is. When set, the box is
  treated as user-owned: out-of-box particles error instead of triggering an
  automatic recenter.
- `rectangular`: when `true`, derived bounds keep per-axis tight extents
  (padded per face) instead of cubing the domain (FastMultipole task 037
  rectangular radix grid). Cells stay physically cubic — the leaf width and
  the auto-geometry rule are unchanged (`ell` and `q` still derive from the
  maximum extent via sigma-adequacy) — and the coarse tree levels above the
  shortest axis's saturation are trimmed, removing launch floor on elongated
  (wake-like) domains. Off by default: the legacy cubic derivation is
  preserved exactly. Ignored when explicit `bounds` are set (the user-owned
  box's shape is final).
- `precision`: lifecycle float type; defaults to `eltype(pfield)`.
- `direct_kernel`: nearfield strategy `:partitioned` (default since task 035
  cycle 1; `PartitionedVortex`, FastMultipole's user-approved 032a default
  for sigma-carrying vortex systems), `:regularized` (the
  regularized-everywhere `RegularizedVortex`), or `:twopass`
  (`TwoPassVortex`).
- `rho_t`: override the nearfield kernel's smoothing-cutoff radius. For
  `:partitioned` the coupling default is `4.789` (conservative Jacobian
  per-pair cutoff; task 048 production selection, 2026-08-22 — see
  `_PARTITIONED_RHO_T_DEFAULT`); `:regularized` and `:twopass` default to
  their shipped constructor values.
- `m2l_strategy`: `:concat` (default since 2026-09-23, `ConcatenatedFixedZM2L`:
  on the H200 at 800k particles the 5MW M2L stage took 7.9 s vs 19.6 s for
  `:dense` over 72 steps, job 13869555, and locally at 22k concat was 2.7x
  faster; identical CP), `:dense` (`DenseTranslationM2L`, the task-035 default
  whose sweep had measured concat 1.6-2.8x slower at matched geometry — no
  longer true on the KA lifecycle), or `:precomputed_y`
  (`PrecomputedFactoredYM2L`).
- `level_radii2`: per-M2L-level near radii (levels `2:ell`, coarse to fine,
  non-increasing, ending at the leaf radius); `nothing` = uniform.
- `accuracy_margin`: multiplier on the kernel's `rho_t` in the auto-geometry
  rule (task 035 cycle 1). The bare adequacy gate
  (`g_min*h_leaf > rho_t*sigma_max`) is insufficient for the 1e-3 velocity
  tolerance at the margin — the 035 sweep measured `x = g_min*h/sigma_max`
  of `4.26` failing (1.17e-3) and `4.92` passing (9.3e-4) at n=1e5 — so
  auto-depth selection requires `x >= accuracy_margin*rho_t`. The default
  `1.03` is the center of the interval `[1.0, 1.061]` for which the rule
  reproduces every measured cycle-3A P5 winner (cube `(4,12)`/`(5,12)`,
  wake `(5,6)`/`(6,6)` at n = 1e5/1e6) at `rho_t = 3.668`; at the cycle-1
  defaults (`rho_t = 4.252`, floor 16) the validated margin was `1.15`.
"""
Base.@kwdef struct RadixFMMSettings
    expansion_order::Union{Nothing,Int} = 6
    ell::Union{Nothing,Int} = nothing
    near_radius2::Int = 6
    window_classes::Union{Nothing,Int} = nothing
    padding::Float64 = 0.1
    bounds::Union{Nothing,Tuple} = nothing
    rectangular::Bool = false
    precision::Union{Nothing,DataType} = nothing
    direct_kernel::Symbol = :partitioned
    rho_t::Union{Nothing,Float64} = nothing
    rho_c::Union{Nothing,Float64} = nothing
    m2l_strategy::Symbol = :concat
    level_radii2::Union{Nothing,Tuple} = nothing
    accuracy_margin::Float64 = 1.03
    # The radix depth is always the DEEPEST the adequacy gate admits, capped
    # by the occupancy heuristic, and there is no selector in front of it. It
    # measured fastest at every scale tried, from one rotor at 69k particles
    # to the NREL 5MW case at 653k, and it carries no fitted constants, so
    # nothing needs recalibrating when the kernels, the precision or the
    # device change.
    # Growth in `np` that makes the cache reconsider its depth. The depth is
    # otherwise held for the whole epoch, and the per-cell near-field work
    # grows faster than `np` does (the box grows too as a wake convects), so a
    # long epoch is expensive: on the NREL 5MW production run one epoch ran
    # 779 steps from 215k to 687k particles with the step going 1.8 s -> 89.3 s.
    # 1.5 since 2026-09-21: with the oversize masking the sigma-triggered
    # rebuilds no longer fire, so the growth check is the only trigger, and the
    # box of a convecting wake grows ~1.8x between doublings. A check costs
    # 0.02 s; it rebuilds only when (ell, q) would change.
    rebuild_growth::Float64 = 1.5
    # Oversize-core handling (2026-09-21). The near-field stencil is sized by the
    # largest core in the field, so a handful of stretched far-wake particles
    # (0.3% above 1.5x the shed core on the NREL 5MW at rev 21, one at 2.5x)
    # forced a coarser grid on a million: the step cost per particle rose 40%
    # over a run. With `oversize_count = K > 0`, every evaluation takes the K
    # particles with the largest cores OUT of the tree (their strength and core
    # are masked to zero while the field is packed, so the geometry, the
    # adequacy gate, the near field and the SFS sweep see the (K+1)-th largest
    # core) and puts them back through FastMultipole's `MaskedBodies`, each at the
    # coarser tree level its reach admits (direct to the targets near it there,
    # far field through the tree; 2026-10-03: 2M 5MW wake on the H200, 74 -> 15 ms
    # against the old all-pairs arm, accuracy identical). Their strength and core
    # are restored before the call returns. Dropped: their contribution to the
    # other particles' SFS estimator. `oversize_count = 0` (default) uses
    # `oversize_fraction` of the live count, clamped to [32, 4096]; a negative
    # count disables the handling. Measured on the NREL 5MW rev-15 state (705k,
    # largest core 10.45 m, auto grid 4.9 s/step): K 256 -> 4.2 s, K 1024 ->
    # 2.55 s (grid one level deeper, near field 50% -> 31% of the pass, the
    # all-pairs arm 0.7%), K 4096 -> 3.05 s (host overhead before the
    # vectorized masking). The core tail is a continuum, not a few outliers, so
    # a few dozen masked particles achieve nothing.
    # `oversize_count = 0` (default) is ADAPTIVE (2026-09-22): the mask takes
    # every core the occupancy-chosen grid cannot admit. The geometry the field
    # would pick if its core tail stopped at the (K_max+1)-th largest core is
    # derived (K_max = `oversize_fraction` of the live count), and every core
    # above THAT geometry's adequacy limit is masked -- so the grid depth is set
    # by the particle count, never by the tail, as long as the tail is smaller
    # than the fraction. The threshold is re-derived when the count grows 5% or
    # every 60 evaluations. Measured: the HVAB hover at 540k particles ran at
    # 36 s/step with the old fixed 0.15% (clamped to 4096) rule and 1.3 s/step
    # with 4096 masked; the tail is a continuum (the 0.15% remainder still
    # tracked the runaway) so a count-free rule is needed.
    oversize_count::Int = 0
    oversize_fraction::Float64 = 0.02
end

# Deepest radix level the dense per-level node table allows (8^ell Int32).
const _RADIX_MAX_ELL = 8

"""
    _validate_radix_fmm_settings(settings::RadixFMMSettings)

Eagerly validate every per-field setting using the same kernel, strategy,
stencil, lifecycle-option, and bounds contracts consumed by cache
construction. This is deliberately independent of a `ParticleField`, so
`radix_fmm_settings!` can reject the complete proposal before changing global
GPU settings or invalidating a live cache.
"""
function _validate_radix_fmm_settings(settings::RadixFMMSettings)
    P = settings.expansion_order
    (P === nothing || P >= 0) || throw(ArgumentError(
        "RadixFMMSettings.expansion_order must be nonnegative or nothing; got $P"))

    ell = settings.ell
    (ell === nothing || 2 <= ell <= fmm.RADIX_GRID_MAX_ELL) || throw(ArgumentError(
        "RadixFMMSettings.ell must lie in 2:$(fmm.RADIX_GRID_MAX_ELL) or be nothing; got $ell"))

    q = fmm._validate_rigid_near_radius2(
        settings.near_radius2, "RadixFMMSettings")
    (settings.window_classes === nothing || settings.window_classes > 0) ||
        throw(ArgumentError("RadixFMMSettings.window_classes must be positive or nothing"))
    (isfinite(settings.padding) && settings.padding > 0) || throw(ArgumentError(
        "RadixFMMSettings.padding must be finite and positive; got $(settings.padding)"))
    (isfinite(settings.accuracy_margin) && settings.accuracy_margin > 0) ||
        throw(ArgumentError("RadixFMMSettings.accuracy_margin must be finite and positive; got $(settings.accuracy_margin)"))

    settings.direct_kernel in (:regularized, :partitioned, :twopass) ||
        throw(ArgumentError("RadixFMMSettings.direct_kernel must be :regularized, " *
            ":partitioned, or :twopass; got $(repr(settings.direct_kernel))"))
    (settings.rho_c === nothing || settings.direct_kernel === :twopass) ||
        throw(ArgumentError("RadixFMMSettings.rho_c applies only to direct_kernel=:twopass"))
    kernel = _radix_direct_kernel(settings) # actual rho_t/rho_c constructor contract
    kernel isa fmm.AbstractDirectKernel || error("unreachable direct-kernel resolution")
    settings.m2l_strategy in (:concat, :dense, :precomputed_y) ||
        throw(ArgumentError("RadixFMMSettings.m2l_strategy must be :concat, :dense, " *
            "or :precomputed_y; got $(repr(settings.m2l_strategy))"))
    m2l_strategy, operator = _radix_m2l_strategy(settings)
    TF = something(settings.precision, Float64)
    TF in (Float32, Float64) || throw(ArgumentError(
        "RadixFMMSettings.precision must be Float32, Float64, or nothing; got $TF"))
    fmm.CUDARadixLifecycleOptions(;
        precision=TF, operator, m2l_strategy) # actual strategy/precision contract

    settings.level_radii2 === nothing ||
        fmm._validate_rigid_level_schedule(settings.level_radii2, q)

    if settings.bounds !== nothing
        length(settings.bounds) == 2 || throw(ArgumentError(
            "RadixFMMSettings.bounds must be (x_min, box_size)"))
        x_min = try
            fmm.SVector{3,TF}(settings.bounds[1])
        catch
            throw(ArgumentError("RadixFMMSettings.bounds x_min must have three numeric entries"))
        end
        all(isfinite, x_min) || throw(ArgumentError(
            "RadixFMMSettings.bounds x_min must be finite"))
        ell_for_bounds = something(ell, 2)
        try
            fmm._resolve_radix_ell_axes(settings.bounds[2], ell_for_bounds, TF)
        catch err
            err isa ArgumentError && rethrow()
            throw(ArgumentError("invalid RadixFMMSettings.bounds box_size: $(sprint(showerror, err))"))
        end
    end
    return settings
end

"Field-aware semantic validation for contracts that depend on derived geometry."
function _validate_radix_fmm_settings(pfield::ParticleField,
                                      settings::RadixFMMSettings)
    _validate_radix_fmm_settings(settings)
    settings.level_radii2 === nothing && return settings

    # Mirror the side-effect-free geometry portion of `_build_radix_fmm_cache`:
    # auto ell and rectangular active levels depend on the live field/bounds.
    bounds, ell, q = fmm.radix_choose_geometry(_radix_geometry_policy(settings),
        fmm.radix_geometry_source(pfield))
    TF = something(settings.precision, eltype(pfield))
    ell_axes, _, _ = fmm._resolve_radix_ell_axes(bounds[2], ell, TF)
    R, L_allnear = fmm._radix_root_level(ell_axes, ell, q)
    first_m2l = R == L_allnear ? R + 1 : R
    # Zero-M2L degenerate geometry runs pure direct (task 052c) — legal
    # without a schedule, but an explicit level_radii2 cannot anchor to zero
    # active levels (we are in the level_radii2 !== nothing branch here).
    first_m2l <= ell || throw(ArgumentError(
        "RadixFMMSettings.level_radii2 cannot apply: the live field resolves " *
        "to ell=$ell, ell_axes=$(Tuple(ell_axes)), near_radius2=$q with no " *
        "M2L level (pure direct evaluation); omit level_radii2"))
    active_length = ell - first_m2l + 1
    legacy_length = ell - 1
    n = length(settings.level_radii2)
    n in (active_length, legacy_length) || throw(ArgumentError(
        "RadixFMMSettings.level_radii2 has $n entries, but the live field " *
        "resolves to ell=$ell with active M2L levels $first_m2l:$ell; " *
        "expected $active_length active entries or $legacy_length legacy " *
        "entries anchored to levels 2:$ell"))
    return settings
end

# The primary direct-list geometry must cover the branch evaluated directly by
# the selected kernel. TwoPassVortex supplies the remaining (rho_c, rho_t]
# regularization deficit through its independent correction traversal, so using
# rho_t here would unnecessarily force the primary list to cover both passes.
# The geometry rule (box, depth, near stencil; when to rebuild or recenter) is
# FastMultipole's (src/radix_geometry.jl); FLOWVPM describes its field to it and
# maps its settings onto the uniform grid's policy.
fmm.radix_geometry_source(pfield::ParticleField) =
    (; P = pfield.particles, x_rows = X_INDEX, core_row = SIGMA_INDEX, n = pfield.np)
_radix_geometry_policy(settings::RadixFMMSettings) = fmm.AutoUniformGeometry(;
    reach = fmm.radix_primary_reach(_radix_direct_kernel(settings)),
    near_radius2 = settings.near_radius2, accuracy_margin = settings.accuracy_margin,
    ell = settings.ell, padding = settings.padding, rectangular = settings.rectangular,
    bounds = settings.bounds, rebuild_growth = settings.rebuild_growth, max_ell = _RADIX_MAX_ELL,
    oversize = settings.oversize_count < 0 ? fmm.NoOversize() :
               settings.oversize_count > 0 ? fmm.FixedOversize(settings.oversize_count) :
               fmm.AdaptiveOversize(settings.oversize_fraction))
_radix_verbose() = get(ENV, "FLOWVPM_RADIX_VERBOSE", "0") == "1"

# WeakKeyDicts so a discarded ParticleField releases its cache (and its GPU
# memory) instead of being pinned forever by the registry.
const _radix_fmm_settings = WeakKeyDict{Any,RadixFMMSettings}()
const _radix_fmm_couplings = WeakKeyDict{Any,Any}()

"""
    radix_fmm_settings!(pfield::ParticleField; kwargs...)

Set radix FMM coupling overrides for `pfield` (see [`RadixFMMSettings`](@ref))
and invalidate any existing cache so the next evaluation rebuilds with the new
settings. GPU mechanism tunables (FastMultipole's radix settings, e.g.
`CUDA_NEARFIELD_GH_MODE`) may be passed as `gpu=(; CUDA_NEARFIELD_GH_MODE=:shipped)`;
they are validated and atomically applied by `FastMultipole.set_radix_settings!` before the
cache invalidation so construction-locked settings take effect on the rebuild. Not exported; internal tuning surface (task 035 owns performance).
"""
function radix_fmm_settings!(pfield::ParticleField; gpu::NamedTuple=NamedTuple(), kwargs...)
    # task 047: GPU mechanism tunables flow through FastMultipole's validated
    # settings surface, applied BEFORE the cache is cleared so the rebuild
    # snapshots the new values (construction-locked settings must be set
    # pre-construction; late flips error loudly at the next device step).
    # Ordering contract: construct and semantically validate the complete local
    # proposal before the atomic GPU batch write; only after both validations
    # succeed do we replace the per-field value and clear its cache. Thus any
    # invalid local or GPU proposal leaves all three state surfaces unchanged.
    proposed = _validate_radix_fmm_settings(
        pfield, RadixFMMSettings(; kwargs...))
    fmm.set_radix_settings!(gpu)
    _radix_fmm_settings[pfield] = proposed
    clear_radix_fmm_cache!(pfield)
    return _radix_fmm_settings[pfield]
end

"Drop the cached `RadixFMMCache` (if any) for `pfield`."
function clear_radix_fmm_cache!(pfield::ParticleField)
    delete!(_radix_fmm_couplings, pfield)
    delete!(_radix_oversize_thr, pfield)
    return nothing
end

"""
    radix_state(pfield) -> Dict or nothing

The history the radix coupling carries between evaluations, for a checkpoint:
the live cache's geometry (box, depth and stencil radius as they are now, after
any `recenter!` -- not what the field would derive from its particles), the
depth check's counters and the oversize threshold. `nothing` when the field has
no coupling yet. [`restore_radix_state!`](@ref) rebuilds the coupling from it,
so a restarted run evaluates exactly as the uninterrupted one would have.
"""
function radix_state(pfield::ParticleField)
    st = get(_radix_fmm_couplings, pfield, nothing)
    st === nothing && return nothing
    c = st.cache
    rectangular = c.ell_axes != SVector(c.ell, c.ell, c.ell)
    # the stencil the cache actually has, which FastMultipole's all-direct
    # fallback can change without the coupling knowing; `built_q` and `st_q`
    # are the coupling's own records, which later rebuild checks read
    pol = c.policy
    rigid = pol isa fmm.HierarchicalRigidStencil
    built_q = get(_radix_built_q, pfield, st.q)
    d = Dict{Symbol,Any}(:x_min => Vector(c.x_min),
        :box => rectangular ? Vector(c.box_extent) : 2 * c.h0,
        :ell => c.ell, :q => rigid ? pol.near_radius2 : built_q,
        :level_radii2 => rigid ? pol.level_radii2 : st.settings.level_radii2,
        :built_q => built_q, :st_q => st.q, :settings => st.settings,
        :np_checked => st.np_checked[], :evals => st.evals[])
    if haskey(_radix_oversize_thr, pfield)
        rec = _radix_oversize_thr[pfield]
        d[:oversize] = rec === nothing ? nothing :
            Dict{Symbol,Any}(:thr => rec.thr, :np => rec.np, :evals => rec.evals[])
    end
    return d
end

"""
    restore_radix_state!(pfield, d)

Rebuild `pfield`'s radix coupling from [`radix_state`](@ref)'s `Dict`; the
particles must already be restored. `nothing` drops any coupling, as for a
field that had none.
"""
function restore_radix_state!(pfield::ParticleField, d)
    clear_radix_fmm_cache!(pfield)
    d === nothing && return nothing
    # the settings the coupling ran with, also for any later rebuild
    settings = d[:settings]
    _radix_fmm_settings[pfield] = settings
    bounds = (d[:x_min], d[:box])
    # Built on the masked field, as an evaluation builds it: the build's first
    # update runs the adequacy gate, which on the unmasked cores demoted the
    # restored geometry to the all-direct cache for the rest of the run (5MW
    # step-3024 checkpoint, 2026-10-03). The mask here is only what the saved
    # geometry cannot admit; the first evaluation re-selects and re-packs.
    lim = fmm.radix_sigma_limit(_radix_geometry_policy(settings), d[:ell], d[:box], d[:q], d[:level_radii2])
    idx = settings.oversize_count < 0 ? Int[] :
        fmm.radix_rows_above(pfield.particles, SIGMA_INDEX, pfield.np, lim, pfield.np)
    saved = isempty(idx) ? nothing : fmm.radix_mask_bodies!(pfield, idx)
    cache = try
        _build_radix_fmm_cache(pfield, settings;
            geometry=(bounds, d[:ell], d[:q], d[:level_radii2]))
    finally
        saved === nothing || fmm.radix_unmask_bodies!(pfield, idx, saved)
    end
    _radix_built_q[pfield] = d[:built_q]
    _radix_fmm_couplings[pfield] = (; cache, settings, np_checked=Ref(d[:np_checked]),
        sigma_limit=fmm.radix_sigma_limit(_radix_geometry_policy(settings), cache), q=d[:st_q], evals=Ref(d[:evals]),
        sfs=Ref{Any}(nothing))
    if haskey(d, :oversize)
        o = d[:oversize]
        _radix_oversize_thr[pfield] = o === nothing ? nothing :
            (; thr=o[:thr], np=o[:np], evals=Ref(o[:evals]))
    end
    return nothing
end

################################################################################
# Configuration derivation
################################################################################

"""
    _validate_radix_fmm_settings(pfield)

The radix path runs with parameters fixed at cache construction: all FMM
autotuning must be off, and the kernel must be `gaussianerf`. Fails loudly
otherwise (no silent fallback).
"""
function _validate_radix_fmm_settings(pfield::ParticleField)
    f = pfield.fmm
    if f.autotune_p || f.autotune_ncrit || f.autotune_reg_error
        error("the radix/GPU FMM path uses parameters fixed at cache " *
            "construction and does not support FMM autotuning. Construct the " *
            "particle field with, e.g., FMM(; p=4, autotune_p=false, " *
            "autotune_ncrit=false, autotune_reg_error=false, " *
            "default_rho_over_sigma=1.0). Got autotune_p=$(f.autotune_p), " *
            "autotune_ncrit=$(f.autotune_ncrit), " *
            "autotune_reg_error=$(f.autotune_reg_error).")
    end
    is_gaussianerf(pfield.kernel) || error(
        "the radix/GPU FMM path supports only the `gaussianerf` kernel.")
    pfield.np >= 1 || error("radix FMM coupling requires at least one particle")
    return nothing
end

# min/max of one particle-matrix row over the live prefix. FastMultipole's
# helper is a plain view reduction on the host and a FIXED-geometry kernel on a
# device field: a GPUArrays reduction over an np-length view compiles a fresh
# kernel for every distinct np (~140 ms each on Metal), and np changes every
# step of a shedding solver.
_radix_row_extrema(pfield::ParticleField, row::Int) =
    fmm._device_row_extrema(pfield.particles, row, pfield.np)

_radix_sigma_max(pfield::ParticleField) = _radix_row_extrema(pfield, SIGMA_INDEX)[2]






################################################################################
# Cache construction and evaluation
################################################################################

function _build_radix_fmm_cache(pfield::ParticleField{R},
                                settings::RadixFMMSettings;
                                max_n_bodies::Int=pfield.maxparticles,
                                geometry=nothing) where R
    _validate_radix_fmm_settings(pfield)
    device = !(pfield.particles isa Array)
    if device
        # A non-Array-backed field takes the device path on whatever backend it
        # lives on. The lifecycle is registered with FastMultipole by its
        # KernelAbstractions extension (register_radix_device_backend!), which
        # loads once KernelAbstractions and a GPU package are both loaded.
        fmm.radix_device_backend_available() ||
            error("GPU FMM requested (device-backed particle field) but no " *
                "device radix lifecycle is available: $(fmm.radix_device_status())")
    end

    P = settings.expansion_order === nothing ? pfield.fmm.p - 1 :
        settings.expansion_order
    direct_kernel = _radix_direct_kernel(settings)
    level_radii2 = settings.level_radii2
    if geometry === nothing
        bounds, ell, q = fmm.radix_choose_geometry(_radix_geometry_policy(settings),
            fmm.radix_geometry_source(pfield); verbose = _radix_verbose())
    else
        # a checkpoint's geometry (see `radix_state`): the box, depth and
        # stencil the live cache had, not the ones the field would derive now
        bounds, ell, q, level_radii2 = geometry
    end
    _radix_built_q[pfield] = q         # the depth check compares (ell, q), not ell alone
    TF = settings.precision === nothing ? R : settings.precision
    K = settings.window_classes === nothing ? (device ? 256 : nothing) :
        settings.window_classes
    m2l_strategy, operator = _radix_m2l_strategy(settings)
    opts = fmm.CUDARadixLifecycleOptions(; precision=TF, operator, m2l_strategy)

    # Capacity contract: sized once to maxparticles; live np may vary below it
    # (particles added/removed between steps) with no reallocation.
    # The SFS pass is FLOWVPM's own (FLOWVPM_fmm_radix_sfs.jl), run through
    # fmm!'s nearfield_pass; it needs hessian=true and sigma in packed row 8.
    return fmm.RadixFMMCache(pfield;
        expansion_order=P, ell,
        max_n_bodies,
        bounds=(SVector{3,TF}(bounds[1]),
            bounds[2] isa Real ? TF(bounds[2]) : SVector{3,TF}(bounds[2])),
        hessian=true,
        near_radius2=q,
        level_radii2, window_classes=K,
        # the leaf radius q is set by core reach; cell pairs between the accuracy
        # floor and q go to M2L whenever no core reaches them (FastMultipole leaf band)
        near_floor2=settings.near_radius2,
        device, options=opts)
end




"""
    _radix_fmm_coupling!(pfield) -> (; cache, settings, np_checked, sigma_limit)

Get-or-create the persistent radix coupling for `pfield`. The cache is sized
once to `pfield.maxparticles` and reused by every subsequent evaluation;
`clear_radix_fmm_cache!` or `radix_fmm_settings!` invalidate it, and the
auto-derived geometry rebuilds when the live field outgrows it in either
direction: particle count vs grid depth (`FastMultipole.radix_depth_outgrown!`)
or `sigma_max` vs the adequacy limit (`FastMultipole.radix_sigma_outgrown!`).
"""
const _radix_built_q = Dict{Any,Int}()      # stencil radius the last build chose, per field
# the live field outgrew the cached geometry in either direction (FastMultipole's
# rule): particle count/box vs depth and stencil, or largest core vs the adequacy limit
function _radix_geometry_outgrown!(pfield::ParticleField, st)
    pol = _radix_geometry_policy(st.settings); src = fmm.radix_geometry_source(pfield)
    v = _radix_verbose()
    return fmm.radix_depth_outgrown!(pol, src, st.cache, st.q, st.np_checked, st.evals; verbose = v) ||
           fmm.radix_sigma_outgrown!(pol, src, st.cache; verbose = v)
end
function _radix_fmm_coupling!(pfield::ParticleField)
    st = get(_radix_fmm_couplings, pfield, nothing)
    if st !== nothing && _radix_geometry_outgrown!(pfield, st)
        get(ENV, "FLOWVPM_RADIX_VERBOSE", "0") == "1" && (println("radix coupling dropped for rebuild at np=$(pfield.np)"); flush(stdout))
        delete!(_radix_fmm_couplings, pfield)
        st = nothing
        # the old cache's device arrays are freed only by their finalizers;
        # collect now so the new cache does not allocate on top of the old one
        GC.gc()
    end
    if st === nothing
        settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
        cache = _build_radix_fmm_cache(pfield, settings)
        st = (; cache, settings, np_checked=Ref(pfield.np),
                sigma_limit=fmm.radix_sigma_limit(_radix_geometry_policy(settings), cache),
                q=get(_radix_built_q, pfield, settings.near_radius2), evals=Ref(0),
                sfs=Ref{Any}(nothing))            # the SFS pass's scratch, on first use
        _radix_fmm_couplings[pfield] = st
    end
    return st
end

"""
    _radix_fmm_evaluate!(pfield)

One U/J evaluation through the radix FMM lifecycle: velocity into `U_INDEX`
and the full 9-component velocity gradient into `J_INDEX`, ACCUMULATED
(FLOWVPM's own `_reset_particles` zeroes U/J before each evaluation; the
framework delivers the total influence of the evaluation).

Recenter policy: `fmm!` throws `ArgumentError` when a particle leaves the
cache's fixed box. With derived bounds the coupling recenters once
(`fmm.recenter!`, derived padded bounds, no reallocation) and retries; with
user-fixed `bounds` the error propagates (the box is a user promise).
"""
#--- oversize cores (see RadixFMMSettings.oversize_count) ---#


# The K particles with the largest cores when they stand clear of the rest:
# global (unsorted) indices, or an empty list. Cores are read once through a
# host copy of the live sigma row.
function _radix_oversize_count(settings, np::Int)
    settings.oversize_count < 0 && return 0
    settings.oversize_count > 0 && return settings.oversize_count
    return clamp(round(Int, settings.oversize_fraction * np), 32, 4096)
end

# per-field adaptive threshold: (; thr, np, evals) or nothing
# FastMultipole's adaptive-threshold record per field (radix_oversize_threshold),
# kept here because the checkpoint carries it
const _radix_oversize_thr = IdDict{Any,Any}()


# global indices of the particles with sigma > thr (host; the GPU extension
# overloads it with the histogram/collect kernel), at most `cap` of them

# Column gather / mask / scatter over the oversize index list. Matrix fields
# index directly; the GPU extension overloads all three with one kernel each
# (K can be in the thousands: per-column device writes would be K launches).
# the core-row selections, kept under these names for FLOWUnsteadyCore's core
# splitting and reset (FastMultipole has the host and device methods)
_radix_oversize_above(P, np::Int, thr, cap::Int) = fmm.radix_rows_above(P, SIGMA_INDEX, np, thr, cap)
_radix_oversize_top(P, np::Int, K::Int) = fmm.radix_rows_top(P, SIGMA_INDEX, np, K)

_radix_oversize_gather(P::Matrix, idx::Vector{Int}, rows) = P[rows, idx]
function _radix_oversize_mask_rows!(P::Matrix, idx::Vector{Int}, rows)
    @inbounds for i in idx, r in rows
        P[r, i] = zero(eltype(P))
    end
    return nothing
end
function _radix_oversize_scatter!(P::Matrix, idx::Vector{Int}, rows, vals::Matrix)
    @inbounds for (k, i) in enumerate(idx), (q, r) in enumerate(rows)
        P[r, i] = vals[q, k]
    end
    return nothing
end

# FastMultipole's masking hooks: the packed columns of the masked particles (the
# layout `source_system_to_buffer!` writes, with the default rho/sigma radius), and
# their strength and core zeroed in place; the unmask writes them back
function fmm.radix_mask_bodies!(pfield::ParticleField, idx::Vector{Int})
    P = pfield.particles
    TF = eltype(P)
    K = length(idx)
    rows = first(X_INDEX):SIGMA_INDEX      # 1:7 -- X, Gamma, sigma
    col = _radix_oversize_gather(P, idx, rows)           # host 7 x K
    buf = zeros(TF, 9, K)
    rho = TF(pfield.fmm.default_rho_over_sigma)
    @inbounds for k in 1:K
        buf[1, k] = col[1, k]; buf[2, k] = col[2, k]; buf[3, k] = col[3, k]
        buf[5, k] = col[4, k]; buf[6, k] = col[5, k]; buf[7, k] = col[6, k]
        sig = col[7, k]
        buf[4, k] = rho * sig
        buf[8, k] = sig
        buf[9, k] = one(TF)
    end
    _radix_oversize_mask_rows!(P, idx, first(GAMMA_INDEX):SIGMA_INDEX)   # 4:7
    return buf
end
function fmm.radix_unmask_bodies!(pfield::ParticleField, idx::Vector{Int}, buf)
    _radix_oversize_scatter!(pfield.particles, idx, first(GAMMA_INDEX):SIGMA_INDEX, buf[5:8, :])
    return nothing
end

function _radix_fmm_evaluate!(pfield::ParticleField; sfs::Bool=false,
        extra_targets::Tuple=(), extra_sources::Tuple=(), tree_sources::Tuple=(),
        self_induce::Bool=true,
        extra_hessian::Tuple=ntuple(_ -> false, length(extra_targets)))
    # oversize cores out of the tree BEFORE the coupling looks at the field, so
    # the geometry and its gates see the (K+1)-th largest core
    # ... on EVERY call, the sources-only ones included: a sources-only call
    # (bodies onto the wake between RK stages) that saw the unmasked cores judged
    # the cached geometry outgrown and rebuilt it one level shallower for all the
    # passes that followed (2026-09-21, 5MW rev 15: 2.3 -> 4.9 s/step). The masked
    # particles ride as an extra source only when the particles are sources.
    settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
    st0 = get(_radix_fmm_couplings, pfield, nothing)
    rec0 = get(_radix_oversize_thr, pfield, nothing)
    oversize, rec = fmm.radix_oversize_select(_radix_geometry_policy(settings),
        fmm.radix_geometry_source(pfield), rec0, st0 === nothing ? nothing : st0.cache;
        verbose = _radix_verbose())
    rec === rec0 || (_radix_oversize_thr[pfield] = rec)
    ov = isempty(oversize) ? nothing :
        fmm.MaskedBodies(fmm.radix_mask_bodies!(pfield, oversize), _radix_direct_kernel(settings), 3;
            idx = oversize, bodytype = fmm.body_type(pfield), margin = settings.accuracy_margin)
    try
        st = _radix_fmm_coupling!(pfield)
        # extra targets (probes, ring nodes) and extra sources (bound segments,
        # ring filaments) ride the same call as direct rectangular evaluations
        # (FastMultipole src/radix_extra_systems.jl); the particles keep their
        # hessian, the extra targets take velocity only unless `extra_hessian`
        # asks for their velocity gradient too (fluid-domain probes).
        #
        # `self_induce=false` drops the particles from the source tuple: the
        # lifecycle is skipped and the call delivers the extra sources alone, which
        # is the "body on wake" direction of a coupled solve (the self-induction
        # was evaluated earlier in the step, over a field that has since changed).
        targets = (pfield, extra_targets...)
        ov_sources = (ov === nothing || !self_induce) ? () : (ov,)
        sources = self_induce ? (pfield, ov_sources..., extra_sources...) : extra_sources
        length(extra_hessian) == length(extra_targets) ||
            throw(ArgumentError("one extra_hessian flag per extra target is required"))
        # The particles take U AND J from every source, the extra sources included:
        # a sources-only call (bound segments and rings onto the wake between the
        # RK3 stages) must deliver the filaments' velocity gradient too, or the
        # reformulated VPM stretches the wake with part of the field missing.
        # (`self_induce` used to sit here, which dropped J on exactly that call.)
        hessian = (true, extra_hessian...)
        # the two-level dynamic procedure asks for the analytic core-scaling
        # derivatives of the SFS pass (see dynamicprocedure_twolevel_beforeUJ)
        sfs_dsigma = sfs && _sfs_dsigma_requested(pfield)
        # the SFS estimator is FLOWVPM's pass over the lifecycle's near field,
        # run inside fmm! when the particles' U and J are complete (before the
        # extra sources), delivered after fmm! returns
        ctx = sfs ? _radix_sfs_context!(pfield, st) : nothing
        # With oversize particles masked, the pass runs after fmm! instead: their
        # velocity gradient (the all-pairs extra source) is in the output only then,
        # and the pass adds their pairs (`radix_nearfield(cache).masked`)
        fmm.radix_set_masked!(st.cache, ov === nothing ? nothing : (oversize, ov.buffer))
        masked_sfs = sfs && ov !== nothing
        nearfield_pass = (sfs && !masked_sfs) ?
            (c -> _radix_sfs_pass!(pfield, ctx, fmm.radix_nearfield(c); dsigma=sfs_dsigma)) : nothing
        try
            fmm.fmm!(targets, sources, st.cache;
                scalar_potential=false, gradient=true, hessian, nearfield_pass,
                tree_sources, metadata=0)
        catch err
            (err isa ArgumentError && st.settings.bounds === nothing) || rethrow()
            # out-of-box (or other geometry) rejection: recenter and retry once;
            # a second failure (e.g. adequacy gate on the grown box) propagates
            bounds = fmm.radix_recenter_bounds(_radix_geometry_policy(st.settings),
                fmm.radix_geometry_source(pfield), st.cache.ell)
            get(ENV, "FLOWVPM_RADIX_VERBOSE", "0") == "1" && (println("radix recenter at np=$(pfield.np): ",
                sprint(showerror, err; context=:limit => true)[1:min(end, 120)]); flush(stdout))
            fmm.recenter!(st.cache, pfield; bounds)
            fmm.fmm!(targets, sources, st.cache;
                scalar_potential=false, gradient=true, hessian, nearfield_pass,
                tree_sources, metadata=0)
        end
        masked_sfs && _radix_sfs_pass!(pfield, ctx, fmm.radix_nearfield(st.cache); dsigma=sfs_dsigma)
        sfs && _radix_sfs_deliver!(pfield, ctx, fmm.radix_nearfield(st.cache); dsigma=sfs_dsigma)
    finally
        ov === nothing || fmm.radix_unmask_bodies!(pfield, oversize, ov.buffer)
    end
    return nothing
end

"""
    UJ_fmm_gpu!(pfield; reset=true, reset_sfs=false, sfs=false, rbf=false,
                verbose=false, extra_targets=(), extra_sources=())

GPU/radix counterpart of `UJ_fmm` for a `CuArray`-backed `ParticleField`
(also runnable on a `Matrix`-backed field through FastMultipole's
transfer-based host path, used by the CPU-side tests). Computes U and J via
the resident radix FMM (device-to-device for a GPU field: no per-step body
transfers) and accumulates into `U_INDEX`/`J_INDEX`. With `sfs=true` the
lifecycle's SFS pass (task 048) additionally ACCUMULATES the vortex-stretching
term `E_str` into `SFS_INDEX` (rows 40:42) — the caller owns the reset
(`UJ_fmm` resets SFS rows before dispatching here, matching the CPU
`Estr_fmm!` convention). Unsupported configurations (`rbf`) fail loudly
rather than silently dropping physics.
"""
function UJ_fmm_gpu!(pfield::ParticleField;
        reset::Bool=true, reset_sfs::Bool=false, sfs::Bool=false,
        rbf::Bool=false, verbose::Bool=false,
        extra_targets::Tuple=(), extra_sources::Tuple=(), tree_sources::Tuple=(),
        self_induce::Bool=true, extra_hessian::Tuple=ntuple(_ -> false, length(extra_targets)),
        optargs...)
    rbf && error("rbf/zeta evaluation is not supported on the radix/GPU FMM " *
        "path (use UJ_direct/zeta_direct for CuArray-backed fields)")
    reset && _reset_particles(pfield)
    reset_sfs && _reset_particles_sfs(pfield)
    _radix_fmm_evaluate!(pfield; sfs, extra_targets, extra_sources, tree_sources,
                         self_induce, extra_hessian)
    return nothing
end

################################################################################
# SFS delivery hook (task 048): FastMultipole hands back a framework-owned
# 3 x np E_str buffer in global particle order; ACCUMULATE into SFS_INDEX
# (rows 40:42), mirroring the `Estr_direct`/`Estr_fmm!` += convention (the
# caller resets SFS rows). This host-Matrix method serves the transfer-based
# host path; the CuArray method lives in ext/FLOWVPMGPUExt.jl.
################################################################################

else # !_FMM_HAS_RADIX ---------------------------------------------------------

function UJ_fmm_gpu!(pfield; optargs...)
    error("GPU FMM requires a FastMultipole version providing the " *
        "device-resident radix interface (RadixFMMCache; branch matrix-ops). " *
        "The installed FastMultipole does not. CPU (Matrix-backed) particle " *
        "fields are unaffected.")
end

function clear_radix_fmm_cache!(pfield)
    return nothing
end

radix_state(pfield) = nothing
restore_radix_state!(pfield, d) = nothing

end # _FMM_HAS_RADIX
