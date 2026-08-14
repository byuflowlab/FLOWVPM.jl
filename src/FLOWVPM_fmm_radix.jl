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
# `rho_t = 3.668` (035 cycles 1 and 3).
function fmm.direct_kernel(pfield::ParticleField)
    is_gaussianerf(pfield.kernel) || error(
        "the radix/GPU FMM path supports only the `gaussianerf` kernel " *
        "(FLOWVPM default; sole CoreSpreading-compatible kernel). " *
        "Got a different `pfield.kernel`.")
    settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
    return _radix_direct_kernel(settings)
end

# Coupling default smoothing cutoff for the partitioned kernel (task 035
# cycle 3, user-approved 2026-08-12): the 031a velocity-RMS-target cutoff.
# This task's winner gate is sampled velocity RMS <= 1e-3 (Jacobian is
# diagnostic), so the coupling defaults to rho_t = 3.668 rather than the
# kernel constructor's Jacobian-RMS-derived 4.252. The cycle-3C error
# decomposition validated it under the conservative sum gate
# (||P-R|| + ||F-P||)/||R|| < 1e-3 at cube/wake, n = 1e5/1e6. Applies only
# to :partitioned; :regularized/:twopass keep their constructor defaults.
const _PARTITIONED_RHO_T_DEFAULT = 3.668

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
  `:partitioned` the coupling default is `3.668` (031a velocity-RMS cutoff;
  task 035 cycle 3 — see `_PARTITIONED_RHO_T_DEFAULT`); `:regularized` and
  `:twopass` default to their shipped constructor values.
- `m2l_strategy`: `:dense` (default since task 035 cycle 1,
  `DenseTranslationM2L` — FastMultipole's own measured auto rule at P <= 4;
  the 035 sweep measured concat 1.6-2.8x slower at matched geometry),
  `:concat` (`ConcatenatedFixedZM2L`), or `:precomputed_y`
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
    expansion_order::Union{Nothing,Int} = 4
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
    m2l_strategy::Symbol = :dense
    level_radii2::Union{Nothing,Tuple} = nothing
    accuracy_margin::Float64 = 1.03
end

# The primary direct-list geometry must cover the branch evaluated directly by
# the selected kernel. TwoPassVortex supplies the remaining (rho_c, rho_t]
# regularization deficit through its independent correction traversal, so using
# rho_t here would unnecessarily force the primary list to cover both passes.
_radix_primary_reach(kernel) = Float64(kernel.rho_t)
_radix_primary_reach(kernel::fmm.TwoPassVortex) = Float64(kernel.rho_c)

# WeakKeyDicts so a discarded ParticleField releases its cache (and its GPU
# memory) instead of being pinned forever by the registry.
const _radix_fmm_settings = WeakKeyDict{Any,RadixFMMSettings}()
const _radix_fmm_couplings = WeakKeyDict{Any,Any}()

"""
    radix_fmm_settings!(pfield::ParticleField; kwargs...)

Set radix FMM coupling overrides for `pfield` (see [`RadixFMMSettings`](@ref))
and invalidate any existing cache so the next evaluation rebuilds with the new
settings. Not exported; internal tuning surface (task 035 owns performance).
"""
function radix_fmm_settings!(pfield::ParticleField; kwargs...)
    _radix_fmm_settings[pfield] = RadixFMMSettings(; kwargs...)
    clear_radix_fmm_cache!(pfield)
    return _radix_fmm_settings[pfield]
end

"Drop the cached `RadixFMMCache` (if any) for `pfield`."
function clear_radix_fmm_cache!(pfield::ParticleField)
    delete!(_radix_fmm_couplings, pfield)
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

# min/max of one particle-matrix row over the live prefix; a device reduction
# for CuArray-backed fields (six scalars cross to the host, never body arrays)
function _radix_row_extrema(pfield::ParticleField, row::Int)
    v = view(pfield.particles, row, 1:pfield.np)
    return minimum(v), maximum(v)
end

_radix_sigma_max(pfield::ParticleField) = _radix_row_extrema(pfield, SIGMA_INDEX)[2]

"""
    _radix_derive_bounds(pfield, padding; rectangular=false)
        -> (x_min::SVector{3}, box_size)

Domain bounds covering the live particles, padded by `padding` of the tight
extent on each face (the `recenter!` convention). Cubic mode (the default)
returns a scalar `box_size` from the maximum tight span; rectangular mode
(task 037) keeps per-axis tight extents and returns a 3-vector `box_size`,
each axis padded by the same per-face convention
(`L_a = (1 + 2*padding)*ext_a`, centered). In both modes degenerate extents
are inflated to `4*sigma_max` (per axis in rectangular mode) so a
near-singleton field still yields a valid box.
"""
function _radix_derive_bounds(pfield::ParticleField, padding::Real;
                              rectangular::Bool=false)
    lo1, hi1 = _radix_row_extrema(pfield, X_INDEX[1])
    lo2, hi2 = _radix_row_extrema(pfield, X_INDEX[2])
    lo3, hi3 = _radix_row_extrema(pfield, X_INDEX[3])
    cx = (lo1 + hi1) / 2
    cy = (lo2 + hi2) / 2
    cz = (lo3 + hi3) / 2
    floor4s = 4 * _radix_sigma_max(pfield)
    if !rectangular
        span = max(hi1 - lo1, hi2 - lo2, hi3 - lo3)
        L_tight = max(span, floor4s)
        L_tight > 0 || error("cannot derive radix FMM bounds: degenerate particle field")
        L = (1 + 2 * padding) * L_tight
        x_min = SVector{3,Float64}(cx - L / 2, cy - L / 2, cz - L / 2)
        return (x_min, Float64(L))
    end
    ex = max(hi1 - lo1, floor4s)
    ey = max(hi2 - lo2, floor4s)
    ez = max(hi3 - lo3, floor4s)
    (ex > 0 && ey > 0 && ez > 0) ||
        error("cannot derive radix FMM bounds: degenerate particle field")
    Lx = (1 + 2 * padding) * ex
    Ly = (1 + 2 * padding) * ey
    Lz = (1 + 2 * padding) * ez
    x_min = SVector{3,Float64}(cx - Lx / 2, cy - Ly / 2, cz - Lz / 2)
    return (x_min, SVector{3,Float64}(Lx, Ly, Lz))
end

"""
    _radix_center_snapped_bounds(bounds, ell) -> (x_min, box_extent)

Center the power-of-two rectangular embedding selected by FastMultipole around
the center of automatically derived tight bounds. The longest extent and leaf
width are unchanged; shorter extents are padded symmetrically to whole
power-of-two leaf-cell counts. Explicit user bounds do not use this helper and
therefore retain their caller-owned `x_min` anchor.
"""
function _radix_center_snapped_bounds(bounds, ell::Integer)
    x_min = SVector{3,Float64}(bounds[1])
    L = SVector{3,Float64}(bounds[2])
    delta = maximum(L) / (1 << Int(ell))
    function snapped_axis(a)
        la = clamp(ceil(Int, log2(L[a] / delta)), 0, Int(ell))
        while la < ell && delta * (1 << la) < L[a]
            la += 1
        end
        return delta * (1 << la)
    end
    snapped = SVector{3,Float64}(
        snapped_axis(1), snapped_axis(2), snapped_axis(3))
    center = x_min + L / 2
    return (center - snapped / 2, snapped)
end

"""
    _radix_auto_geometry(L, sigma_max, np, q_floor, rho_t, margin) -> (ell, q)

Task 035 cycle-1 joint depth/leaf-radius rule. Chooses the deepest radix
depth `ell` for which some supported leaf near radius `q >= q_floor`
satisfies the margin-guarded inequality
`g_min(q) * h_leaf >= margin * rho_t * sigma_max` (`h_leaf = L / 2^ell`),
capped by an occupancy heuristic of about `n^(1/3)` cells per side; at the
chosen depth the smallest passing `q` (cheapest direct near set) is used.
The margin buys regularization-deficit accuracy headroom over the bare
adequacy gate FastMultipole enforces (`margin = 1` reproduces adequacy-only
selection). Errors loudly when no depth `>= 2` is admissible.
"""
function _radix_auto_geometry(L::Real, sigma_max::Real, np::Int, q_floor::Int,
                              rho_t::Real, margin::Real)
    reach = margin * rho_t * sigma_max
    qs = sort!([Int(q) for q in fmm._SUPPORTED_RIGID_NEAR_RADII2 if q >= q_floor])
    isempty(qs) && error("near_radius2=$q_floor exceeds every supported rigid " *
        "near radius $(fmm._SUPPORTED_RIGID_NEAR_RADII2)")
    gaps = Dict(q => fmm._ball_stencil_min_gap(q) for q in qs)
    ell_occupancy = max(2, floor(Int, log2(max(np, 8)) / 3))
    for ell in ell_occupancy:-1:2
        h = L / 2^ell
        for q in qs
            gaps[q] * h >= reach && return (ell, q)
        end
    end
    error("no admissible radix depth (need ell >= 2): the margin-guarded " *
        "near-set inequality requires g_min(q)*L/2^ell >= " *
        "margin*rho_t*sigma_max = $reach, but even ell = 2 with the largest " *
        "supported q >= $q_floor gives $(maximum(gaps[q] for q in qs) * L / 4). " *
        "Reduce the smoothing overlap, enlarge the domain box, or use more " *
        "particles.")
end

################################################################################
# Cache construction and evaluation
################################################################################

function _build_radix_fmm_cache(pfield::ParticleField{R},
                                settings::RadixFMMSettings) where R
    _validate_radix_fmm_settings(pfield)
    device = !(pfield.particles isa Array)
    if device
        fmm.load_cuda_radix_lifecycle!() || error(
            "GPU FMM requested (CuArray-backed particle field) but the CUDA " *
            "radix lifecycle is unavailable: $(fmm.cuda_radix_status())")
    end

    bounds = settings.bounds === nothing ?
        _radix_derive_bounds(pfield, settings.padding;
            rectangular=settings.rectangular) : settings.bounds
    sigma_max = Float64(_radix_sigma_max(pfield))
    P = settings.expansion_order === nothing ? pfield.fmm.p - 1 :
        settings.expansion_order
    direct_kernel = _radix_direct_kernel(settings)
    kernel_primary_reach = _radix_primary_reach(direct_kernel)
    # The auto-geometry rule is shape-independent (task 037): L = max extent
    # drives ell and q via sigma-adequacy exactly as in cubic mode, so the leaf
    # width L/2^ell is identical — rectangularity only trims per-axis counts.
    L_geo = bounds[2] isa Real ? Float64(bounds[2]) :
        Float64(maximum(bounds[2]))
    ell, q = settings.ell === nothing ?
        _radix_auto_geometry(L_geo, sigma_max, pfield.np,
            settings.near_radius2, kernel_primary_reach,
            settings.accuracy_margin) :
        (settings.ell, settings.near_radius2)
    if settings.bounds === nothing && settings.rectangular
        bounds = _radix_center_snapped_bounds(bounds, ell)
    end
    TF = settings.precision === nothing ? R : settings.precision
    K = settings.window_classes === nothing ? (device ? 256 : nothing) :
        settings.window_classes
    m2l_strategy, operator = _radix_m2l_strategy(settings)
    opts = fmm.CUDARadixLifecycleOptions(; precision=TF, operator, m2l_strategy)

    # Capacity contract: sized once to maxparticles; live np may vary below it
    # (particles added/removed between steps) with no reallocation.
    return fmm.RadixFMMCache(pfield;
        expansion_order=P, ell,
        max_n_bodies=pfield.maxparticles,
        bounds=(SVector{3,TF}(bounds[1]),
            bounds[2] isa Real ? TF(bounds[2]) : SVector{3,TF}(bounds[2])),
        hessian=true, near_radius2=q,
        level_radii2=settings.level_radii2, window_classes=K,
        device, options=opts)
end

"""
    _radix_fmm_coupling!(pfield) -> (; cache, settings)

Get-or-create the persistent radix coupling for `pfield`. The cache is built
once (sized to `pfield.maxparticles`) and reused by every subsequent
evaluation; `clear_radix_fmm_cache!` or `radix_fmm_settings!` invalidate it.
"""
function _radix_fmm_coupling!(pfield::ParticleField)
    st = get(_radix_fmm_couplings, pfield, nothing)
    if st === nothing
        settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
        cache = _build_radix_fmm_cache(pfield, settings)
        st = (; cache, settings)
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
function _radix_fmm_evaluate!(pfield::ParticleField)
    st = _radix_fmm_coupling!(pfield)
    try
        fmm.fmm!(pfield, st.cache;
            scalar_potential=false, gradient=true, hessian=true)
    catch err
        (err isa ArgumentError && st.settings.bounds === nothing) || rethrow()
        # out-of-box (or other geometry) rejection: recenter and retry once;
        # a second failure (e.g. adequacy gate on the grown box) propagates
        bounds = _radix_derive_bounds(pfield, st.settings.padding;
            rectangular=st.settings.rectangular)
        st.settings.rectangular &&
            (bounds = _radix_center_snapped_bounds(bounds, st.cache.ell))
        fmm.recenter!(st.cache, pfield; bounds)
        fmm.fmm!(pfield, st.cache;
            scalar_potential=false, gradient=true, hessian=true)
    end
    return nothing
end

"""
    UJ_fmm_gpu!(pfield; reset=true, reset_sfs=false, sfs=false, rbf=false,
                verbose=false)

GPU/radix counterpart of `UJ_fmm` for a `CuArray`-backed `ParticleField`
(also runnable on a `Matrix`-backed field through FastMultipole's
transfer-based host path, used by the CPU-side tests). Computes U and J via
the resident radix FMM (device-to-device for a GPU field: no per-step body
transfers) and accumulates into `U_INDEX`/`J_INDEX`. Unsupported
configurations (`rbf`, `sfs`) fail loudly rather than silently dropping
physics.
"""
function UJ_fmm_gpu!(pfield::ParticleField;
        reset::Bool=true, reset_sfs::Bool=false, sfs::Bool=false,
        rbf::Bool=false, verbose::Bool=false, optargs...)
    rbf && error("rbf/zeta evaluation is not supported on the radix/GPU FMM " *
        "path (use UJ_direct/zeta_direct for CuArray-backed fields)")
    sfs && error("SFS (Estr_fmm) is not supported on the radix/GPU FMM path " *
        "yet; use SFS=NoSFS() with GPU-backed particle fields")
    reset && _reset_particles(pfield)
    reset_sfs && _reset_particles_sfs(pfield)
    _radix_fmm_evaluate!(pfield)
    return nothing
end

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

end # _FMM_HAS_RADIX
