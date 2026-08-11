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
# (task 035 tuning surface); the default remains the regularized-everywhere
# `RegularizedVortex` shipped by task 034.
function fmm.direct_kernel(pfield::ParticleField)
    is_gaussianerf(pfield.kernel) || error(
        "the radix/GPU FMM path supports only the `gaussianerf` kernel " *
        "(FLOWVPM default; sole CoreSpreading-compatible kernel). " *
        "Got a different `pfield.kernel`.")
    settings = get(_radix_fmm_settings, pfield, RadixFMMSettings())
    return _radix_direct_kernel(settings)
end

"Resolve the FastMultipole nearfield direct-kernel functor from settings."
function _radix_direct_kernel(settings)
    sym = settings.direct_kernel
    rho_t = settings.rho_t
    if sym === :regularized
        return rho_t === nothing ? fmm.RegularizedVortex(; sigma_row=8) :
            fmm.RegularizedVortex(; sigma_row=8, rho_t)
    elseif sym === :partitioned
        return rho_t === nothing ? fmm.PartitionedVortex(; sigma_row=8) :
            fmm.PartitionedVortex(; sigma_row=8, rho_t)
    elseif sym === :twopass
        return rho_t === nothing ? fmm.TwoPassVortex(; sigma_row=8) :
            fmm.TwoPassVortex(; sigma_row=8, rho_t)
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

- `expansion_order`: defaults to `pfield.fmm.p - 1` (literature `P = p`).
- `ell`: radix tree depth; defaults to the deepest depth passing both the
  near-set adequacy inequality `g_min*h_leaf > rho_t*sigma_max` and an
  occupancy cap `~n^(1/3)` cells per side.
- `near_radius2`: direct-stencil ball radius squared (16 raises the minimum
  M2L gap to sqrt(6), the task-032 accuracy-validated default).
- `window_classes`: M2L window classes; `nothing` = 256 on device, framework
  default on host.
- `padding`: per-face padding fraction applied to derived domain bounds
  (construction and automatic `recenter!`).
- `bounds`: explicit `(x_min::SVector{3}, box_size)` domain box. When set, the
  box is treated as user-owned: out-of-box particles error instead of
  triggering an automatic recenter.
- `precision`: lifecycle float type; defaults to `eltype(pfield)`.
- `direct_kernel`: nearfield strategy `:regularized` (default; the
  regularized-everywhere `RegularizedVortex`), `:partitioned`
  (`PartitionedVortex`, FastMultipole's measured 032a winner), or `:twopass`
  (`TwoPassVortex`).
- `rho_t`: override the nearfield kernel's smoothing-cutoff radius (defaults
  to each kernel's shipped constructor default).
- `m2l_strategy`: `:concat` (default, `ConcatenatedFixedZM2L`), `:dense`
  (`DenseTranslationM2L`), or `:precomputed_y` (`PrecomputedFactoredYM2L`).
- `level_radii2`: per-M2L-level near radii (levels `2:ell`, coarse to fine,
  non-increasing, ending at `near_radius2`); `nothing` = uniform
  `near_radius2` at every level.
"""
Base.@kwdef struct RadixFMMSettings
    expansion_order::Union{Nothing,Int} = nothing
    ell::Union{Nothing,Int} = nothing
    near_radius2::Int = 16
    window_classes::Union{Nothing,Int} = nothing
    padding::Float64 = 0.1
    bounds::Union{Nothing,Tuple} = nothing
    precision::Union{Nothing,DataType} = nothing
    direct_kernel::Symbol = :regularized
    rho_t::Union{Nothing,Float64} = nothing
    m2l_strategy::Symbol = :concat
    level_radii2::Union{Nothing,Tuple} = nothing
end

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
    _radix_derive_bounds(pfield, padding) -> (x_min::SVector{3}, box_size)

Cubic domain bounds covering the live particles, padded by `padding` of the
tight cube's side on each face (the `recenter!` convention). Degenerate
extents are inflated to `4*sigma_max` so a near-singleton field still yields
a valid box.
"""
function _radix_derive_bounds(pfield::ParticleField, padding::Real)
    lo1, hi1 = _radix_row_extrema(pfield, X_INDEX[1])
    lo2, hi2 = _radix_row_extrema(pfield, X_INDEX[2])
    lo3, hi3 = _radix_row_extrema(pfield, X_INDEX[3])
    span = max(hi1 - lo1, hi2 - lo2, hi3 - lo3)
    L_tight = max(span, 4 * _radix_sigma_max(pfield))
    L_tight > 0 || error("cannot derive radix FMM bounds: degenerate particle field")
    L = (1 + 2 * padding) * L_tight
    cx = (lo1 + hi1) / 2
    cy = (lo2 + hi2) / 2
    cz = (lo3 + hi3) / 2
    x_min = SVector{3,Float64}(cx - L / 2, cy - L / 2, cz - L / 2)
    return (x_min, Float64(L))
end

"""
    _radix_auto_ell(L, sigma_max, np, near_radius2, rho_t) -> ell

Deepest radix depth satisfying the near-set adequacy inequality
`2^ell < g_min * L / (rho_t * sigma_max)` (strict), capped by an occupancy
heuristic of about `n^(1/3)` cells per side. `rho_t` is the selected
nearfield kernel's smoothing-cutoff radius. Errors (loudly, naming the
measured ratio) when no depth `>= 2` is admissible — the same geometric gate
FastMultipole enforces at every regularized-kernel evaluation.
"""
function _radix_auto_ell(L::Real, sigma_max::Real, np::Int, near_radius2::Int,
                         rho_t::Real)
    g_min = fmm._ball_stencil_min_gap(near_radius2)
    x = g_min * L / (rho_t * sigma_max)
    ell_adequacy = floor(Int, log2(x))
    2.0^ell_adequacy < x || (ell_adequacy -= 1)   # strict inequality
    ell_occupancy = max(2, floor(Int, log2(max(np, 8)) / 3))
    ell = min(ell_adequacy, ell_occupancy)
    ell >= 2 || error("no admissible radix depth (need ell >= 2): the " *
        "near-set adequacy inequality requires 2^ell < g_min*L/(rho_t*sigma_max) " *
        "= $x. Reduce the smoothing overlap, enlarge the domain box, or use " *
        "more particles.")
    return ell
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
        _radix_derive_bounds(pfield, settings.padding) : settings.bounds
    sigma_max = Float64(_radix_sigma_max(pfield))
    P = settings.expansion_order === nothing ? pfield.fmm.p - 1 :
        settings.expansion_order
    q = settings.near_radius2
    kernel_rho_t = Float64(_radix_direct_kernel(settings).rho_t)
    ell = settings.ell === nothing ?
        _radix_auto_ell(bounds[2], sigma_max, pfield.np, q, kernel_rho_t) :
        settings.ell
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
        bounds=(SVector{3,TF}(bounds[1]), TF(bounds[2])),
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
        fmm.recenter!(st.cache, pfield;
            bounds=_radix_derive_bounds(pfield, st.settings.padding))
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
