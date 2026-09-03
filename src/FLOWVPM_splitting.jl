#=##############################################################################
# DESCRIPTION
    Particle subdivision (splitting). Mirrors the architecture of
    FLOWVPM_merging.jl: a workspace on `ParticleField`, a single pass over
    candidates, severity-ranked cap, in-place mutation of `pfield.particles`.

    See sfs_musings.md §Innovations / Particle subdivision for motivation.
    Default split is symmetric two-child, constant-σ, Γ/2 each, offset
    by ±a along `e_split` where 2a = κ_split · σ_p.

# AUTHORSHIP
  * Created   : 2026 (flowpanel branch)
=###############################################################################

# ------------------------------------------------------------------------------
# Local rates (P_p, Q_p, Z_p) used by triggers and the H_chi integrator
# ------------------------------------------------------------------------------

# Resolved stretching vector S_p depends on the `transposed` flag. Returns
# (Sx, Sy, Sz). Mirrors the index convention in _euler/_rungekutta3.
@inline function _stretching_S(J, G, transposed::Bool)
    if transposed
        return (J[1]*G[1] + J[2]*G[2] + J[3]*G[3],
                J[4]*G[1] + J[5]*G[2] + J[6]*G[3],
                J[7]*G[1] + J[8]*G[2] + J[9]*G[3])
    else
        return (J[1]*G[1] + J[4]*G[2] + J[7]*G[3],
                J[2]*G[1] + J[5]*G[2] + J[8]*G[3],
                J[3]*G[1] + J[6]*G[2] + J[9]*G[3])
    end
end

# Formulation-specific f, g for Z_p. ClassicVPM has Z_p = 0.
@inline _fg(::ClassicVPM) = (0.0, 0.0)
@inline _fg(form::ReformulatedVPM) = (form.f, form.g)

# Returns (P, Q, Z, normGamma2). Z is zero for ClassicVPM.
@inline function _compute_PQZ(pfield::ParticleField, i::Int)
    G = get_Gamma(pfield, i)
    J = get_J(pfield, i)
    Sx, Sy, Sz = _stretching_S(J, G, pfield.transposed)
    P = G[1]*Sx + G[2]*Sy + G[3]*Sz
    nG2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]

    sigma = pfield.particles[SIGMA_INDEX, i]
    C = get_C(pfield, i)
    E = get_SFS(pfield, i)
    zeta0 = pfield.kernel.zeta(0)
    # Q_p uses C_p[1] only — the integration loops also use C[1]
    Cs = C[1]
    Q = Cs * (G[1]*E[1] + G[2]*E[2] + G[3]*E[3]) * sigma^3 / zeta0

    if pfield.formulation isa ClassicVPM
        Z = zero(P)
    else
        f, g = _fg(pfield.formulation)
        denom = 1 + 3*f
        if nG2 > 0
            Z = ((f + g)/denom * P - f/denom * Q) / nG2
        else
            Z = zero(P)
        end
    end
    return P, Q, Z, nG2
end

# μ_{Γ,p} = (Γ · R_Γ) / ||Γ||² = (P - 3 Z ||Γ||² - Q) / ||Γ||²
@inline function _compute_mu_Gamma(pfield::ParticleField, i::Int)
    P, Q, Z, nG2 = _compute_PQZ(pfield, i)
    nG2 <= 0 && return zero(P)
    return (P - 3 * Z * nG2 - Q) / nG2
end

# Symmetric strain tensor S = (J + Jᵀ)/2 packed as SMatrix
@inline function _strain_tensor(J)
    s11 = J[1]
    s22 = J[5]
    s33 = J[9]
    s12 = (J[2] + J[4]) / 2
    s13 = (J[3] + J[7]) / 2
    s23 = (J[6] + J[8]) / 2
    return @SMatrix [s11 s12 s13; s12 s22 s23; s13 s23 s33]
end

# eᵀ S e for unit (or near-unit) e
@inline function _eSe(S, ex, ey, ez)
    return ex*(S[1,1]*ex + S[1,2]*ey + S[1,3]*ez) +
           ey*(S[2,1]*ex + S[2,2]*ey + S[2,3]*ez) +
           ez*(S[3,1]*ex + S[3,2]*ey + S[3,3]*ez)
end

# Leading (largest-eigenvalue) eigenvector of a symmetric 3x3.
# Returns (vx, vy, vz, lambda_max). Keep the eigensolve on StaticArrays so
# strain-axis paths stay allocation-free for ordinary floating-point fields.
@inline function _leading_eig_sym3(S::SMatrix{3,3,R}) where {R}
    es = eigen(Symmetric(S))
    # Sort eigenvalues; eigen on Symmetric returns ascending — last is largest
    λmax = es.values[3]
    V = es.vectors
    return V[1, 3], V[2, 3], V[3, 3], λmax
end

# Lazy unit-vector helpers; return (vx, vy, vz, ok)
@inline function _unit_strength(pfield, i)
    G = get_Gamma(pfield, i)
    n = sqrt(G[1]*G[1] + G[2]*G[2] + G[3]*G[3])
    n <= 0 && return (zero(n), zero(n), zero(n), false)
    inv_n = inv(n)
    return (G[1]*inv_n, G[2]*inv_n, G[3]*inv_n, true)
end

@inline function _unit_streamline(pfield, i)
    U = get_U(pfield, i)
    n = sqrt(U[1]*U[1] + U[2]*U[2] + U[3]*U[3])
    n <= 0 && return (zero(n), zero(n), zero(n), false)
    inv_n = inv(n)
    return (U[1]*inv_n, U[2]*inv_n, U[3]*inv_n, true)
end

# Filament axis unit vector for inference. :strength uses Γ̂; :strain1 uses
# the leading eigenvector of the symmetric strain tensor.
@inline function _filament_axis_unit(pfield, i, axis::Symbol)
    if axis === :strength
        return _unit_strength(pfield, i)
    elseif axis === :strain1
        J = get_J(pfield, i)
        S = _strain_tensor(J)
        vx, vy, vz, _ = _leading_eig_sym3(S)
        R = eltype(pfield.particles)
        n = sqrt(vx*vx + vy*vy + vz*vz)
        n <= 0 && return (zero(R), zero(R), zero(R), false)
        inv_n = inv(n)
        return (vx*inv_n, vy*inv_n, vz*inv_n, true)
    else
        error("Unknown filament axis $(axis); expected :strength or :strain1")
    end
end

# λ_χ(e) = eᵀ S e + Z_p
@inline function _lambda_chi_axis(pfield, i, axis::Symbol)
    J = get_J(pfield, i)
    S = _strain_tensor(J)
    _, _, Z, _ = _compute_PQZ(pfield, i)
    R = eltype(pfield.particles)
    if axis === :strength
        ex, ey, ez, ok = _unit_strength(pfield, i)
        ok || return zero(R)
        return _eSe(S, ex, ey, ez) + Z
    elseif axis === :streamline
        ex, ey, ez, ok = _unit_streamline(pfield, i)
        ok || return zero(R)
        return _eSe(S, ex, ey, ez) + Z
    elseif axis === :strain1
        _, _, _, λ = _leading_eig_sym3(S)
        return λ + Z
    elseif axis === :max
        # max over the three axes (skipping ones with zero norm)
        ex, ey, ez, okG = _unit_strength(pfield, i)
        ux, uy, uz, okU = _unit_streamline(pfield, i)
        _, _, _, λ1 = _leading_eig_sym3(S)
        m = λ1 + Z
        okG && (m = max(m, _eSe(S, ex, ey, ez) + Z))
        okU && (m = max(m, _eSe(S, ux, uy, uz) + Z))
        return m
    else
        error("Unknown H_chi_axis $(axis); expected :strength, :streamline, :strain1, or :max")
    end
end

# ------------------------------------------------------------------------------
# H_chi integrator hook — call once per accepted step
# ------------------------------------------------------------------------------
"""
    accumulate_H_chi!(pfield::ParticleField, dt::Real)

Advances the accumulated overlap-loss exposure `H_chi[i] += dt · λ_χ(e[i])`
on each non-static particle. The axis `e` is selected by `pfield.H_chi_axis`
(`:strength`, `:streamline`, `:strain1`, or `:max`); if
`pfield.H_chi_clip_positive` is true, only positive instantaneous rates are
accumulated. The hook is a no-op unless `pfield.track_H_chi == true`.

`H_chi` estimates `log(χ_p(t)/χ_p(t_0))` where χ = h/σ along the chosen
direction; thus `H_chi > log R_max` means χ has grown by more than the
dimensionless factor R_max since the counter was last reset.
"""
function accumulate_H_chi!(pfield::ParticleField, dt::Real)
    pfield.track_H_chi || return nothing
    axis = pfield.H_chi_axis
    clip = pfield.H_chi_clip_positive
    st = pfield.splitting_state
    for i in 1:get_np(pfield)
        get_static(pfield, i) && continue
        λ = _lambda_chi_axis(pfield, i, axis)
        if clip && λ < 0
            continue
        end
        st.H_chi[i] += dt * λ
    end
    return nothing
end

# ------------------------------------------------------------------------------
# Trigger types
# ------------------------------------------------------------------------------
"""
    SplitTrigger

Abstract supertype for particle-split triggers. Concrete triggers are
concretely typed so that the entire trigger tree dispatches statically.
"""
abstract type SplitTrigger end

"""
    GammaMagTrigger(c_Gamma)

Fires when `‖Γ_p‖ > c_Gamma`.
"""
struct GammaMagTrigger{R} <: SplitTrigger
    c_Gamma::R
end

@inline function should_split(t::GammaMagTrigger, pfield, state, i, dt)
    G = get_Gamma(pfield, i)
    return sqrt(G[1]*G[1] + G[2]*G[2] + G[3]*G[3]) > t.c_Gamma
end

@inline function severity(t::GammaMagTrigger, pfield, state, i, dt)
    G = get_Gamma(pfield, i)
    n = sqrt(G[1]*G[1] + G[2]*G[2] + G[3]*G[3])
    return max(zero(n), n - t.c_Gamma)
end

"""
    ZTrigger(c_Z)

Fires when `dt · |Z_p| > c_Z`. Z_p is the rVPM compensation rate.
"""
struct ZTrigger{R} <: SplitTrigger
    c_Z::R
end

@inline function should_split(t::ZTrigger, pfield, state, i, dt)
    _, _, Z, _ = _compute_PQZ(pfield, i)
    return dt * abs(Z) > t.c_Z
end

@inline function severity(t::ZTrigger, pfield, state, i, dt)
    _, _, Z, _ = _compute_PQZ(pfield, i)
    return max(zero(Z), dt * abs(Z) - t.c_Z)
end

"""
    StretchTrigger(c_mu)

Fires when the projected strength-growth exposure `dt · μ_{Γ,p} > c_mu`.
"""
struct StretchTrigger{R} <: SplitTrigger
    c_mu::R
end

@inline function should_split(t::StretchTrigger, pfield, state, i, dt)
    μ = _compute_mu_Gamma(pfield, i)
    return dt * μ > t.c_mu
end

@inline function severity(t::StretchTrigger, pfield, state, i, dt)
    μ = _compute_mu_Gamma(pfield, i)
    return max(zero(μ), dt * μ - t.c_mu)
end

"""
    SeparationTrigger(log_R_max, axis::Symbol)

Fires when the accumulated overlap-loss exposure `H_chi[i] > log_R_max`.
This is the time-integrated form: `H_chi[i] ≈ log(χ_p(t)/χ_p(t_0))` along
the axis selected by `pfield.H_chi_axis`. The integrator hook
`accumulate_H_chi!` (called from `nextstep`) must be active —
`pfield.track_H_chi` is set to `true` and `pfield.H_chi_axis` is set
to `axis` when a `SeparationTrigger` is found in the trigger tree on
the first call to `split_particles!`.
"""
struct SeparationTrigger{R} <: SplitTrigger
    log_R_max::R
    axis::Symbol
end

SeparationTrigger(log_R_max::R) where {R<:Real} = SeparationTrigger(log_R_max, :strength)

@inline function should_split(t::SeparationTrigger, pfield, state, i, dt)
    return state.H_chi[i] > t.log_R_max
end

@inline function severity(t::SeparationTrigger, pfield, state, i, dt)
    return max(zero(t.log_R_max), state.H_chi[i] - t.log_R_max)
end

"""
    SigmaShrinkTrigger(C_sigma)

Fires when `σ_p / σ_{0,p} < C_sigma`, i.e. the particle has shrunk by
more than `C_sigma`-fraction of its creation radius.
"""
struct SigmaShrinkTrigger{R} <: SplitTrigger
    C_sigma::R
end

@inline function should_split(t::SigmaShrinkTrigger, pfield, state, i, dt)
    s0 = state.sigma_0[i]
    s0 > 0 || return false
    σ = pfield.particles[SIGMA_INDEX, i]
    return σ / s0 < t.C_sigma
end

@inline function severity(t::SigmaShrinkTrigger, pfield, state, i, dt)
    s0 = state.sigma_0[i]
    s0 > 0 || return zero(t.C_sigma)
    σ = pfield.particles[SIGMA_INDEX, i]
    return max(zero(σ), t.C_sigma - σ / s0)
end

"""
    HoldTrigger(inner, N_hold)

Wraps an inner trigger so it fires only after `N_hold` consecutive
accepted-step calls in which the inner trigger was true. Reads and
mutates `state.hold_counter[i]` on each call.
"""
struct HoldTrigger{T<:SplitTrigger} <: SplitTrigger
    inner::T
    N_hold::Int
end

@inline function should_split(t::HoldTrigger, pfield, state, i, dt)
    if should_split(t.inner, pfield, state, i, dt)
        state.hold_counter[i] += 1
        return state.hold_counter[i] >= t.N_hold
    else
        state.hold_counter[i] = 0
        return false
    end
end

@inline severity(t::HoldTrigger, pfield, state, i, dt) =
    severity(t.inner, pfield, state, i, dt)

# Composites — concretely typed via tuple
"""
    AllTrigger(triggers::Tuple)

Composite trigger that fires only when every sub-trigger fires.
"""
struct AllTrigger{T<:Tuple} <: SplitTrigger
    triggers::T
end
AllTrigger(triggers::SplitTrigger...) = AllTrigger(triggers)

"""
    AnyTrigger(triggers::Tuple)

Composite trigger that fires when at least one sub-trigger fires.
Sub-triggers are evaluated in order; **all** are evaluated so that
stateful triggers (e.g. `HoldTrigger`) update their counters consistently
regardless of short-circuit order.
"""
struct AnyTrigger{T<:Tuple} <: SplitTrigger
    triggers::T
end
AnyTrigger(triggers::SplitTrigger...) = AnyTrigger(triggers)

# Recursive helpers that unroll over the tuple at compile time.
@inline _all_split(::Tuple{}, pfield, state, i, dt) = true
@inline _all_split(ts::Tuple, pfield, state, i, dt) =
    should_split(ts[1], pfield, state, i, dt) && _all_split(Base.tail(ts), pfield, state, i, dt)

@inline _any_split(::Tuple{}, pfield, state, i, dt, acc::Bool) = acc
@inline _any_split(ts::Tuple, pfield, state, i, dt, acc::Bool) =
    _any_split(Base.tail(ts), pfield, state, i, dt,
               acc | should_split(ts[1], pfield, state, i, dt))

@inline should_split(t::AllTrigger, pfield, state, i, dt) =
    _all_split(t.triggers, pfield, state, i, dt)
@inline should_split(t::AnyTrigger, pfield, state, i, dt) =
    _any_split(t.triggers, pfield, state, i, dt, false)

# Composite severity = sum of children's severities (a reasonable ranking)
@inline _sev_sum(::Tuple{}, pfield, state, i, dt, acc) = acc
@inline _sev_sum(ts::Tuple, pfield, state, i, dt, acc) =
    _sev_sum(Base.tail(ts), pfield, state, i, dt,
             acc + severity(ts[1], pfield, state, i, dt))

@inline severity(t::AllTrigger, pfield, state, i, dt) =
    _sev_sum(t.triggers, pfield, state, i, dt, zero(eltype(pfield.particles)))
@inline severity(t::AnyTrigger, pfield, state, i, dt) =
    _sev_sum(t.triggers, pfield, state, i, dt, zero(eltype(pfield.particles)))

# ------------------------------------------------------------------------------
# Split-direction selection
# ------------------------------------------------------------------------------
"""
    SplitDirection

Specifies the axis along which a candidate particle is split.
`STRENGTH` uses `Γ_p/‖Γ_p‖` (default; aligned with vortex-tube stretching),
`STREAMLINE` uses `U_p/‖U_p‖`, and `STRAIN1` uses the leading eigenvector
of `S = (J+Jᵀ)/2`.
"""
@enum SplitDirection STRENGTH STREAMLINE STRAIN1

# Returns (ex, ey, ez, ok). ok=false ⇒ caller should skip the particle.
@inline function compute_split_direction(dir::SplitDirection, pfield, i)
    if dir == STRENGTH
        return _unit_strength(pfield, i)
    elseif dir == STREAMLINE
        return _unit_streamline(pfield, i)
    else  # STRAIN1
        J = get_J(pfield, i)
        S = _strain_tensor(J)
        vx, vy, vz, _ = _leading_eig_sym3(S)
        # Normalize defensively (StaticArrays eigen returns unit vectors,
        # but allow for ForwardDiff/dual-number drift)
        n = sqrt(vx*vx + vy*vy + vz*vz)
        n <= 0 && return (zero(n), zero(n), zero(n), false)
        inv_n = inv(n)
        return (vx*inv_n, vy*inv_n, vz*inv_n, true)
    end
end

# ------------------------------------------------------------------------------
# Split options
# ------------------------------------------------------------------------------
"""
    SplitOptions(; trigger, direction=STRENGTH, kappa_split=1.0,
                   preserve_sigma=true, skip_static=true,
                   max_fraction=0.05, N_cooldown=0)

Settings passed to `split_particles!`. `trigger` is required and must
be a `SplitTrigger`. `direction` selects the split axis. `kappa_split`
sets the child offset `2a = kappa_split · σ_p`. With `preserve_sigma=true`
children inherit the parent's `σ` (constant-σ subdivision, recommended);
otherwise `σ_c² = σ_p² − a²` (moment-match). `max_fraction` caps the
number of splits per call to `⌊max_fraction · N_p⌋`. `N_cooldown` is the
number of subsequent `split_particles!` calls during which the children
(and the in-place-replaced parent slot) are not eligible to be split
again.
"""
struct SplitOptions{T<:SplitTrigger, R}
    trigger::T
    direction::SplitDirection
    kappa_split::R
    preserve_sigma::Bool
    skip_static::Bool
    max_fraction::R
    N_cooldown::Int
end

function SplitOptions(; trigger::T,
                       direction::SplitDirection=STRENGTH,
                       kappa_split::Real=1.0,
                       preserve_sigma::Bool=true,
                       skip_static::Bool=true,
                       max_fraction::Real=0.05,
                       N_cooldown::Int=0) where {T<:SplitTrigger}
    R = promote_type(typeof(float(kappa_split)), typeof(float(max_fraction)))
    return SplitOptions{T, R}(trigger, direction, R(kappa_split),
                              preserve_sigma, skip_static,
                              R(max_fraction), N_cooldown)
end

# ------------------------------------------------------------------------------
# H_chi tracking auto-enable
# ------------------------------------------------------------------------------
# Recursively walks the trigger tree looking for a SeparationTrigger and, if
# found, sets pfield.track_H_chi = true and pfield.H_chi_axis = axis.
# Idempotent and cheap (compile-time unrolled).
@inline _has_sep(t::SeparationTrigger) = (true, t.axis)
@inline _has_sep(t::SplitTrigger) = (false, :strength)
@inline _has_sep(t::HoldTrigger) = _has_sep(t.inner)

@inline _scan_tuple(::Tuple{}) = (false, :strength)
@inline function _scan_tuple(ts::Tuple)
    h, ax = _has_sep(ts[1])
    h && return (true, ax)
    return _scan_tuple(Base.tail(ts))
end
@inline _has_sep(t::AllTrigger) = _scan_tuple(t.triggers)
@inline _has_sep(t::AnyTrigger) = _scan_tuple(t.triggers)

function _maybe_enable_H_chi!(pfield, trigger)
    has, axis = _has_sep(trigger)
    if has
        pfield.track_H_chi = true
        pfield.H_chi_axis = axis
    end
    return nothing
end

# ------------------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------------------
"""
    split_particles!(pfield::ParticleField, opts::SplitOptions; dt::Real, verbose::Bool=false)

Subdivide candidate particles into two children each, in place. Returns
the number of particles split.

A particle `i` is considered if (a) it is not static (when
`opts.skip_static`), (b) its `cooldown_counter` is zero, and (c)
`opts.trigger` fires. Among the surviving candidates, at most
`⌊opts.max_fraction · N_p⌋` are split, ranked by trigger severity
(highest first). Each split places child A in slot `i` (overwriting
the parent) and appends child B via `add_particle`. By default both
children have `Γ_parent / 2`, are offset by `±a · e_split` with
`2a = opts.kappa_split · σ_p`, and share the parent's `σ`. Resolved
field state (U, J, SFS, M, PSE, vorticity, U_prev, C) is zeroed on
both children so that noisy parent transients are not inherited.
"""
function split_particles!(pfield::ParticleField,
                          opts::SplitOptions;
                          dt::Real, verbose::Bool=false)
    np = get_np(pfield)
    np == 0 && return 0

    _maybe_enable_H_chi!(pfield, opts.trigger)

    ws = pfield.splitting_workspace
    state = pfield.splitting_state
    R = eltype(pfield.particles)

    empty!(ws.candidate_indices)
    empty!(ws.severity)

    # Pass 1: identify candidates and score them
    for i in 1:np
        if opts.skip_static && get_static(pfield, i)
            continue
        end
        if state.cooldown_counter[i] > 0
            state.cooldown_counter[i] -= 1
            continue
        end
        should_split(opts.trigger, pfield, state, i, dt) || continue
        push!(ws.candidate_indices, i)
        push!(ws.severity, R(severity(opts.trigger, pfield, state, i, dt)))
    end

    isempty(ws.candidate_indices) && return 0

    # Rank by severity (descending) and cap
    n_cap = max(0, floor(Int, opts.max_fraction * np))
    n_cap = min(n_cap, length(ws.candidate_indices))
    n_cap == 0 && return 0

    resize!(ws.order, length(ws.candidate_indices))
    @inbounds for k in 1:length(ws.candidate_indices)
        ws.order[k] = k
    end
    sort!(ws.order; by=k -> ws.severity[k], rev=true)

    # Pass 2: perform splits up to the cap, guarding maxparticles
    n_split = 0
    @inbounds for j in 1:n_cap
        get_np(pfield) >= pfield.maxparticles && break
        k = ws.order[j]
        i = ws.candidate_indices[k]
        ex, ey, ez, ok = compute_split_direction(opts.direction, pfield, i)
        ok || continue
        _do_split!(pfield, state, i, ex, ey, ez, opts)
        n_split += 1
    end

    verbose && n_split > 0 && println(
        "Split $n_split particles; np now $(get_np(pfield))")

    return n_split
end

# In-place subdivision of slot `i`:
# - parent's slot becomes child A at x - a e
# - child B appended via add_particle at x + a e
function _do_split!(pfield::ParticleField{R},
                    state, i::Int,
                    ex::Real, ey::Real, ez::Real,
                    opts::SplitOptions) where {R}
    σ = pfield.particles[SIGMA_INDEX, i]
    a = (opts.kappa_split * σ) / 2

    # Child σ policy
    σ_c = σ
    if !opts.preserve_sigma
        # Moment-match alternative: σ_c² = σ_p² − a²
        σ_c2 = σ*σ - a*a
        σ_c = σ_c2 > 0 ? sqrt(σ_c2) : σ
    end

    # Parent quantities
    x0 = pfield.particles[X_INDEX.start,     i]
    y0 = pfield.particles[X_INDEX.start + 1, i]
    z0 = pfield.particles[X_INDEX.start + 2, i]
    gx = pfield.particles[GAMMA_INDEX.start,     i]
    gy = pfield.particles[GAMMA_INDEX.start + 1, i]
    gz = pfield.particles[GAMMA_INDEX.start + 2, i]
    vol = pfield.particles[VOL_INDEX, i]
    circ = pfield.particles[CIRCULATION_INDEX, i]
    is_stat = get_static(pfield, i)

    # Child positions
    xA = x0 - a*ex; yA = y0 - a*ey; zA = z0 - a*ez
    xB = x0 + a*ex; yB = y0 + a*ey; zB = z0 + a*ez

    # Symmetric two-child split: Γ/2 each, vol/2 each, circ preserved per child
    halfgx = gx/2; halfgy = gy/2; halfgz = gz/2
    half_vol = vol/2

    # ---- Child A overwrites slot i ----
    set_X(pfield, i, (xA, yA, zA))
    set_Gamma(pfield, i, (halfgx, halfgy, halfgz))
    set_sigma(pfield, i, σ_c)
    set_vol(pfield, i, half_vol)
    set_circulation(pfield, i, circ)

    # Zero out resolved-field state so child does not inherit noisy parent
    # transients; recomputed on the next UJ pass.
    zeroR = zero(R)
    set_U(pfield, i, zeroR)
    set_vorticity(pfield, i, zeroR)
    set_J(pfield, i, zeroR)
    set_PSE(pfield, i, zeroR)
    set_M(pfield, i, zeroR)
    set_C(pfield, i, zeroR)
    set_SFS(pfield, i, zeroR)
    set_U_prev(pfield, i, zeroR)
    set_static(pfield, i, Float64(is_stat))

    # Reset splitting-state for parent slot to a fresh child
    state.sigma_0[i] = σ_c
    state.H_chi[i] = zeroR
    state.hold_counter[i] = 0
    state.cooldown_counter[i] = opts.N_cooldown
    # Children start a fresh Δσ² attribution history — the routing decision
    # they were born from has been consumed.
    state.dsigma2_visc[i] = zeroR
    state.dsigma2_rvpm[i] = zeroR

    # ---- Child B appended via add_particle ----
    add_particle(pfield, (xB, yB, zB), (halfgx, halfgy, halfgz), σ_c;
                 vol=half_vol, circulation=circ, C=zeroR, static=is_stat)
    j = get_np(pfield)
    # add_particle initialized sigma_0=σ_c, H_chi=0, counters=0;
    # apply cooldown to the freshly appended child too.
    state.cooldown_counter[j] = opts.N_cooldown

    return nothing
end

# ------------------------------------------------------------------------------
# Filament edge graph storage helpers (Phase 2 — adjacency-only)
# ------------------------------------------------------------------------------
# Bounded local-work mutators for `FilamentEdgeGraph`. No edge ids: an edge
# is the matched pair of slots (down_neighbor[k, src] == dst,
# up_neighbor[k′, dst] == src). Per-edge state (coherent, score) is stored
# once on the canonical downstream slot. Phase 3+ refinement/inference will
# layer score-based replacement, validation, and the cell-list candidate
# scan on top of these primitives.

"""
    add_edge!(graph, src, dst; coherent=false, score=0) -> Bool

Insert a directed edge `src → dst` into the bounded edge graph. Returns
`true` on success, `false` on rejection. Rejection conditions
(all bounded local checks):

- `src == dst` (self-loop).
- duplicate: `(src, dst)` is already present.
- degree cap: `down_count(src) == 2` or `up_count(dst) == 2`.

On success the new edge occupies the first empty slot in
`down_neighbor[:, src]` and `up_neighbor[:, dst]`; `coherent` and `score`
are stored at the same `(k, src)` index as the downstream slot.
"""
function add_edge!(graph::FilamentEdgeGraph{R}, src::Int, dst::Int;
                   coherent::Bool=false, score::Real=zero(R)) where {R}
    src == dst && return false
    find_down_slot(graph, src, dst) != 0 && return false
    down_count(graph, src) >= 2 && return false
    up_count(graph, dst) >= 2 && return false

    # First-empty slot on the down side of src
    k_down = graph.down_neighbor[1, src] == 0 ? 1 : 2
    graph.down_neighbor[k_down, src] = dst
    graph.down_coherent[k_down, src] = coherent
    graph.down_score[k_down, src]    = R(score)
    set_down_count!(graph, src, down_count(graph, src) + 1)

    # Mirror back-pointer on up side of dst
    k_up = graph.up_neighbor[1, dst] == 0 ? 1 : 2
    graph.up_neighbor[k_up, dst] = src
    set_up_count!(graph, dst, up_count(graph, dst) + 1)

    return true
end

"""
    remove_edge!(graph, src, dst) -> Bool

Remove the directed edge `src → dst`. Returns `true` if the edge existed
and was cleared, `false` otherwise. Removal compacts each local two-slot
adjacency list so live slots remain packed in `1:count`: if the removed
slot is not the last live slot, the last live edge is moved down into it.
On the downstream side, coherence and score move with that edge. Bounded
local work (≤2 slot scans per side).
"""
function remove_edge!(graph::FilamentEdgeGraph{R}, src::Int, dst::Int) where {R}
    k_down = find_down_slot(graph, src, dst)
    k_down == 0 && return false
    dcount = down_count(graph, src)
    if k_down < dcount
        graph.down_neighbor[k_down, src] = graph.down_neighbor[dcount, src]
        graph.down_coherent[k_down, src] = graph.down_coherent[dcount, src]
        graph.down_score[k_down, src]    = graph.down_score[dcount, src]
    end
    graph.down_neighbor[dcount, src] = 0
    graph.down_coherent[dcount, src] = false
    graph.down_score[dcount, src]    = zero(R)
    set_down_count!(graph, src, dcount - 1)

    k_up = find_up_slot(graph, dst, src)
    if k_up != 0
        ucount = up_count(graph, dst)
        if k_up < ucount
            graph.up_neighbor[k_up, dst] = graph.up_neighbor[ucount, dst]
        end
        graph.up_neighbor[ucount, dst] = 0
        set_up_count!(graph, dst, ucount - 1)
    end

    return true
end

"""
    clear_edges!(graph)

Bulk reset: zeroes every adjacency slot, coherence bit, score entry, and
degree byte. Leaves `filament_id` untouched (explicit identity may still
be wanted after a wipe). Primarily for tests and re-inference scenarios.
"""
function clear_edges!(graph::FilamentEdgeGraph{R}) where {R}
    fill!(graph.up_neighbor, 0)
    fill!(graph.down_neighbor, 0)
    fill!(graph.down_coherent, false)
    fill!(graph.down_score, zero(R))
    fill!(graph.degree, UInt8(0))
    return nothing
end

"""
    FilamentEdgeReport

Result of [`validate_filament_edges`](@ref). `ok == true` iff every counter
is zero. Each torn edge fires exactly once because the broken side has no
matching mirror entry to revisit.
"""
struct FilamentEdgeReport
    ok::Bool
    mirror_mismatches::Int
    count_mismatches::Int
    self_loops::Int
    duplicate_locals::Int
    invalid_endpoints::Int
    stale_inactive_slots::Int
end

"""
    validate_filament_edges(pfield; on_issue=nothing) -> FilamentEdgeReport

Walk every particle's bounded adjacency once (≤2 down + ≤2 up slots) and
diagnose structural inconsistencies. Total cost O(N) with O(E) edge
checks (E ≤ 4N).

Issue kinds:
- `:mirror_mismatch` — slot points at a neighbor whose mirror slot is
  missing.
- `:count_mismatch` — packed degree disagrees with the number of nonzero
  slots, or an active slot is zero.
- `:self_loop` — slot points at the owning particle.
- `:duplicate_local` — both slots on the same side hold the same neighbor.
- `:invalid_endpoint` — neighbor index is outside `1:np`.
- `:stale_inactive_slot` — slot index above the live count is nonzero.

When `on_issue !== nothing`, the callback is invoked as
`on_issue(kind::Symbol, src::Int, slot::Int, dst::Int, direction::Symbol)`
for each detected issue, where `direction` is `:down` or `:up`. `slot == 0`
for the live-count parity check.
"""
function validate_filament_edges(pfield::ParticleField{R}; on_issue=nothing) where {R}
    graph = pfield.filament_edge_graph
    np = get_np(pfield)
    mirror = 0; count_m = 0; selfl = 0; dup = 0; inv = 0; stale = 0

    @inbounds for p in 1:np
        # Downstream slots
        dcount = down_count(graph, p)
        live_d = 0
        for k in 1:2
            q = graph.down_neighbor[k, p]
            if k > dcount
                if q != 0
                    stale += 1
                    on_issue === nothing || on_issue(:stale_inactive_slot, p, k, q, :down)
                end
                continue
            end
            if q == 0
                count_m += 1
                on_issue === nothing || on_issue(:count_mismatch, p, k, q, :down)
                continue
            end
            live_d += 1
            if q == p
                selfl += 1
                on_issue === nothing || on_issue(:self_loop, p, k, q, :down)
                continue
            end
            if q < 1 || q > np
                inv += 1
                on_issue === nothing || on_issue(:invalid_endpoint, p, k, q, :down)
                continue
            end
            if k == 2 && q == graph.down_neighbor[1, p]
                dup += 1
                on_issue === nothing || on_issue(:duplicate_local, p, k, q, :down)
                continue
            end
            if find_up_slot(graph, q, p) == 0
                mirror += 1
                on_issue === nothing || on_issue(:mirror_mismatch, p, k, q, :down)
            end
        end
        if live_d != dcount
            count_m += 1
            on_issue === nothing || on_issue(:count_mismatch, p, 0, 0, :down)
        end

        # Upstream slots
        ucount = up_count(graph, p)
        live_u = 0
        for k in 1:2
            q = graph.up_neighbor[k, p]
            if k > ucount
                if q != 0
                    stale += 1
                    on_issue === nothing || on_issue(:stale_inactive_slot, p, k, q, :up)
                end
                continue
            end
            if q == 0
                count_m += 1
                on_issue === nothing || on_issue(:count_mismatch, p, k, q, :up)
                continue
            end
            live_u += 1
            if q == p
                selfl += 1
                on_issue === nothing || on_issue(:self_loop, p, k, q, :up)
                continue
            end
            if q < 1 || q > np
                inv += 1
                on_issue === nothing || on_issue(:invalid_endpoint, p, k, q, :up)
                continue
            end
            if k == 2 && q == graph.up_neighbor[1, p]
                dup += 1
                on_issue === nothing || on_issue(:duplicate_local, p, k, q, :up)
                continue
            end
            if find_down_slot(graph, q, p) == 0
                mirror += 1
                on_issue === nothing || on_issue(:mirror_mismatch, p, k, q, :up)
            end
        end
        if live_u != ucount
            count_m += 1
            on_issue === nothing || on_issue(:count_mismatch, p, 0, 0, :up)
        end
    end

    ok = (mirror == 0) & (count_m == 0) & (selfl == 0) & (dup == 0) & (inv == 0) & (stale == 0)
    return FilamentEdgeReport(ok, mirror, count_m, selfl, dup, inv, stale)
end

"""
    repair_filament_edges!(pfield) -> FilamentEdgeReport

Conservative repair: clear `down_coherent` on every edge slot
[`validate_filament_edges`](@ref) reports as suspect. Split/coarsen passes
gate on coherence, so this is enough to keep them from acting on torn
topology. The next [`infer_filament_edges!`](@ref) call refreshes the
region.

Does not mutate `up_neighbor`, `down_neighbor`, `degree`, or `down_score`
in this phase — structural mirror breakage that is not self-healing
remains visible in the returned report so calibration can decide whether
to escalate to `remove_edge!`.
"""
function repair_filament_edges!(pfield::ParticleField{R}) where {R}
    graph = pfield.filament_edge_graph
    np = get_np(pfield)
    return validate_filament_edges(pfield;
        on_issue = (kind, src, slot, dst, direction) -> begin
            if direction === :down
                if 1 <= slot <= 2
                    graph.down_coherent[slot, src] = false
                end
            else
                if 1 <= dst <= np
                    k = find_down_slot(graph, dst, src)
                    if k != 0
                        graph.down_coherent[k, dst] = false
                    end
                end
            end
            return nothing
        end)
end

# Center-out 27-cell stencil offsets: own cell, 6 face neighbors, 12 edge
# neighbors, 8 corner neighbors. Iteration in this order biases the visit
# cap toward closer cells.
const STENCIL_OFFSETS_27 = (
    ( 0, 0, 0),
    ( 1, 0, 0), (-1, 0, 0), ( 0, 1, 0), ( 0,-1, 0), ( 0, 0, 1), ( 0, 0,-1),
    ( 1, 1, 0), ( 1,-1, 0), (-1, 1, 0), (-1,-1, 0),
    ( 1, 0, 1), ( 1, 0,-1), (-1, 0, 1), (-1, 0,-1),
    ( 0, 1, 1), ( 0, 1,-1), ( 0,-1, 1), ( 0,-1,-1),
    ( 1, 1, 1), ( 1, 1,-1), ( 1,-1, 1), ( 1,-1,-1),
    (-1, 1, 1), (-1, 1,-1), (-1,-1, 1), (-1,-1,-1),
)

# Insert (q, score) into a length-2 best buffer at column p. Replaces the
# weaker entry if the buffer is full and score beats the minimum. Bounded
# local work.
@inline function _best2_insert!(best::Matrix{Int}, best_score::Matrix{R},
                                p::Int, q::Int, score::R) where {R}
    b1 = best[1, p]; s1 = best_score[1, p]
    b2 = best[2, p]; s2 = best_score[2, p]
    if b1 == 0
        best[1, p] = q; best_score[1, p] = score
    elseif b2 == 0
        best[2, p] = q; best_score[2, p] = score
    elseif s1 <= s2
        if score > s1
            best[1, p] = q; best_score[1, p] = score
        end
    else
        if score > s2
            best[2, p] = q; best_score[2, p] = score
        end
    end
    return nothing
end

"""
    infer_filament_edges!(pfield; max_eta=2.0, angle_tol=π/4,
                          axis=:strength, candidate_cap=16) -> Int

Add inferred filament edges to `pfield.filament_edge_graph` using a
cell-list scan with symmetric per-pair updates and a mutual-NN commit.

Algorithm:
1. **Gather pass.** For each non-static particle with a well-defined
   filament axis (`_filament_axis_unit`), cache the axis unit and track
   bounding box plus `σ_max`.
2. **Bin pass.** Build a 3D cell list with `cell_size = max_eta * σ_max`
   via `_build_cell_list!`.
3. **Scoring pass.** For each particle `p`, walk the 27-cell stencil
   center-out. For each candidate `q` with `q > p` (per-pair dedupe),
   increment `workspace.visits[p]` *before* the η/cos θ gates. If
   `visits[p] > candidate_cap`, set `workspace.capped[p]` and break.
   Otherwise apply the gates (η ≤ max_eta, cos θ ≥ cos angle_tol) and,
   on success, symmetrically insert into both endpoints' best-2 buffers
   (downstream/upstream determined by the sign of the projection).
4. **Mutual-best commit.** For each `q ∈ p.down_best`, insert edge
   `p → q` via `add_edge!` only if `p ∈ q.up_best`. Non-destructive:
   the existing degree-cap rejection in `add_edge!` means inference
   never displaces an explicit edge.

The cap counts *visits*, so the work per particle is bounded regardless
of pass/fail rates. Symmetric updates close the mutual-NN asymmetry: a
pair found by `p`'s walk also appears in `q`'s best buffer without
requiring `q`'s walk to reach `p`.

Returns the number of edges added.
"""
function infer_filament_edges!(pfield::ParticleField{R};
                               max_eta::Real      = 2.0,
                               angle_tol::Real    = π/4,
                               axis::Symbol       = :strength,
                               candidate_cap::Int = 16) where {R}
    np = get_np(pfield)
    np < 2 && return 0

    graph = pfield.filament_edge_graph
    ws    = pfield.filament_edge_workspace

    # Resize workspace
    resize!(ws.axis_x, np); resize!(ws.axis_y, np); resize!(ws.axis_z, np)
    resize!(ws.axis_ok, np); fill!(ws.axis_ok, false)
    resize!(ws.visits, np); fill!(ws.visits, 0)
    resize!(ws.capped, np); fill!(ws.capped, false)
    if size(ws.down_best, 2) < np
        ws.down_best      = zeros(Int, 2, np)
        ws.up_best        = zeros(Int, 2, np)
        ws.down_best_score = zeros(R, 2, np)
        ws.up_best_score   = zeros(R, 2, np)
    else
        # Zero only the slice we will write
        @inbounds for i in 1:np
            ws.down_best[1, i] = 0; ws.down_best[2, i] = 0
            ws.up_best[1, i]   = 0; ws.up_best[2, i]   = 0
            ws.down_best_score[1, i] = zero(R); ws.down_best_score[2, i] = zero(R)
            ws.up_best_score[1, i]   = zero(R); ws.up_best_score[2, i]   = zero(R)
        end
    end

    # ---------------------------------------------------------------- gather
    candidate_indices = ws.candidates
    empty!(candidate_indices)

    xmin = ymin = zmin =  Inf
    sigma_max = zero(R)

    @inbounds for i in 1:np
        get_static(pfield, i) && continue
        ex, ey, ez, ok = _filament_axis_unit(pfield, i, axis)
        ok || continue
        ws.axis_x[i] = ex; ws.axis_y[i] = ey; ws.axis_z[i] = ez
        ws.axis_ok[i] = true

        push!(candidate_indices, i)

        x = pfield.particles[1, i]
        y = pfield.particles[2, i]
        z = pfield.particles[3, i]
        sigma = pfield.particles[SIGMA_INDEX, i]

        xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z)
        sigma_max = max(sigma_max, R(sigma))
    end

    length(candidate_indices) < 2 && return 0
    sigma_max > 0 || return 0

    # ------------------------------------------------------------------- bin
    cell_size = R(max_eta) * sigma_max
    resize!(ws.keys, np)

    origin = (xmin, ymin, zmin)
    n_cells = _build_cell_list!(ws.sorted_indices, ws.offsets, ws.counts, ws.keys,
                                candidate_indices, pfield, cell_size, origin)

    # --------------------------------------------------------------- scoring
    inv_cell = inv(cell_size)
    cos_tol  = R(cos(angle_tol))
    max_eta_R = R(max_eta)

    @inbounds for p in candidate_indices
        ws.axis_ok[p] || continue
        xp = pfield.particles[1, p]
        yp = pfield.particles[2, p]
        zp = pfield.particles[3, p]
        sp = R(pfield.particles[SIGMA_INDEX, p])
        ex_p = ws.axis_x[p]; ey_p = ws.axis_y[p]; ez_p = ws.axis_z[p]

        ix = _cell_coord(xp - xmin, inv_cell)
        iy = _cell_coord(yp - ymin, inv_cell)
        iz = _cell_coord(zp - zmin, inv_cell)

        capped_p = false
        for (dx, dy, dz) in STENCIL_OFFSETS_27
            capped_p && break
            jx = ix + dx; jy = iy + dy; jz = iz + dz
            (jx < 0 || jx > CELL_COORD_MAX) && continue
            (jy < 0 || jy > CELL_COORD_MAX) && continue
            (jz < 0 || jz > CELL_COORD_MAX) && continue
            key = _pack_cell_key(jx, jy, jz)
            for slot in _cell_range(ws.offsets, ws.counts, n_cells, key)
                q = ws.sorted_indices[slot]
                q <= p && continue
                ws.axis_ok[q] || continue

                ws.visits[p] += 1
                if ws.visits[p] > candidate_cap
                    ws.capped[p] = true
                    capped_p = true
                    break
                end
                # Symmetric: the visit also counts against q since this pair
                # would have been q's responsibility had we walked from q.
                # Tracking it on p only is sufficient as a work bound.

                xq = pfield.particles[1, q]
                yq = pfield.particles[2, q]
                zq = pfield.particles[3, q]
                sq = R(pfield.particles[SIGMA_INDEX, q])

                rx = xq - xp; ry = yq - yp; rz = zq - zp
                dist = sqrt(rx*rx + ry*ry + rz*rz)
                dist > 0 || continue
                sigma_ref = sp >= sq ? sp : sq
                eta_pq = dist / sigma_ref
                eta_pq > max_eta_R && continue

                ex_q = ws.axis_x[q]; ey_q = ws.axis_y[q]; ez_q = ws.axis_z[q]
                cos_axes = ex_p * ex_q + ey_p * ey_q + ez_p * ez_q
                abs(cos_axes) < cos_tol && continue

                # Displacement alignment with p's axis; sign determines
                # downstream (positive) vs upstream (negative).
                inv_dist = inv(dist)
                proj_p = (rx * ex_p + ry * ey_p + rz * ez_p) * inv_dist
                abs(proj_p) < cos_tol && continue

                score = abs(cos_axes) / eta_pq

                if proj_p > 0
                    # q is downstream of p
                    _best2_insert!(ws.down_best, ws.down_best_score, p, q, score)
                    _best2_insert!(ws.up_best,   ws.up_best_score,   q, p, score)
                else
                    # q is upstream of p
                    _best2_insert!(ws.up_best,   ws.up_best_score,   p, q, score)
                    _best2_insert!(ws.down_best, ws.down_best_score, q, p, score)
                end
            end
        end
    end

    # ---------------------------------------------------------------- commit
    n_added = 0
    @inbounds for p in candidate_indices
        for k in 1:2
            q = ws.down_best[k, p]
            q == 0 && continue
            # Mutual-NN check: p must appear in q.up_best
            (ws.up_best[1, q] == p || ws.up_best[2, q] == p) || continue
            if add_edge!(graph, p, q;
                         coherent = true,
                         score    = ws.down_best_score[k, p])
                n_added += 1
            end
        end
    end

    return n_added
end

"""
    refine_filament_edges!(pfield; L_max=1.5, only_coherent=true,
                           max_splits=typemax(Int)) -> Int

Edge-driven split: walk the active edges of `pfield.filament_edge_graph`,
and for each edge `p → q` with `‖x_q − x_p‖ > L_max * (σ_p + σ_q)/2`,
insert a new particle `m` at the midpoint and replace the single edge
with `p → m` and `m → q`. Returns the number of edges split.

Per-particle quantities for the inserted `m` (see
`hybrid_filement_edge.md` Phase 4b for the derivation):

- `x_m  = (x_p + x_q) / 2`
- `σ_m  = (σ_p + σ_q) / 2`
- `Γ_m  = (Γ_p + Γ_q) / 3`         (with `Γ_p ← (2/3) Γ_p`, same for q)
- `vol_m = (4/3) π σ_m^3`
- `circ_m = (circ_p + circ_q) / 2`

Total `Γ_p + Γ_q` is conserved by the rebalance. Linear impulse is
conserved exactly whenever the local strength is tangent-aligned
(the condition under which `infer_filament_edges!` accepted the edge as
coherent in the first place). Angular impulse is exact only in narrower
special cases such as uniform tangent-aligned strengths or centered /
symmetric geometry; offset edges with along-tangent strength variation
generally retain a residual defect under this midpoint rule.

The pass is non-destructive on existing topology: it only mutates an
edge that meets the trigger. Edges incident on freshly-inserted `m`
particles are by construction half the length of the parent edge and
are intentionally not re-considered in the same call.

Bounded local work per split (≤4 adjacency-slot scans); no allocations
after the `ParticleField` is sized.

Kwargs:
- `L_max`: edge-length cap as a multiple of mean σ. Default `1.5`.
- `only_coherent`: when `true` (default), skip edges whose
  `down_coherent` flag is unset.
- `max_splits`: hard cap on the number of splits performed in this call.
"""
function refine_filament_edges!(pfield::ParticleField{R};
                                L_max::Real = 1.5,
                                only_coherent::Bool = true,
                                max_splits::Int = typemax(Int)) where {R}
    graph = pfield.filament_edge_graph
    np_at_start = get_np(pfield)
    maxp = pfield.maxparticles
    L_max_R = R(L_max)
    n_split = 0
    four_thirds_pi = R(4) * R(pi) / R(3)

    @inbounds for p in 1:np_at_start
        # Walk p's down slots. Inactive slots have neighbor==0 and are skipped.
        for k in 1:2
            n_split >= max_splits && break
            get_np(pfield) >= maxp && break  # capacity guard

            q = graph.down_neighbor[k, p]
            q == 0 && continue
            # Skip edges pointing to a particle inserted earlier in this
            # same call — they are by construction at half the parent's
            # length and not reconsidered until the next refine pass.
            q > np_at_start && continue
            (only_coherent && !graph.down_coherent[k, p]) && continue

            # Read geometry.
            xp1 = pfield.particles[X_INDEX.start,     p]
            xp2 = pfield.particles[X_INDEX.start + 1, p]
            xp3 = pfield.particles[X_INDEX.start + 2, p]
            xq1 = pfield.particles[X_INDEX.start,     q]
            xq2 = pfield.particles[X_INDEX.start + 1, q]
            xq3 = pfield.particles[X_INDEX.start + 2, q]
            σp = pfield.particles[SIGMA_INDEX, p]
            σq = pfield.particles[SIGMA_INDEX, q]

            dx1 = xq1 - xp1; dx2 = xq2 - xp2; dx3 = xq3 - xp3
            L2 = dx1*dx1 + dx2*dx2 + dx3*dx3
            σbar = (σp + σq) / R(2)
            thresh = L_max_R * σbar
            L2 > thresh * thresh || continue

            # Capture the existing edge's per-edge state before mutation.
            edge_coherent = graph.down_coherent[k, p]
            edge_score    = graph.down_score[k, p]

            # Read Γ + auxiliary fields for the redistribution.
            gp1 = pfield.particles[GAMMA_INDEX.start,     p]
            gp2 = pfield.particles[GAMMA_INDEX.start + 1, p]
            gp3 = pfield.particles[GAMMA_INDEX.start + 2, p]
            gq1 = pfield.particles[GAMMA_INDEX.start,     q]
            gq2 = pfield.particles[GAMMA_INDEX.start + 1, q]
            gq3 = pfield.particles[GAMMA_INDEX.start + 2, q]
            volp = pfield.particles[VOL_INDEX, p]
            volq = pfield.particles[VOL_INDEX, q]
            circp = pfield.particles[CIRCULATION_INDEX, p]
            circq = pfield.particles[CIRCULATION_INDEX, q]

            # Child quantities for m.
            xm1 = (xp1 + xq1) / R(2)
            xm2 = (xp2 + xq2) / R(2)
            xm3 = (xp3 + xq3) / R(2)
            σm  = σbar
            gm1 = (gp1 + gq1) / R(3)
            gm2 = (gp2 + gq2) / R(3)
            gm3 = (gp3 + gq3) / R(3)
            volm = four_thirds_pi * σm * σm * σm
            circm = (circp + circq) / R(2)

            # Rebalance endpoints (2/3 of original Γ each).
            two_thirds = R(2) / R(3)
            pfield.particles[GAMMA_INDEX.start,     p] = two_thirds * gp1
            pfield.particles[GAMMA_INDEX.start + 1, p] = two_thirds * gp2
            pfield.particles[GAMMA_INDEX.start + 2, p] = two_thirds * gp3
            pfield.particles[GAMMA_INDEX.start,     q] = two_thirds * gq1
            pfield.particles[GAMMA_INDEX.start + 1, q] = two_thirds * gq2
            pfield.particles[GAMMA_INDEX.start + 2, q] = two_thirds * gq3

            # Append m. add_particle zeros m's edge-graph adjacency.
            add_particle(pfield, (xm1, xm2, xm3), (gm1, gm2, gm3), σm;
                         vol = volm, circulation = circm, static = false)
            m = get_np(pfield)

            # Rewire: drop the old edge, add the two replacements.
            # Both add_edge! calls succeed by construction (m has empty
            # adjacency; p and q each just freed one slot).
            remove_edge!(graph, p, q)
            add_edge!(graph, p, m; coherent = edge_coherent, score = edge_score)
            add_edge!(graph, m, q; coherent = edge_coherent, score = edge_score)

            n_split += 1
        end
        n_split >= max_splits && break
        get_np(pfield) >= maxp && break
    end

    return n_split
end

"""
    coarsen_filament_edges!(pfield; L_min=0.75, only_coherent=true,
                            max_coarsens=typemax(Int),
                            atol=sqrt(eps(R))) -> Int

Conservative edge-driven coarsen pass. This is the exact inverse of
[`refine_filament_edges!`](@ref)'s midpoint split, not a general merge:
only particles with topology `p → m → q` and the exact split fingerprints
are removed. Returns the number of midpoints coarsened.

Eligibility:
- `m` has exactly one upstream and one downstream edge.
- `p != q`, both endpoints are live, and no duplicate `p → q` exists.
- `‖x_q - x_p‖ <= L_min * (σ_p + σ_q) / 2`.
- when `only_coherent=true`, both replacement edges are coherent.
- both replacement edges have matching coherence and matching score.
- `x_m ≈ (x_p + x_q)/2`, `σ_m ≈ (σ_p + σ_q)/2`, and
  `Γ_m ≈ (Γ_p + Γ_q)/2`, where `Γ_p` and `Γ_q` are the current
  post-split endpoint strengths.

On acceptance, the pass removes `p → m` and `m → q`, adds `p → q` with the
inherited edge metadata, scales both endpoint strengths by `3/2`, and then
removes `m`. Endpoint `σ`, volume, and circulation are left unchanged.
"""
function coarsen_filament_edges!(pfield::ParticleField{R};
                                 L_min::Real = 0.75,
                                 only_coherent::Bool = true,
                                 max_coarsens::Int = typemax(Int),
                                 atol::Real = sqrt(eps(R))) where {R}
    graph = pfield.filament_edge_graph
    L_min_R = R(L_min)
    atol_R = R(atol)
    n_coarsen = 0
    m = get_np(pfield)

    @inbounds while m >= 1
        if n_coarsen >= max_coarsens
            break
        end

        if up_count(graph, m) != 1 || down_count(graph, m) != 1
            m -= 1
            continue
        end

        p = graph.up_neighbor[1, m]
        q = graph.down_neighbor[1, m]
        np = get_np(pfield)
        if p == 0 || q == 0 || p == q || p > np || q > np
            m -= 1
            continue
        end

        kpm = find_down_slot(graph, p, m)
        kmq = find_down_slot(graph, m, q)
        if kpm == 0 || kmq == 0 || find_down_slot(graph, p, q) != 0
            m -= 1
            continue
        end

        coh_pm = graph.down_coherent[kpm, p]
        coh_mq = graph.down_coherent[kmq, m]
        if coh_pm != coh_mq || (only_coherent && (!coh_pm || !coh_mq))
            m -= 1
            continue
        end

        score_pm = graph.down_score[kpm, p]
        score_mq = graph.down_score[kmq, m]
        if abs(score_pm - score_mq) > atol_R
            m -= 1
            continue
        end

        xp1 = pfield.particles[X_INDEX.start,     p]
        xp2 = pfield.particles[X_INDEX.start + 1, p]
        xp3 = pfield.particles[X_INDEX.start + 2, p]
        xm1 = pfield.particles[X_INDEX.start,     m]
        xm2 = pfield.particles[X_INDEX.start + 1, m]
        xm3 = pfield.particles[X_INDEX.start + 2, m]
        xq1 = pfield.particles[X_INDEX.start,     q]
        xq2 = pfield.particles[X_INDEX.start + 1, q]
        xq3 = pfield.particles[X_INDEX.start + 2, q]

        σp = pfield.particles[SIGMA_INDEX, p]
        σm = pfield.particles[SIGMA_INDEX, m]
        σq = pfield.particles[SIGMA_INDEX, q]
        σbar = (σp + σq) / R(2)
        dx1 = xq1 - xp1; dx2 = xq2 - xp2; dx3 = xq3 - xp3
        L2 = dx1*dx1 + dx2*dx2 + dx3*dx3
        thresh = L_min_R * σbar
        if L2 > thresh * thresh
            m -= 1
            continue
        end

        if abs(xm1 - (xp1 + xq1) / R(2)) > atol_R ||
           abs(xm2 - (xp2 + xq2) / R(2)) > atol_R ||
           abs(xm3 - (xp3 + xq3) / R(2)) > atol_R ||
           abs(σm - σbar) > atol_R
            m -= 1
            continue
        end

        gp1 = pfield.particles[GAMMA_INDEX.start,     p]
        gp2 = pfield.particles[GAMMA_INDEX.start + 1, p]
        gp3 = pfield.particles[GAMMA_INDEX.start + 2, p]
        gm1 = pfield.particles[GAMMA_INDEX.start,     m]
        gm2 = pfield.particles[GAMMA_INDEX.start + 1, m]
        gm3 = pfield.particles[GAMMA_INDEX.start + 2, m]
        gq1 = pfield.particles[GAMMA_INDEX.start,     q]
        gq2 = pfield.particles[GAMMA_INDEX.start + 1, q]
        gq3 = pfield.particles[GAMMA_INDEX.start + 2, q]

        if abs(gm1 - (gp1 + gq1) / R(2)) > atol_R ||
           abs(gm2 - (gp2 + gq2) / R(2)) > atol_R ||
           abs(gm3 - (gp3 + gq3) / R(2)) > atol_R
            m -= 1
            continue
        end

        remove_edge!(graph, p, m)
        remove_edge!(graph, m, q)
        add_edge!(graph, p, q; coherent = coh_pm, score = score_pm)

        three_halves = R(3) / R(2)
        pfield.particles[GAMMA_INDEX.start,     p] = three_halves * gp1
        pfield.particles[GAMMA_INDEX.start + 1, p] = three_halves * gp2
        pfield.particles[GAMMA_INDEX.start + 2, p] = three_halves * gp3
        pfield.particles[GAMMA_INDEX.start,     q] = three_halves * gq1
        pfield.particles[GAMMA_INDEX.start + 1, q] = three_halves * gq2
        pfield.particles[GAMMA_INDEX.start + 2, q] = three_halves * gq3

        remove_particle(pfield, m)
        n_coarsen += 1
        m = min(m, get_np(pfield))
        # Re-check this slot because remove_particle may have moved the
        # previous last particle into index m.
    end

    return n_coarsen
end

"""
    bundle_coarsen_filament_edges!(pfield; bundle_overlap=0.5,
                                   bundle_angle_tol=π/4,
                                   bundle_sigma_tol=0.25,
                                   only_coherent=true,
                                   max_coarsens=typemax(Int),
                                   atol=sqrt(eps(R))) -> Int

Intra-filament bundle merge over the bounded edge graph. For each local
trio pattern — `(u, v) → c` (2→1 convergence) or `c → (u, v)` (1→2
divergence) — collapse the pair `(u, v)` into a single particle when
the conservative redundancy gates pass. The kept trio edge stays
incident to `c`; `v`'s other-side edges are transferred to the survivor
`u`. Returns the number of trios collapsed (each 2→1 or 1→2 counts as
1).

Gates (all must pass — opt-in conservative):

- The non-trio side of `u` and `v` is sized so the merged degree never
  exceeds 2.
- Both trio edges share the same `down_coherent` value, and when
  `only_coherent` both are coherent.
- `‖x_u − x_v‖ ≤ bundle_overlap · min(σ_u, σ_v)` — overlap-based
  redundancy.
- `cos∠(Γ_u, Γ_v) ≥ cos(bundle_angle_tol)` — tangent alignment between
  the merging pair.
- `|σ_u − σ_v| / max(σ_u, σ_v) ≤ bundle_sigma_tol`.
- Each transferred edge is dropped silently when it would duplicate an
  existing edge of `u`; the count gate guarantees the survivor still
  satisfies the 2-in/2-out cap after the transfer.

Conservation on acceptance:
- `Γ_uv = Γ_u + Γ_v` (total Γ exact).
- `x_uv = (α x_u + β x_v) / (α + β)` with `α = Γ_u · t̂`,
  `β = Γ_v · t̂`, `t̂ = (Γ_u + Γ_v) / |Γ_u + Γ_v|`. Linear impulse is
  exact when `Γ_u, Γ_v` are tangent-aligned (the gated case). Falls
  back to a `(1/2, 1/2)` weighted centroid when the Γ sum vanishes.
- `σ_uv = max(σ_u, σ_v)`, `vol_uv = (4/3) π σ_uv^3`,
  `circ_uv = circ_u + circ_v`.
- Angular impulse picks up an `O(|Δx|² · ΔΓ)` defect under the same
  condition.

The survivor is written back into the lower-indexed slot; the
higher-indexed particle is removed via `remove_particle`, which
preserves bounded local work via swap-with-last with full back-pointer
rewiring.
"""
function bundle_coarsen_filament_edges!(pfield::ParticleField{R};
                                        bundle_overlap::Real = 0.5,
                                        bundle_angle_tol::Real = π/4,
                                        bundle_sigma_tol::Real = 0.25,
                                        only_coherent::Bool = true,
                                        max_coarsens::Int = typemax(Int),
                                        atol::Real = sqrt(eps(R))) where {R}
    graph = pfield.filament_edge_graph
    overlap_R = R(bundle_overlap)
    sigma_tol_R = R(bundle_sigma_tol)
    cos_tol = R(cos(bundle_angle_tol))
    atol_R = R(atol)
    four_thirds_pi = R(4) * R(pi) / R(3)
    n_coarsen = 0
    i = get_np(pfield)

    @inbounds while i >= 1
        if n_coarsen >= max_coarsens
            break
        end
        merged = false

        # 2→1 convergence: (a, b) → c where c = i.
        if !merged && up_count(graph, i) == 2
            a = graph.up_neighbor[1, i]
            b = graph.up_neighbor[2, i]
            if _try_bundle_converge!(pfield, graph, a, b, i,
                                     overlap_R, cos_tol, sigma_tol_R,
                                     only_coherent, atol_R, four_thirds_pi)
                merged = true
                n_coarsen += 1
            end
        end

        # 1→2 divergence: a → (b, c) where a = i.
        if !merged && down_count(graph, i) == 2
            b = graph.down_neighbor[1, i]
            c = graph.down_neighbor[2, i]
            if _try_bundle_diverge!(pfield, graph, i, b, c,
                                    overlap_R, cos_tol, sigma_tol_R,
                                    only_coherent, atol_R, four_thirds_pi)
                merged = true
                n_coarsen += 1
            end
        end

        if merged
            # Re-examine the same slot: remove_particle may have moved the
            # previous last particle into i if i wasn't the removed slot.
            i = min(i, get_np(pfield))
        else
            i -= 1
        end
    end

    return n_coarsen
end

# Internal: try to merge (a, b) at convergence point c. Returns true on
# success. All gates are evaluated here so the outer walk stays compact.
@inline function _try_bundle_converge!(pfield::ParticleField{R}, graph,
                                       a::Int, b::Int, c::Int,
                                       overlap_R::R, cos_tol::R,
                                       sigma_tol_R::R, only_coherent::Bool,
                                       atol_R::R, four_thirds_pi::R) where {R}
    a == b && return false
    np = get_np(pfield)
    (a < 1 || a > np || b < 1 || b > np || c < 1 || c > np) && return false
    a == c && return false
    b == c && return false

    # Trio shape: each of a, b has exactly one downstream edge (to c).
    down_count(graph, a) == 1 || return false
    down_count(graph, b) == 1 || return false

    k_ac = find_down_slot(graph, a, c)
    k_bc = find_down_slot(graph, b, c)
    (k_ac == 0 || k_bc == 0) && return false

    coh_ac = graph.down_coherent[k_ac, a]
    coh_bc = graph.down_coherent[k_bc, b]
    (coh_ac != coh_bc) && return false
    only_coherent && (!coh_ac || !coh_bc) && return false

    # Geometry of the merging pair.
    xa1 = pfield.particles[X_INDEX.start,     a]
    xa2 = pfield.particles[X_INDEX.start + 1, a]
    xa3 = pfield.particles[X_INDEX.start + 2, a]
    xb1 = pfield.particles[X_INDEX.start,     b]
    xb2 = pfield.particles[X_INDEX.start + 1, b]
    xb3 = pfield.particles[X_INDEX.start + 2, b]
    σa  = pfield.particles[SIGMA_INDEX, a]
    σb  = pfield.particles[SIGMA_INDEX, b]

    dx1 = xa1 - xb1; dx2 = xa2 - xb2; dx3 = xa3 - xb3
    dist2 = dx1*dx1 + dx2*dx2 + dx3*dx3
    σmin = min(σa, σb)
    σmax = max(σa, σb)
    thresh = overlap_R * σmin
    dist2 > thresh * thresh && return false
    σmax > 0 || return false
    abs(σa - σb) > sigma_tol_R * σmax && return false

    ga1 = pfield.particles[GAMMA_INDEX.start,     a]
    ga2 = pfield.particles[GAMMA_INDEX.start + 1, a]
    ga3 = pfield.particles[GAMMA_INDEX.start + 2, a]
    gb1 = pfield.particles[GAMMA_INDEX.start,     b]
    gb2 = pfield.particles[GAMMA_INDEX.start + 1, b]
    gb3 = pfield.particles[GAMMA_INDEX.start + 2, b]
    mag_a2 = ga1*ga1 + ga2*ga2 + ga3*ga3
    mag_b2 = gb1*gb1 + gb2*gb2 + gb3*gb3
    (mag_a2 <= 0 || mag_b2 <= 0) && return false
    dot_ab = ga1*gb1 + ga2*gb2 + ga3*gb3
    mag_ab = sqrt(mag_a2 * mag_b2)
    dot_ab < cos_tol * mag_ab && return false

    # Survivor = lower index, removed = higher index.
    u, v = a < b ? (a, b) : (b, a)
    k_uc = (u == a) ? k_ac : k_bc
    score_uc = graph.down_score[k_uc, u]
    coh_uc   = graph.down_coherent[k_uc, u]

    # Degree gate on the non-trio side. Count how many of v's upstream
    # neighbors are not already upstream of u; this is the new edges u
    # would gain after the transfer. Self-references to u itself are
    # excluded.
    new_ups = 0
    for k in 1:2
        w = graph.up_neighbor[k, v]
        w == 0 && continue
        w == u && continue
        find_down_slot(graph, w, u) == 0 && (new_ups += 1)
    end
    up_count(graph, u) + new_ups > 2 && return false

    # Merged quantities.
    gm1 = ga1 + gb1
    gm2 = ga2 + gb2
    gm3 = ga3 + gb3
    mag_m2 = gm1*gm1 + gm2*gm2 + gm3*gm3
    if mag_m2 > 0
        mag_m = sqrt(mag_m2)
        if u == a
            α = (ga1*gm1 + ga2*gm2 + ga3*gm3) / mag_m
            β = (gb1*gm1 + gb2*gm2 + gb3*gm3) / mag_m
        else
            α = (gb1*gm1 + gb2*gm2 + gb3*gm3) / mag_m
            β = (ga1*gm1 + ga2*gm2 + ga3*gm3) / mag_m
        end
        denom = α + β
    else
        # Degenerate: anti-parallel sum cancels. Fall back to centroid.
        α = R(1); β = R(1); denom = R(2)
    end
    abs(denom) < atol_R && return false  # numerically degenerate

    xu1 = pfield.particles[X_INDEX.start,     u]
    xu2 = pfield.particles[X_INDEX.start + 1, u]
    xu3 = pfield.particles[X_INDEX.start + 2, u]
    xv1 = pfield.particles[X_INDEX.start,     v]
    xv2 = pfield.particles[X_INDEX.start + 1, v]
    xv3 = pfield.particles[X_INDEX.start + 2, v]

    xm1 = (α * xu1 + β * xv1) / denom
    xm2 = (α * xu2 + β * xv2) / denom
    xm3 = (α * xu3 + β * xv3) / denom
    σm  = σmax
    volm = four_thirds_pi * σm * σm * σm
    circu = pfield.particles[CIRCULATION_INDEX, u]
    circv = pfield.particles[CIRCULATION_INDEX, v]

    # Write survivor in place.
    pfield.particles[X_INDEX.start,         u] = xm1
    pfield.particles[X_INDEX.start + 1,     u] = xm2
    pfield.particles[X_INDEX.start + 2,     u] = xm3
    pfield.particles[GAMMA_INDEX.start,     u] = gm1
    pfield.particles[GAMMA_INDEX.start + 1, u] = gm2
    pfield.particles[GAMMA_INDEX.start + 2, u] = gm3
    pfield.particles[SIGMA_INDEX,           u] = σm
    pfield.particles[VOL_INDEX,             u] = volm
    pfield.particles[CIRCULATION_INDEX,     u] = circu + circv

    # Edge surgery: drop v→c (the trio edge on the removed side); transfer
    # v's upstream edges to u. The kept u→c edge already carries the
    # right metadata (coh_uc, score_uc) and is left alone.
    remove_edge!(graph, v, c)
    # Note: coh_uc / score_uc would only need to be re-applied if the
    # kept slot moved. find_down_slot(graph, u, c) is still k_uc unless v
    # happened to overlap u's columns — but v ≠ u by construction and we
    # only mutated v's column.

    # Transfer v's upstream edges. After each remove_edge!, compaction
    # moves slot 2 into slot 1, so we always read from slot 1 to drain.
    for _ in 1:2
        w = graph.up_neighbor[1, v]
        w == 0 && break
        kwv = find_down_slot(graph, w, v)
        # kwv should be nonzero by mirror invariant; guard anyway.
        coh_wv = (kwv == 0) ? false : graph.down_coherent[kwv, w]
        score_wv = (kwv == 0) ? zero(R) : graph.down_score[kwv, w]
        remove_edge!(graph, w, v)
        if w != u && find_down_slot(graph, w, u) == 0
            add_edge!(graph, w, u; coherent = coh_wv, score = score_wv)
        end
    end

    # v has no remaining edges at this point. remove_particle does its
    # own cleanup and swap-with-last back-pointer rewiring.
    remove_particle(pfield, v)
    return true
end

# Internal: try to merge (b, c) at divergence point a = i. Returns true on
# success. Mirror of `_try_bundle_converge!` on the downstream side.
@inline function _try_bundle_diverge!(pfield::ParticleField{R}, graph,
                                      a::Int, b::Int, c::Int,
                                      overlap_R::R, cos_tol::R,
                                      sigma_tol_R::R, only_coherent::Bool,
                                      atol_R::R, four_thirds_pi::R) where {R}
    b == c && return false
    np = get_np(pfield)
    (a < 1 || a > np || b < 1 || b > np || c < 1 || c > np) && return false
    a == b && return false
    a == c && return false

    # Trio shape: each of b, c has exactly one upstream edge (from a).
    up_count(graph, b) == 1 || return false
    up_count(graph, c) == 1 || return false

    k_ab = find_down_slot(graph, a, b)
    k_ac = find_down_slot(graph, a, c)
    (k_ab == 0 || k_ac == 0) && return false

    coh_ab = graph.down_coherent[k_ab, a]
    coh_ac = graph.down_coherent[k_ac, a]
    (coh_ab != coh_ac) && return false
    only_coherent && (!coh_ab || !coh_ac) && return false

    xb1 = pfield.particles[X_INDEX.start,     b]
    xb2 = pfield.particles[X_INDEX.start + 1, b]
    xb3 = pfield.particles[X_INDEX.start + 2, b]
    xc1 = pfield.particles[X_INDEX.start,     c]
    xc2 = pfield.particles[X_INDEX.start + 1, c]
    xc3 = pfield.particles[X_INDEX.start + 2, c]
    σb  = pfield.particles[SIGMA_INDEX, b]
    σc  = pfield.particles[SIGMA_INDEX, c]

    dx1 = xb1 - xc1; dx2 = xb2 - xc2; dx3 = xb3 - xc3
    dist2 = dx1*dx1 + dx2*dx2 + dx3*dx3
    σmin = min(σb, σc)
    σmax = max(σb, σc)
    thresh = overlap_R * σmin
    dist2 > thresh * thresh && return false
    σmax > 0 || return false
    abs(σb - σc) > sigma_tol_R * σmax && return false

    gb1 = pfield.particles[GAMMA_INDEX.start,     b]
    gb2 = pfield.particles[GAMMA_INDEX.start + 1, b]
    gb3 = pfield.particles[GAMMA_INDEX.start + 2, b]
    gc1 = pfield.particles[GAMMA_INDEX.start,     c]
    gc2 = pfield.particles[GAMMA_INDEX.start + 1, c]
    gc3 = pfield.particles[GAMMA_INDEX.start + 2, c]
    mag_b2 = gb1*gb1 + gb2*gb2 + gb3*gb3
    mag_c2 = gc1*gc1 + gc2*gc2 + gc3*gc3
    (mag_b2 <= 0 || mag_c2 <= 0) && return false
    dot_bc = gb1*gc1 + gb2*gc2 + gb3*gc3
    mag_bc = sqrt(mag_b2 * mag_c2)
    dot_bc < cos_tol * mag_bc && return false

    u, v = b < c ? (b, c) : (c, b)

    # Degree gate on the non-trio (downstream) side.
    new_downs = 0
    for k in 1:2
        w = graph.down_neighbor[k, v]
        w == 0 && continue
        w == u && continue
        find_down_slot(graph, u, w) == 0 && (new_downs += 1)
    end
    down_count(graph, u) + new_downs > 2 && return false

    gm1 = gb1 + gc1
    gm2 = gb2 + gc2
    gm3 = gb3 + gc3
    mag_m2 = gm1*gm1 + gm2*gm2 + gm3*gm3
    if mag_m2 > 0
        mag_m = sqrt(mag_m2)
        if u == b
            α = (gb1*gm1 + gb2*gm2 + gb3*gm3) / mag_m
            β = (gc1*gm1 + gc2*gm2 + gc3*gm3) / mag_m
        else
            α = (gc1*gm1 + gc2*gm2 + gc3*gm3) / mag_m
            β = (gb1*gm1 + gb2*gm2 + gb3*gm3) / mag_m
        end
        denom = α + β
    else
        α = R(1); β = R(1); denom = R(2)
    end
    abs(denom) < atol_R && return false

    xu1 = pfield.particles[X_INDEX.start,     u]
    xu2 = pfield.particles[X_INDEX.start + 1, u]
    xu3 = pfield.particles[X_INDEX.start + 2, u]
    xv1 = pfield.particles[X_INDEX.start,     v]
    xv2 = pfield.particles[X_INDEX.start + 1, v]
    xv3 = pfield.particles[X_INDEX.start + 2, v]

    xm1 = (α * xu1 + β * xv1) / denom
    xm2 = (α * xu2 + β * xv2) / denom
    xm3 = (α * xu3 + β * xv3) / denom
    σm  = σmax
    volm = four_thirds_pi * σm * σm * σm
    circu = pfield.particles[CIRCULATION_INDEX, u]
    circv = pfield.particles[CIRCULATION_INDEX, v]

    pfield.particles[X_INDEX.start,         u] = xm1
    pfield.particles[X_INDEX.start + 1,     u] = xm2
    pfield.particles[X_INDEX.start + 2,     u] = xm3
    pfield.particles[GAMMA_INDEX.start,     u] = gm1
    pfield.particles[GAMMA_INDEX.start + 1, u] = gm2
    pfield.particles[GAMMA_INDEX.start + 2, u] = gm3
    pfield.particles[SIGMA_INDEX,           u] = σm
    pfield.particles[VOL_INDEX,             u] = volm
    pfield.particles[CIRCULATION_INDEX,     u] = circu + circv

    # Drop a→v trio edge; transfer v's downstream edges to u.
    remove_edge!(graph, a, v)

    for _ in 1:2
        w = graph.down_neighbor[1, v]
        w == 0 && break
        coh_vw   = graph.down_coherent[1, v]
        score_vw = graph.down_score[1, v]
        remove_edge!(graph, v, w)
        if w != u && find_down_slot(graph, u, w) == 0
            add_edge!(graph, u, w; coherent = coh_vw, score = score_vw)
        end
    end

    remove_particle(pfield, v)
    return true
end

"""
    merge_filament_bundles!(pfield; kwargs...) -> Int

Cross-filament bundle merge (Phase 4f). Spatial-search-driven collapse of
particles from *different* filaments that are spatially overlapping and
Γ-aligned. Thin wrapper around [`merge_particles!`](@ref) that supplies a
Γ-alignment gate (`cos(cross_angle_tol)`) and a per-cluster callback
which clears `down_coherent` on every edge incident to the survivor so
the next [`infer_filament_edges!`](@ref) refreshes them.

Anti-parallel pairs are rejected by the alignment gate. Static particles
are skipped (inherited from `merge_particles!`).

Returns the number of removed particles.
"""
function merge_filament_bundles!(pfield::ParticleField{R};
                                  r_merge::Real=0.5,
                                  r_hash::Real=-1.0,
                                  sigma_relative::Bool=true,
                                  max_sigma_ratio::Real=2.0,
                                  cross_angle_tol::Real=π/4,
                                  skip_static::Bool=true) where {R}
    graph = pfield.filament_edge_graph
    cb = function(rep::Int)
        for k in 1:down_count(graph, rep)
            graph.down_coherent[k, rep] = false
        end
        for k in 1:up_count(graph, rep)
            src = graph.up_neighbor[k, rep]
            src == 0 && continue
            slot = find_down_slot(graph, src, rep)
            slot != 0 && (graph.down_coherent[slot, src] = false)
        end
        return nothing
    end
    return merge_particles!(pfield;
                            r_merge=r_merge, r_hash=r_hash,
                            sigma_relative=sigma_relative,
                            max_sigma_ratio=max_sigma_ratio,
                            skip_static=skip_static,
                            gamma_align_cos=cos(cross_angle_tol),
                            on_representative=cb)
end

"""
    refine_filaments!(pfield; kwargs...) -> Int

Intra-filament refinement cadence: validate/repair, infer edges, split
long coherent edges, exact-inverse coarsen short split midpoints, and
bundle-merge redundant (a,b)→c / a→(b,c) trios. Returns the total
structural mutation count
`inferred + split + exact_coarsened + bundle_coarsened`.

Cross-filament merging (proximity-based collapse of unrelated particles)
runs only when `do_cross_merge=true`. It calls
[`merge_filament_bundles!`](@ref) as the last step of the cadence and the
result is included in the returned total.

Diagnostics are intentionally omitted from this runtime API to keep the
return type stable. Use [`refine_filaments_observables!`](@ref) when
reports and reserved diagnostic counts are needed.
"""
function refine_filaments!(pfield::ParticleField{R};
                           max_eta::Real = 2.0,
                           angle_tol::Real = π/4,
                           axis::Symbol = :strength,
                           candidate_cap::Int = 16,
                           L_max::Real = 1.5,
                           L_min::Real = 0.75,
                           only_coherent::Bool = true,
                           max_splits::Int = typemax(Int),
                           max_coarsens::Int = typemax(Int),
                           coarsen_atol::Real = sqrt(eps(R)),
                           bundle_overlap::Real = 0.5,
                           bundle_angle_tol::Real = angle_tol,
                           bundle_sigma_tol::Real = 0.25,
                           do_repair::Bool = true,
                           do_infer::Bool = true,
                           do_split::Bool = true,
                           do_coarsen::Bool = true,
                           do_bundle_coarsen::Bool = true,
                           do_cross_merge::Bool = false,
                           cross_r_merge::Real = 0.5,
                           cross_angle_tol::Real = angle_tol,
                           cross_max_sigma_ratio::Real = 2.0) where {R}
    return _refine_impl!(Val(false), pfield;
                         max_eta = max_eta,
                         angle_tol = angle_tol,
                         axis = axis,
                         candidate_cap = candidate_cap,
                         L_max = L_max,
                         L_min = L_min,
                         only_coherent = only_coherent,
                         max_splits = max_splits,
                         max_coarsens = max_coarsens,
                         coarsen_atol = coarsen_atol,
                         bundle_overlap = bundle_overlap,
                         bundle_angle_tol = bundle_angle_tol,
                         bundle_sigma_tol = bundle_sigma_tol,
                         do_repair = do_repair,
                         do_infer = do_infer,
                         do_split = do_split,
                         do_coarsen = do_coarsen,
                         do_bundle_coarsen = do_bundle_coarsen,
                         do_cross_merge = do_cross_merge,
                         cross_r_merge = cross_r_merge,
                         cross_angle_tol = cross_angle_tol,
                         cross_max_sigma_ratio = cross_max_sigma_ratio)
end

"""
    refine_filaments_observables!(pfield; kwargs...) -> NamedTuple

Run the same cadence as [`refine_filaments!`](@ref), returning validation
reports and stable nested mutation counts:

```
(
    reports = (initial = ..., repair = ..., final = ...),
    counts = (
        inferred = ..., split = ..., exact_coarsened = ...,
        bundle_coarsened = ..., merged = 0, total = ...,
    ),
)
```

`counts.merged` is the number of particles removed by the cross-filament
merge pass (Phase 4f). It is zero unless `do_cross_merge=true`.

`reports.repair === reports.initial` (object identity) iff no repair was
attempted — either because `do_repair=false` or because the initial
report was already `ok`. After a repair attempt,
`reports.repair.ok == false` indicates non-self-healing structural
breakage.
"""
function refine_filaments_observables!(pfield::ParticleField{R};
                                       max_eta::Real = 2.0,
                                       angle_tol::Real = π/4,
                                       axis::Symbol = :strength,
                                       candidate_cap::Int = 16,
                                       L_max::Real = 1.5,
                                       L_min::Real = 0.75,
                                       only_coherent::Bool = true,
                                       max_splits::Int = typemax(Int),
                                       max_coarsens::Int = typemax(Int),
                                       coarsen_atol::Real = sqrt(eps(R)),
                                       bundle_overlap::Real = 0.5,
                                       bundle_angle_tol::Real = angle_tol,
                                       bundle_sigma_tol::Real = 0.25,
                                       do_repair::Bool = true,
                                       do_infer::Bool = true,
                                       do_split::Bool = true,
                                       do_coarsen::Bool = true,
                                       do_bundle_coarsen::Bool = true,
                                       do_cross_merge::Bool = false,
                                       cross_r_merge::Real = 0.5,
                                       cross_angle_tol::Real = angle_tol,
                                       cross_max_sigma_ratio::Real = 2.0) where {R}
    return _refine_impl!(Val(true), pfield;
                         max_eta = max_eta,
                         angle_tol = angle_tol,
                         axis = axis,
                         candidate_cap = candidate_cap,
                         L_max = L_max,
                         L_min = L_min,
                         only_coherent = only_coherent,
                         max_splits = max_splits,
                         max_coarsens = max_coarsens,
                         coarsen_atol = coarsen_atol,
                         bundle_overlap = bundle_overlap,
                         bundle_angle_tol = bundle_angle_tol,
                         bundle_sigma_tol = bundle_sigma_tol,
                         do_repair = do_repair,
                         do_infer = do_infer,
                         do_split = do_split,
                         do_coarsen = do_coarsen,
                         do_bundle_coarsen = do_bundle_coarsen,
                         do_cross_merge = do_cross_merge,
                         cross_r_merge = cross_r_merge,
                         cross_angle_tol = cross_angle_tol,
                         cross_max_sigma_ratio = cross_max_sigma_ratio)
end

function _refine_core!(pfield::ParticleField{R};
                       max_eta::Real,
                       angle_tol::Real,
                       axis::Symbol,
                       candidate_cap::Int,
                       L_max::Real,
                       L_min::Real,
                       only_coherent::Bool,
                       max_splits::Int,
                       max_coarsens::Int,
                       coarsen_atol::Real,
                       bundle_overlap::Real,
                       bundle_angle_tol::Real,
                       bundle_sigma_tol::Real,
                       do_repair::Bool,
                       do_infer::Bool,
                       do_split::Bool,
                       do_coarsen::Bool,
                       do_bundle_coarsen::Bool,
                       do_cross_merge::Bool,
                       cross_r_merge::Real,
                       cross_angle_tol::Real,
                       cross_max_sigma_ratio::Real) where {R}
    initial_report = validate_filament_edges(pfield)
    repair_report = (do_repair && !initial_report.ok) ?
        repair_filament_edges!(pfield) : initial_report

    inferred = do_infer ? infer_filament_edges!(pfield;
                                                max_eta = max_eta,
                                                angle_tol = angle_tol,
                                                axis = axis,
                                                candidate_cap = candidate_cap) : 0
    split = do_split ? refine_filament_edges!(pfield;
                                              L_max = L_max,
                                              only_coherent = only_coherent,
                                              max_splits = max_splits) : 0
    exact_coarsened = do_coarsen ? coarsen_filament_edges!(pfield;
                                                           L_min = L_min,
                                                           only_coherent = only_coherent,
                                                           max_coarsens = max_coarsens,
                                                           atol = coarsen_atol) : 0
    bundle_coarsened = do_bundle_coarsen ? bundle_coarsen_filament_edges!(pfield;
                                                                         bundle_overlap = bundle_overlap,
                                                                         bundle_angle_tol = bundle_angle_tol,
                                                                         bundle_sigma_tol = bundle_sigma_tol,
                                                                         only_coherent = only_coherent,
                                                                         max_coarsens = max_coarsens,
                                                                         atol = coarsen_atol) : 0
    merged = do_cross_merge ? merge_filament_bundles!(pfield;
                                                      r_merge = cross_r_merge,
                                                      max_sigma_ratio = cross_max_sigma_ratio,
                                                      cross_angle_tol = cross_angle_tol) : 0
    return (initial_report, repair_report, inferred, split, exact_coarsened, bundle_coarsened, merged)
end

function _refine_impl!(::Val{false}, pfield::ParticleField{R};
                       kwargs...)::Int where {R}
    _, _, inferred, split, exact_coarsened, bundle_coarsened, merged = _refine_core!(pfield; kwargs...)
    return inferred + split + exact_coarsened + bundle_coarsened + merged
end

function _refine_impl!(::Val{true}, pfield::ParticleField{R};
                       kwargs...)::NamedTuple where {R}
    initial_report, repair_report, inferred, split, exact_coarsened, bundle_coarsened, merged =
        _refine_core!(pfield; kwargs...)
    final_report = validate_filament_edges(pfield)
    total = inferred + split + exact_coarsened + bundle_coarsened + merged

    return (
        reports = (
            initial = initial_report,
            repair = repair_report,
            final = final_report,
        ),
        counts = (
            inferred = inferred,
            split = split,
            exact_coarsened = exact_coarsened,
            bundle_coarsened = bundle_coarsened,
            merged = merged,
            total = total,
        ),
    )
end

# ------------------------------------------------------------------------------
# Filament-edge calibration harness (Phase 5 — observation only)
# ------------------------------------------------------------------------------

"""
    FilamentCalibrationReport

Read-only diagnostics returned by [`calibrate_filament_edges`](@ref).
The report records graph state, bounded inference-work observations, and
would-act counts for split/coarsen/bundle/cross-merge criteria. It does
not imply that any runtime threshold has been selected.
"""
struct FilamentCalibrationReport{R}
    np::Int
    active_edges::Int
    coherent_edges::Int
    incoherent_edges::Int
    degree_histogram::NTuple{9, Int}
    capped_particles::Int
    candidate_visits::Int
    candidate_pairs::Int
    candidate_accepted::Int
    candidate_mutual::Int
    split_eligible::Int
    exact_coarsen_eligible::Int
    bundle_coarsen_eligible::Int
    cross_merge_observations::Int
    edge_stats::NamedTuple
    settings::NamedTuple
    validation::FilamentEdgeReport
end

@inline function _filament_edge_length_sigma(pfield::ParticleField{R}, p::Int, q::Int) where {R}
    dx1 = pfield.particles[X_INDEX.start,     q] - pfield.particles[X_INDEX.start,     p]
    dx2 = pfield.particles[X_INDEX.start + 1, q] - pfield.particles[X_INDEX.start + 1, p]
    dx3 = pfield.particles[X_INDEX.start + 2, q] - pfield.particles[X_INDEX.start + 2, p]
    L = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3)
    σp = pfield.particles[SIGMA_INDEX, p]
    σq = pfield.particles[SIGMA_INDEX, q]
    return L, σp, σq
end

@inline function _gamma_norm_dot(pfield::ParticleField{R}, a::Int, b::Int) where {R}
    ga1 = pfield.particles[GAMMA_INDEX.start,     a]
    ga2 = pfield.particles[GAMMA_INDEX.start + 1, a]
    ga3 = pfield.particles[GAMMA_INDEX.start + 2, a]
    gb1 = pfield.particles[GAMMA_INDEX.start,     b]
    gb2 = pfield.particles[GAMMA_INDEX.start + 1, b]
    gb3 = pfield.particles[GAMMA_INDEX.start + 2, b]
    na2 = ga1*ga1 + ga2*ga2 + ga3*ga3
    nb2 = gb1*gb1 + gb2*gb2 + gb3*gb3
    return ga1*gb1 + ga2*gb2 + ga3*gb3, na2, nb2
end

function _empty_edge_stats(::Type{R}) where {R}
    z = zero(R)
    return (
        eta_edge = (min=z, mean=z, max=z),
        tangent_alignment = (min=z, mean=z, max=z),
        strength_compatibility = (min=z, mean=z, max=z),
        sigma_ratio = (min=z, mean=z, max=z),
        circulation_mismatch = (min=z, mean=z, max=z),
        score = (min=z, mean=z, max=z),
    )
end

@inline _finish_stat(minv, sumv, maxv, n, ::Type{R}) where {R} =
    n == 0 ? (min=zero(R), mean=zero(R), max=zero(R)) :
             (min=minv, mean=sumv / R(n), max=maxv)

function _edge_summary_stats(pfield::ParticleField{R}) where {R}
    graph = pfield.filament_edge_graph
    np = get_np(pfield)
    n = 0
    eta_min = typemax(R); eta_sum = zero(R); eta_max = zero(R)
    tan_min = typemax(R); tan_sum = zero(R); tan_max = zero(R)
    str_min = typemax(R); str_sum = zero(R); str_max = zero(R)
    sig_min = typemax(R); sig_sum = zero(R); sig_max = zero(R)
    circ_min = typemax(R); circ_sum = zero(R); circ_max = zero(R)
    score_min = typemax(R); score_sum = zero(R); score_max = zero(R)

    @inbounds for p in 1:np
        ex, ey, ez, ok_axis = _unit_strength(pfield, p)
        for k in 1:2
            q = graph.down_neighbor[k, p]
            q == 0 && continue
            (q < 1 || q > np) && continue
            L, σp, σq = _filament_edge_length_sigma(pfield, p, q)
            σref = max(σp, σq)
            σbar = (σp + σq) / R(2)
            eta = σref > 0 ? L / σref : zero(R)
            ratio = σref > 0 ? min(σp, σq) / σref : zero(R)
            tan = zero(R)
            if ok_axis && L > 0
                rx = pfield.particles[X_INDEX.start,     q] - pfield.particles[X_INDEX.start,     p]
                ry = pfield.particles[X_INDEX.start + 1, q] - pfield.particles[X_INDEX.start + 1, p]
                rz = pfield.particles[X_INDEX.start + 2, q] - pfield.particles[X_INDEX.start + 2, p]
                tan = abs((rx*ex + ry*ey + rz*ez) / L)
            end
            gd, gn1, gn2 = _gamma_norm_dot(pfield, p, q)
            compat = (gn1 > 0 && gn2 > 0) ? abs(gd) / sqrt(gn1 * gn2) : zero(R)
            cp = pfield.particles[CIRCULATION_INDEX, p]
            cq = pfield.particles[CIRCULATION_INDEX, q]
            cmis = σbar > 0 ? abs(cp - cq) / max(abs(cp) + abs(cq), eps(R)) : zero(R)
            score = graph.down_score[k, p]

            n += 1
            eta_min = min(eta_min, eta); eta_sum += eta; eta_max = max(eta_max, eta)
            tan_min = min(tan_min, tan); tan_sum += tan; tan_max = max(tan_max, tan)
            str_min = min(str_min, compat); str_sum += compat; str_max = max(str_max, compat)
            sig_min = min(sig_min, ratio); sig_sum += ratio; sig_max = max(sig_max, ratio)
            circ_min = min(circ_min, cmis); circ_sum += cmis; circ_max = max(circ_max, cmis)
            score_min = min(score_min, score); score_sum += score; score_max = max(score_max, score)
        end
    end
    n == 0 && return _empty_edge_stats(R)
    return (
        eta_edge = _finish_stat(eta_min, eta_sum, eta_max, n, R),
        tangent_alignment = _finish_stat(tan_min, tan_sum, tan_max, n, R),
        strength_compatibility = _finish_stat(str_min, str_sum, str_max, n, R),
        sigma_ratio = _finish_stat(sig_min, sig_sum, sig_max, n, R),
        circulation_mismatch = _finish_stat(circ_min, circ_sum, circ_max, n, R),
        score = _finish_stat(score_min, score_sum, score_max, n, R),
    )
end

@inline function _would_split_edge(pfield::ParticleField{R}, p::Int, q::Int, L_max_R::R) where {R}
    L, σp, σq = _filament_edge_length_sigma(pfield, p, q)
    thresh = L_max_R * (σp + σq) / R(2)
    return L > thresh
end

function _count_would_split_edges(pfield::ParticleField{R}, L_max::Real, only_coherent::Bool) where {R}
    graph = pfield.filament_edge_graph
    np = get_np(pfield)
    L_max_R = R(L_max)
    n = 0
    @inbounds for p in 1:np
        for k in 1:2
            q = graph.down_neighbor[k, p]
            q == 0 && continue
            (only_coherent && !graph.down_coherent[k, p]) && continue
            _would_split_edge(pfield, p, q, L_max_R) && (n += 1)
        end
    end
    return n
end

function _would_exact_coarsen_midpoint(pfield::ParticleField{R}, m::Int,
                                       L_min_R::R, only_coherent::Bool,
                                       atol_R::R) where {R}
    graph = pfield.filament_edge_graph
    up_count(graph, m) == 1 || return false
    down_count(graph, m) == 1 || return false
    p = graph.up_neighbor[1, m]
    q = graph.down_neighbor[1, m]
    np = get_np(pfield)
    (p == 0 || q == 0 || p == q || p > np || q > np) && return false
    kpm = find_down_slot(graph, p, m)
    kmq = find_down_slot(graph, m, q)
    (kpm == 0 || kmq == 0 || find_down_slot(graph, p, q) != 0) && return false
    coh_pm = graph.down_coherent[kpm, p]
    coh_mq = graph.down_coherent[kmq, m]
    (coh_pm != coh_mq) && return false
    only_coherent && (!coh_pm || !coh_mq) && return false
    abs(graph.down_score[kpm, p] - graph.down_score[kmq, m]) > atol_R && return false

    xp1 = pfield.particles[X_INDEX.start,     p]
    xp2 = pfield.particles[X_INDEX.start + 1, p]
    xp3 = pfield.particles[X_INDEX.start + 2, p]
    xm1 = pfield.particles[X_INDEX.start,     m]
    xm2 = pfield.particles[X_INDEX.start + 1, m]
    xm3 = pfield.particles[X_INDEX.start + 2, m]
    xq1 = pfield.particles[X_INDEX.start,     q]
    xq2 = pfield.particles[X_INDEX.start + 1, q]
    xq3 = pfield.particles[X_INDEX.start + 2, q]
    σp = pfield.particles[SIGMA_INDEX, p]
    σm = pfield.particles[SIGMA_INDEX, m]
    σq = pfield.particles[SIGMA_INDEX, q]
    σbar = (σp + σq) / R(2)
    dx1 = xq1 - xp1; dx2 = xq2 - xp2; dx3 = xq3 - xp3
    L2 = dx1*dx1 + dx2*dx2 + dx3*dx3
    thresh = L_min_R * σbar
    L2 > thresh * thresh && return false
    abs(xm1 - (xp1 + xq1) / R(2)) > atol_R && return false
    abs(xm2 - (xp2 + xq2) / R(2)) > atol_R && return false
    abs(xm3 - (xp3 + xq3) / R(2)) > atol_R && return false
    abs(σm - σbar) > atol_R && return false

    gp1 = pfield.particles[GAMMA_INDEX.start,     p]
    gp2 = pfield.particles[GAMMA_INDEX.start + 1, p]
    gp3 = pfield.particles[GAMMA_INDEX.start + 2, p]
    gm1 = pfield.particles[GAMMA_INDEX.start,     m]
    gm2 = pfield.particles[GAMMA_INDEX.start + 1, m]
    gm3 = pfield.particles[GAMMA_INDEX.start + 2, m]
    gq1 = pfield.particles[GAMMA_INDEX.start,     q]
    gq2 = pfield.particles[GAMMA_INDEX.start + 1, q]
    gq3 = pfield.particles[GAMMA_INDEX.start + 2, q]
    abs(gm1 - (gp1 + gq1) / R(2)) <= atol_R || return false
    abs(gm2 - (gp2 + gq2) / R(2)) <= atol_R || return false
    abs(gm3 - (gp3 + gq3) / R(2)) <= atol_R || return false
    return true
end

function _count_would_exact_coarsen(pfield::ParticleField{R},
                                    L_min::Real, only_coherent::Bool,
                                    atol::Real) where {R}
    n = 0
    L_min_R = R(L_min)
    atol_R = R(atol)
    @inbounds for m in 1:get_np(pfield)
        _would_exact_coarsen_midpoint(pfield, m, L_min_R, only_coherent, atol_R) && (n += 1)
    end
    return n
end

function _would_bundle_converge(pfield::ParticleField{R}, a::Int, b::Int, c::Int,
                                overlap_R::R, cos_tol::R, sigma_tol_R::R,
                                only_coherent::Bool) where {R}
    graph = pfield.filament_edge_graph
    a == b && return false
    np = get_np(pfield)
    (a < 1 || a > np || b < 1 || b > np || c < 1 || c > np) && return false
    (a == c || b == c) && return false
    down_count(graph, a) == 1 || return false
    down_count(graph, b) == 1 || return false
    k_ac = find_down_slot(graph, a, c)
    k_bc = find_down_slot(graph, b, c)
    (k_ac == 0 || k_bc == 0) && return false
    coh_ac = graph.down_coherent[k_ac, a]
    coh_bc = graph.down_coherent[k_bc, b]
    coh_ac == coh_bc || return false
    only_coherent && !coh_ac && return false
    L, σa, σb = _filament_edge_length_sigma(pfield, a, b)
    σmin = min(σa, σb); σmax = max(σa, σb)
    L <= overlap_R * σmin || return false
    σmax > 0 || return false
    abs(σa - σb) <= sigma_tol_R * σmax || return false
    gd, ga2, gb2 = _gamma_norm_dot(pfield, a, b)
    (ga2 > 0 && gb2 > 0) || return false
    gd >= cos_tol * sqrt(ga2 * gb2) || return false
    u, v = a < b ? (a, b) : (b, a)
    new_ups = 0
    for k in 1:2
        w = graph.up_neighbor[k, v]
        w == 0 && continue
        w == u && continue
        find_down_slot(graph, w, u) == 0 && (new_ups += 1)
    end
    return up_count(graph, u) + new_ups <= 2
end

function _would_bundle_diverge(pfield::ParticleField{R}, a::Int, b::Int, c::Int,
                               overlap_R::R, cos_tol::R, sigma_tol_R::R,
                               only_coherent::Bool) where {R}
    graph = pfield.filament_edge_graph
    b == c && return false
    np = get_np(pfield)
    (a < 1 || a > np || b < 1 || b > np || c < 1 || c > np) && return false
    (a == b || a == c) && return false
    up_count(graph, b) == 1 || return false
    up_count(graph, c) == 1 || return false
    k_ab = find_down_slot(graph, a, b)
    k_ac = find_down_slot(graph, a, c)
    (k_ab == 0 || k_ac == 0) && return false
    coh_ab = graph.down_coherent[k_ab, a]
    coh_ac = graph.down_coherent[k_ac, a]
    coh_ab == coh_ac || return false
    only_coherent && !coh_ab && return false
    L, σb, σc = _filament_edge_length_sigma(pfield, b, c)
    σmin = min(σb, σc); σmax = max(σb, σc)
    L <= overlap_R * σmin || return false
    σmax > 0 || return false
    abs(σb - σc) <= sigma_tol_R * σmax || return false
    gd, gb2, gc2 = _gamma_norm_dot(pfield, b, c)
    (gb2 > 0 && gc2 > 0) || return false
    gd >= cos_tol * sqrt(gb2 * gc2) || return false
    u, v = b < c ? (b, c) : (c, b)
    new_downs = 0
    for k in 1:2
        w = graph.down_neighbor[k, v]
        w == 0 && continue
        w == u && continue
        find_down_slot(graph, u, w) == 0 && (new_downs += 1)
    end
    return down_count(graph, u) + new_downs <= 2
end

function _count_would_bundle_coarsen(pfield::ParticleField{R},
                                     bundle_overlap::Real,
                                     bundle_angle_tol::Real,
                                     bundle_sigma_tol::Real,
                                     only_coherent::Bool) where {R}
    graph = pfield.filament_edge_graph
    overlap_R = R(bundle_overlap)
    cos_tol = R(cos(bundle_angle_tol))
    sigma_tol_R = R(bundle_sigma_tol)
    n = 0
    @inbounds for i in 1:get_np(pfield)
        if up_count(graph, i) == 2
            _would_bundle_converge(pfield, graph.up_neighbor[1, i], graph.up_neighbor[2, i], i,
                                   overlap_R, cos_tol, sigma_tol_R, only_coherent) && (n += 1)
        end
        if down_count(graph, i) == 2
            _would_bundle_diverge(pfield, i, graph.down_neighbor[1, i], graph.down_neighbor[2, i],
                                  overlap_R, cos_tol, sigma_tol_R, only_coherent) && (n += 1)
        end
    end
    return n
end

function _observe_inference_candidates!(pfield::ParticleField{R};
                                        max_eta::Real,
                                        angle_tol::Real,
                                        axis::Symbol,
                                        candidate_cap::Int) where {R}
    np = get_np(pfield)
    np < 2 && return (capped=0, visits=0, pairs=0, accepted=0, mutual=0)
    ws = pfield.filament_edge_workspace
    resize!(ws.axis_x, np); resize!(ws.axis_y, np); resize!(ws.axis_z, np)
    resize!(ws.axis_ok, np); fill!(ws.axis_ok, false)
    resize!(ws.visits, np); fill!(ws.visits, 0)
    resize!(ws.capped, np); fill!(ws.capped, false)
    if size(ws.down_best, 2) < np
        ws.down_best = zeros(Int, 2, np); ws.up_best = zeros(Int, 2, np)
        ws.down_best_score = zeros(R, 2, np); ws.up_best_score = zeros(R, 2, np)
    else
        @inbounds for i in 1:np
            ws.down_best[1, i] = 0; ws.down_best[2, i] = 0
            ws.up_best[1, i] = 0; ws.up_best[2, i] = 0
            ws.down_best_score[1, i] = zero(R); ws.down_best_score[2, i] = zero(R)
            ws.up_best_score[1, i] = zero(R); ws.up_best_score[2, i] = zero(R)
        end
    end

    cand = ws.candidates
    empty!(cand)
    xmin = ymin = zmin = Inf
    sigma_max = zero(R)
    @inbounds for i in 1:np
        get_static(pfield, i) && continue
        ex, ey, ez, ok = _filament_axis_unit(pfield, i, axis)
        ok || continue
        ws.axis_x[i] = ex; ws.axis_y[i] = ey; ws.axis_z[i] = ez
        ws.axis_ok[i] = true
        push!(cand, i)
        x = pfield.particles[1, i]; y = pfield.particles[2, i]; z = pfield.particles[3, i]
        σ = pfield.particles[SIGMA_INDEX, i]
        xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z)
        sigma_max = max(sigma_max, R(σ))
    end
    length(cand) < 2 && return (capped=0, visits=0, pairs=0, accepted=0, mutual=0)
    sigma_max > 0 || return (capped=0, visits=0, pairs=0, accepted=0, mutual=0)

    cell_size = R(max_eta) * sigma_max
    resize!(ws.keys, np)
    n_cells = _build_cell_list!(ws.sorted_indices, ws.offsets, ws.counts, ws.keys,
                                cand, pfield, cell_size, (xmin, ymin, zmin))

    inv_cell = inv(cell_size)
    cos_tol = R(cos(angle_tol))
    max_eta_R = R(max_eta)
    pairs = 0
    accepted = 0
    @inbounds for p in cand
        xp = pfield.particles[1, p]; yp = pfield.particles[2, p]; zp = pfield.particles[3, p]
        sp = R(pfield.particles[SIGMA_INDEX, p])
        ex_p = ws.axis_x[p]; ey_p = ws.axis_y[p]; ez_p = ws.axis_z[p]
        ix = _cell_coord(xp - xmin, inv_cell)
        iy = _cell_coord(yp - ymin, inv_cell)
        iz = _cell_coord(zp - zmin, inv_cell)
        capped_p = false
        for (dx, dy, dz) in STENCIL_OFFSETS_27
            capped_p && break
            jx = ix + dx; jy = iy + dy; jz = iz + dz
            (jx < 0 || jx > CELL_COORD_MAX || jy < 0 || jy > CELL_COORD_MAX || jz < 0 || jz > CELL_COORD_MAX) && continue
            key = _pack_cell_key(jx, jy, jz)
            for slot in _cell_range(ws.offsets, ws.counts, n_cells, key)
                q = ws.sorted_indices[slot]
                q <= p && continue
                ws.axis_ok[q] || continue
                ws.visits[p] += 1
                pairs += 1
                if ws.visits[p] > candidate_cap
                    ws.capped[p] = true
                    capped_p = true
                    break
                end
                xq = pfield.particles[1, q]; yq = pfield.particles[2, q]; zq = pfield.particles[3, q]
                sq = R(pfield.particles[SIGMA_INDEX, q])
                rx = xq - xp; ry = yq - yp; rz = zq - zp
                dist = sqrt(rx*rx + ry*ry + rz*rz)
                dist > 0 || continue
                eta = dist / max(sp, sq)
                eta <= max_eta_R || continue
                ex_q = ws.axis_x[q]; ey_q = ws.axis_y[q]; ez_q = ws.axis_z[q]
                cos_axes = ex_p*ex_q + ey_p*ey_q + ez_p*ez_q
                abs(cos_axes) >= cos_tol || continue
                proj_p = (rx*ex_p + ry*ey_p + rz*ez_p) / dist
                abs(proj_p) >= cos_tol || continue
                score = abs(cos_axes) / eta
                accepted += 1
                if proj_p > 0
                    _best2_insert!(ws.down_best, ws.down_best_score, p, q, score)
                    _best2_insert!(ws.up_best, ws.up_best_score, q, p, score)
                else
                    _best2_insert!(ws.up_best, ws.up_best_score, p, q, score)
                    _best2_insert!(ws.down_best, ws.down_best_score, q, p, score)
                end
            end
        end
    end
    mutual = 0
    @inbounds for p in cand
        for k in 1:2
            q = ws.down_best[k, p]
            q == 0 && continue
            (ws.up_best[1, q] == p || ws.up_best[2, q] == p) && (mutual += 1)
        end
    end
    visits = 0
    capped = 0
    @inbounds for i in 1:np
        visits += ws.visits[i]
        ws.capped[i] && (capped += 1)
    end
    return (capped=capped, visits=visits, pairs=pairs, accepted=accepted, mutual=mutual)
end

function _count_cross_merge_observations(pfield::ParticleField{R};
                                         r_merge::Real,
                                         max_sigma_ratio::Real,
                                         cross_angle_tol::Real,
                                         candidate_cap::Int,
                                         skip_static::Bool) where {R}
    np = get_np(pfield)
    np < 2 && return 0
    cos_tol = R(cos(cross_angle_tol))
    n = 0
    @inbounds for i in 1:np-1
        skip_static && get_static(pfield, i) && continue
        visits_i = 0
        xi = pfield.particles[X_INDEX.start, i]
        yi = pfield.particles[X_INDEX.start + 1, i]
        zi = pfield.particles[X_INDEX.start + 2, i]
        σi = pfield.particles[SIGMA_INDEX, i]
        for j in i+1:np
            skip_static && get_static(pfield, j) && continue
            visits_i += 1
            visits_i > candidate_cap && break
            σj = pfield.particles[SIGMA_INDEX, j]
            σmin = min(σi, σj); σmax = max(σi, σj)
            σmin > 0 || continue
            σmax / σmin <= max_sigma_ratio || continue
            dx = pfield.particles[X_INDEX.start, j] - xi
            dy = pfield.particles[X_INDEX.start + 1, j] - yi
            dz = pfield.particles[X_INDEX.start + 2, j] - zi
            thresh = R(r_merge) * σmin
            dx*dx + dy*dy + dz*dz <= thresh * thresh || continue
            gd, gi2, gj2 = _gamma_norm_dot(pfield, i, j)
            (gi2 > 0 && gj2 > 0) || continue
            gd >= cos_tol * sqrt(gi2 * gj2) || continue
            n += 1
        end
    end
    return n
end

"""
    calibrate_filament_edges(pfield; kwargs...) -> FilamentCalibrationReport

Measure current filament-edge topology and read-only candidate decisions.
The function does not add/remove graph edges and does not change particle
topology. Workspace buffers may be reused internally for bounded candidate
observation.
"""
function calibrate_filament_edges(pfield::ParticleField{R};
                                  max_eta::Real = 2.0,
                                  angle_tol::Real = π/4,
                                  axis::Symbol = :strength,
                                  candidate_cap::Int = 16,
                                  L_max::Real = 1.5,
                                  L_min::Real = 0.75,
                                  only_coherent::Bool = true,
                                  coarsen_atol::Real = sqrt(eps(R)),
                                  bundle_overlap::Real = 0.5,
                                  bundle_angle_tol::Real = angle_tol,
                                  bundle_sigma_tol::Real = 0.25,
                                  cross_r_merge::Real = 0.5,
                                  cross_angle_tol::Real = angle_tol,
                                  cross_max_sigma_ratio::Real = 2.0,
                                  cross_candidate_cap::Int = candidate_cap,
                                  skip_static::Bool = true) where {R}
    graph = pfield.filament_edge_graph
    np = get_np(pfield)
    active = 0
    coherent = 0
    hist = ntuple(_ -> 0, 9)
    @inbounds for i in 1:np
        u = up_count(graph, i)
        d = down_count(graph, i)
        idx = 1 + 3u + d
        hist = Base.setindex(hist, hist[idx] + 1, idx)
        for k in 1:2
            q = graph.down_neighbor[k, i]
            q == 0 && continue
            active += 1
            graph.down_coherent[k, i] && (coherent += 1)
        end
    end
    obs = _observe_inference_candidates!(pfield;
                                         max_eta=max_eta,
                                         angle_tol=angle_tol,
                                         axis=axis,
                                         candidate_cap=candidate_cap)
    split_n = _count_would_split_edges(pfield, L_max, only_coherent)
    exact_n = _count_would_exact_coarsen(pfield, L_min, only_coherent, coarsen_atol)
    bundle_n = _count_would_bundle_coarsen(pfield, bundle_overlap, bundle_angle_tol,
                                           bundle_sigma_tol, only_coherent)
    cross_n = _count_cross_merge_observations(pfield;
                                              r_merge=cross_r_merge,
                                              max_sigma_ratio=cross_max_sigma_ratio,
                                              cross_angle_tol=cross_angle_tol,
                                              candidate_cap=cross_candidate_cap,
                                              skip_static=skip_static)
    settings = (
        max_eta=R(max_eta),
        angle_tol=R(angle_tol),
        axis=axis,
        candidate_cap=candidate_cap,
        L_max=R(L_max),
        L_min=R(L_min),
        only_coherent=only_coherent,
        coarsen_atol=R(coarsen_atol),
        bundle_overlap=R(bundle_overlap),
        bundle_angle_tol=R(bundle_angle_tol),
        bundle_sigma_tol=R(bundle_sigma_tol),
        cross_r_merge=R(cross_r_merge),
        cross_angle_tol=R(cross_angle_tol),
        cross_max_sigma_ratio=R(cross_max_sigma_ratio),
        cross_candidate_cap=cross_candidate_cap,
    )
    return FilamentCalibrationReport{R}(
        np, active, coherent, active - coherent, hist,
        obs.capped, obs.visits, obs.pairs, obs.accepted, obs.mutual,
        split_n, exact_n, bundle_n, cross_n,
        _edge_summary_stats(pfield), settings, validate_filament_edges(pfield),
    )
end

"""
    filament_calibration_sweep(make_case, grid; kwargs...)

Run [`calibrate_filament_edges`](@ref) over a grid of cases. `make_case`
is called as `make_case(spec)` for each element of `grid`, and must return
a `ParticleField`.
"""
function filament_calibration_sweep(make_case, grid; kwargs...)
    reports = FilamentCalibrationReport[]
    for spec in grid
        push!(reports, calibrate_filament_edges(make_case(spec); kwargs...))
    end
    return reports
end
