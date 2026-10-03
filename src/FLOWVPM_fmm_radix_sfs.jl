#=##############################################################################
# DESCRIPTION
    The subfilter-scale vortex-stretching estimator E_str, its analytic
    core-scaling derivatives (the two-level dynamic procedure) and the SFS
    repass, computed over the near field of FastMultipole's resident radix
    lifecycle. FLOWVPM owns every equation, every scratch array and the
    delivery into the particle rows; FastMultipole only hands out its sorted
    near-field data (`FastMultipole.radix_nearfield`) at the point of the
    lifecycle where the resident bodies' U and J are complete
    (`fmm!(...; nearfield_pass)`). Moved here from FastMultipole on
    2026-09-27 (maintainer request); the host loops are verbatim, the device
    kernels live in ext/FLOWVPMGPUExt.jl.

    Per evaluation: (a) T_i = op(J_i) Γ_i and the ζ-accumulators zeroed,
    (b) one sweep of the direct pair list accumulating Ω_i = Σ_j ζ_σj(r_ij) Γ_j
    and Q_i = Σ_j ζ_σj(r_ij) T_j (self pair skipped, it cancels exactly in E),
    (c) after the lifecycle returns, E_i = op(J_i) Ω_i − Q_i, de-permuted
    sorted -> global and ACCUMULATED into SFS_INDEX. Γ is packed rows 5:7, the
    raw σ row 8 (source σ convention), J is output rows 5:13 in FLOWVPM's
    J[(j-1)*3 + i] order.

    ζ_σ(r) = K1 exp(-ρ²/2)/σ³ with ρ = r/σ and K1 = (2π)^{-3/2}, the
    gaussianerf regularization's ζ. The saturation cutoff ρ² ≤ rc² drops
    contributions below ~1e-9 (F32) / ~1e-18 (F64) of ζ(0); it exists so the
    device kernel's exp is never fed huge arguments and host/device agree.
=###############################################################################

const _SFS_ZETA_K1 = 0.06349363593424097  # (2π)^(-3/2)
@inline _sfs_saturation_rc2(::Type{Float32}) = 42.25f0
@inline _sfs_saturation_rc2(::Type{Float64}) = 81.0

# Target-owned device sweeps (one thread per target body, walking its cell's
# pairs) instead of one workgroup per direct pair: no atomics, so a run
# reproduces bit for bit; the host loops are unaffected.
const _SFS_TARGET_MAJOR = Ref{Bool}(true)

"""
    _radix_sfs_context(pfield, maxn) -> NamedTuple

The persistent accumulators of the SFS pass, on the field's own array type
(host `Matrix` or the device), sized once at the cache capacity: `tg` (op(J)Γ,
then E), `om`, `q` (the ζ-pair sums), and the core-scaling channel `dj` (∂J/∂α),
`dt`, `dom`, `dq`; `e_buf` (3 x maxn) and `de_buf` (6 x maxn) receive the
de-permuted global-order results for delivery.
"""
function _radix_sfs_context(pfield::ParticleField{R}, maxn::Int) where R
    z(r) = fill!(similar(pfield.particles, R, r, maxn), zero(R))
    return (; tg=z(3), om=z(3), q=z(3), dj=z(9), dt=z(3), dom=z(3), dq=z(3),
              e_buf=z(3), de_buf=z(6), transposed=pfield.transposed)
end

# the context of the coupling `st`, created on first use
function _radix_sfs_context!(pfield::ParticleField, st)
    ctx = st.sfs[]
    if ctx === nothing || size(ctx.tg, 2) < pfield.maxparticles
        ctx = _radix_sfs_context(pfield, pfield.maxparticles)
        st.sfs[] = ctx
    end
    return ctx
end

"""
    _radix_sfs_pass!(pfield, ctx, nf; dsigma=false)

Stages (a) and (b) over the near field `nf` (`FastMultipole.radix_nearfield`):
called from inside `fmm!` through `nearfield_pass`, when the resident bodies'
U and J are complete and before any extra source is added. With `dsigma` the
core-scaling channel runs its own ∂J pair sweep and the ∂ζ sweep.
"""
function _radix_sfs_pass!(pfield::ParticleField, ctx, nf; dsigma::Bool=false)
    nf.device && return _radix_sfs_pass_device!(pfield, ctx, nf; dsigma)
    size(nf.output, 1) >= 13 || throw(AssertionError("the SFS pass requires the 13-row (hessian) output"))
    n = nf.n_bodies
    _host_sfs_tg_and_zero!(ctx.tg, ctx.om, ctx.q, nf.output, nf.source_bodies, ctx.transposed, n)
    _host_sfs_zeta_pairs!(ctx.om, ctx.q, ctx.tg, nf.source_bodies, nf.cell_ranges,
        nf.direct_targets, nf.direct_sources, nf.n_direct, 9)
    if dsigma
        _host_sfs_dj_pairs!(ctx.dj, nf.source_bodies, nf.cell_ranges, nf.direct_targets,
            nf.direct_sources, nf.n_direct, n, 9)
        _host_sfs_dsigma_tg_and_zero!(ctx.dt, ctx.dom, ctx.dq, ctx.dj, nf.source_bodies, ctx.transposed, n)
        _host_sfs_dzeta_pairs!(ctx.dom, ctx.dq, ctx.tg, ctx.dt, nf.source_bodies, nf.cell_ranges,
            nf.direct_targets, nf.direct_sources, nf.n_direct, 9)
    end
    return nothing
end

"""
    _radix_sfs_deliver!(pfield, ctx, nf; dsigma=false)

Stage (c), after `fmm!` returned: E (and ∂E) formed in sorted order, de-permuted
to global particle order and delivered: E ACCUMULATES into `SFS_INDEX` (the
caller owns the SFS reset); the core-scaling channel REPLACES `M[1:3]` with
L = (Γ⋅∇)∂U/∂α and `M[4:6]` with ∂E/∂α (the two-level procedure owns those rows
between its beforeUJ and afterUJ).
"""
function _radix_sfs_deliver!(pfield::ParticleField, ctx, nf; dsigma::Bool=false)
    nf.device && return _radix_sfs_deliver_device!(pfield, ctx, nf; dsigma)
    n = nf.n_bodies; np = pfield.np
    _host_sfs_form_e!(ctx.tg, ctx.om, ctx.q, nf.output, ctx.transposed, n)
    # ∂E needs Ω intact: formed after E, before any buffer is reused
    dsigma && _host_sfs_form_de!(ctx.dq, ctx.dj, ctx.om, ctx.dom, nf.output, ctx.transposed, n)
    e = view(ctx.e_buf, :, 1:np)
    _scatter_sfs_host!(e, ctx.tg, nf.host_body_perm, nf.host_body_system_ids, nf.host_body_indices, 1, n)
    view(pfield.particles, SFS_INDEX, 1:np) .+= e
    if dsigma
        de = view(ctx.de_buf, :, 1:np)
        _scatter_dsfs_host!(de, ctx.dt, ctx.dq, nf.host_body_perm, nf.host_body_system_ids, nf.host_body_indices, 1, n)
        m0 = first(M_INDEX)
        view(pfield.particles, m0:m0+5, 1:np) .= de
        _sfs_dsigma_delivered!(pfield, true)
    end
    return nothing
end

# The particles' current U (rows 10:12) and J (16:24) into the resident output
# (rows 2:4 and 5:13) in sorted body order, for a repass. Host arrays here; the
# GPU extension has the kernel.
# host storage only: the device method (ext) dispatches on the field's GPU array
# type; a bare `::ParticleField` here out-ranked it (2026-09-27)
function _radix_output_from_particles!(pfield::ParticleField{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Matrix}, nf)
    P = pfield.particles; output = nf.output
    perm = nf.host_body_perm; n = nf.n_bodies
    u0 = first(U_INDEX); j0 = first(J_INDEX)
    @inbounds for sorted_i in 1:n
        i = perm[sorted_i]
        output[2, sorted_i] = P[u0, i]; output[3, sorted_i] = P[u0 + 1, i]; output[4, sorted_i] = P[u0 + 2, i]
        for k in 0:8
            output[5 + k, sorted_i] = P[j0 + k, i]
        end
    end
    return output
end

"""
    sfs_repass!(pfield)

Recompute the SFS estimator (`SFS_INDEX`) from the particles' CURRENT velocity
gradient over the direct pairs of the last radix evaluation. For a caller that
reused that evaluation's U/J and then added another source's field to them.
"""
function sfs_repass!(pfield::ParticleField)
    st = get(_radix_fmm_couplings, pfield, nothing)
    st === nothing && error("sfs_repass!: no radix evaluation on record for this field")
    _reset_particles_sfs(pfield)
    nf = fmm.radix_nearfield(st.cache)
    nf.n_bodies > 0 || return nothing
    dsigma = _sfs_dsigma_requested(pfield)
    ctx = _radix_sfs_context!(pfield, st)
    _radix_output_from_particles!(pfield, nf)
    _radix_sfs_pass!(pfield, ctx, nf; dsigma)
    _radix_sfs_deliver!(pfield, ctx, nf; dsigma)
    return nothing
end

# device methods: ext/FLOWVPMGPUExt.jl
_radix_sfs_pass_device!(pfield, ctx, nf; dsigma=false) =
    error("the SFS pass on a device field needs the FLOWVPM GPU extension (load KernelAbstractions and GPUArraysCore)")
_radix_sfs_deliver_device!(pfield, ctx, nf; dsigma=false) =
    error("the SFS delivery on a device field needs the FLOWVPM GPU extension")

#------- the pass, verbatim from FastMultipole's host mirror of the device pass -------#
@inline function _sfs_apply_op(J5, J6, J7, J8, J9, J10, J11, J12, J13,
        v1, v2, v3, transposed::Bool)
    if transposed
        return (J5 * v1 + J6 * v2 + J7 * v3,
                J8 * v1 + J9 * v2 + J10 * v3,
                J11 * v1 + J12 * v2 + J13 * v3)
    else
        return (J5 * v1 + J8 * v2 + J11 * v3,
                J6 * v1 + J9 * v2 + J12 * v3,
                J7 * v1 + J10 * v2 + J13 * v3)
    end
end

function _host_sfs_tg_and_zero!(tg, om, q, output::AbstractMatrix{TF},
        source_bodies, transposed::Bool, n::Int) where TF
    @inbounds for i in 1:n
        g1 = source_bodies[5, i]
        g2 = source_bodies[6, i]
        g3 = source_bodies[7, i]
        t1, t2, t3 = _sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], g1, g2, g3, transposed)
        tg[1, i] = t1; tg[2, i] = t2; tg[3, i] = t3
        om[1, i] = zero(TF); om[2, i] = zero(TF); om[3, i] = zero(TF)
        q[1, i] = zero(TF); q[2, i] = zero(TF); q[3, i] = zero(TF)
    end
    return tg
end

function _host_sfs_zeta_pairs!(om::AbstractMatrix{TF}, q, tg, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        active_row::Int=0) where TF
    rc2 = _sfs_saturation_rc2(TF)
    K1 = TF(_SFS_ZETA_K1)
    half = TF(0.5)
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            active_row != 0 && iszero(source_bodies[active_row, i]) && continue
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            o1 = zero(TF); o2 = zero(TF); o3 = zero(TF)
            q1 = zero(TF); q2 = zero(TF); q3 = zero(TF)
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                active_row != 0 && iszero(source_bodies[active_row, j]) && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                sigma = source_bodies[8, j]
                rho2 = r2 / (sigma * sigma)
                if rho2 <= rc2
                    z = K1 * exp(-half * rho2) / (sigma * sigma * sigma)
                    o1 += z * source_bodies[5, j]
                    o2 += z * source_bodies[6, j]
                    o3 += z * source_bodies[7, j]
                    q1 += z * tg[1, j]
                    q2 += z * tg[2, j]
                    q3 += z * tg[3, j]
                end
            end
            om[1, i] += o1; om[2, i] += o2; om[3, i] += o3
            q[1, i] += q1; q[2, i] += q2; q[3, i] += q3
        end
    end
    return om
end

#------- analytic core-scaling derivative for the dynamic SFS procedure (2026-09-21) -------#
#
# The dynamic procedure's coefficient is C = <Γ⋅L>/<Γ⋅m> with L = (Γ⋅∇)∂U/∂σ and
# m = σ³/ζ(0) ∂E/∂σ: the derivatives of the resolved stretching and of the
# estimator with respect to a UNIFORM scaling σ → ασ of every core, at α = 1
# (Alvarez 2022 §4.7, two-level procedure). The pseudo-three-level
# implementation approximates them by a finite difference over a second full
# evaluation at 0.999σ; here they are accumulated exactly over the same direct
# pairs. With ρ = r/σ_j and G(ρ) = ρ g′(ρ) = A ρ³ e^{−ρ²/2}, A = √(2/π):
#   ∂g/∂α = −G,   ∂h/∂α = ρ² G   (h = ρg′ − 3g),   ∂ζ_σ/∂α = (ρ² − 3) ζ_σ,
# so ∂J/∂α reuses the U/J pair assembly with (g, h) → (−G, ρ²G), and from
# E_i = op(J_i) Ω_i − Q_i,
#   ∂E_i = op(∂J_i) Ω_i + op(J_i) ∂Ω_i − ∂Q_i,
#   ∂Ω_i = Σ_j ∂ζ Γ_j,   ∂Q_i = Σ_j (∂ζ T_j + ζ ∂T_j),   ∂T_j = op(∂J_j) Γ_j = L_j.
# The far field (multipoles) carries no σ dependence, so the derivative is a
# near-field quantity; the ζ saturation cutoff bounds it (G < 3e-7 beyond).
# Sources include the static bodies (they induce velocity); a source with
# σ = 0 (an oversize-masked particle) contributes nothing, matching its
# absence from the ζ sweep. Delivered as a 6-row slab: rows 1:3 L = ∂T,
# rows 4:6 ∂E, delivered into M[1:6] by `_radix_sfs_deliver!`.

function _host_sfs_dj_pairs!(dj::AbstractMatrix{TF}, source_bodies, cell_ranges,
        direct_targets, direct_sources, n_direct::Int, n::Int,
        active_row::Int=0) where TF
    rc2 = _sfs_saturation_rc2(TF)
    A = TF(fmm._GAUSSERF_A)
    half = TF(0.5)
    fill!(view(dj, :, 1:n), zero(TF))
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            active_row != 0 && iszero(source_bodies[active_row, i]) && continue
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            d1 = zero(TF); d2 = zero(TF); d3 = zero(TF)
            d4 = zero(TF); d5 = zero(TF); d6 = zero(TF)
            d7 = zero(TF); d8 = zero(TF); d9 = zero(TF)
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                sigma = source_bodies[8, j]
                sigma > zero(TF) || continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                r2 == zero(TF) && continue
                rho2 = r2 / (sigma * sigma)
                rho2 <= rc2 || continue
                invr = inv(sqrt(r2))
                G = A * rho2 * sqrt(rho2) * exp(-half * rho2)
                _, _, _, _, h1, h2, h3, h4, h5, h6, h7, h8, h9 = fmm._vortex_pair_ugh(
                    dx, dy, dz, r2, invr, source_bodies[5, j], source_bodies[6, j],
                    source_bodies[7, j], -G, rho2 * G)
                d1 += h1; d2 += h2; d3 += h3
                d4 += h4; d5 += h5; d6 += h6
                d7 += h7; d8 += h8; d9 += h9
            end
            dj[1, i] += d1; dj[2, i] += d2; dj[3, i] += d3
            dj[4, i] += d4; dj[5, i] += d5; dj[6, i] += d6
            dj[7, i] += d7; dj[8, i] += d8; dj[9, i] += d9
        end
    end
    return dj
end

# ∂T_i = op(∂J_i) Γ_i (= L_i) and the ∂Ω/∂Q accumulators zeroed
function _host_sfs_dsigma_tg_and_zero!(dt, dom, dq, dj::AbstractMatrix{TF},
        source_bodies, transposed::Bool, n::Int) where TF
    @inbounds for i in 1:n
        t1, t2, t3 = _sfs_apply_op(
            dj[1, i], dj[2, i], dj[3, i], dj[4, i], dj[5, i], dj[6, i],
            dj[7, i], dj[8, i], dj[9, i],
            source_bodies[5, i], source_bodies[6, i], source_bodies[7, i], transposed)
        dt[1, i] = t1; dt[2, i] = t2; dt[3, i] = t3
        dom[1, i] = zero(TF); dom[2, i] = zero(TF); dom[3, i] = zero(TF)
        dq[1, i] = zero(TF); dq[2, i] = zero(TF); dq[3, i] = zero(TF)
    end
    return dt
end

# ∂Ω_i = Σ_j ∂ζ Γ_j and ∂Q_i = Σ_j (∂ζ T_j + ζ ∂T_j) over the ζ sweep's pairs
function _host_sfs_dzeta_pairs!(dom::AbstractMatrix{TF}, dq, tg, dt, source_bodies,
        cell_ranges, direct_targets, direct_sources, n_direct::Int,
        active_row::Int=0) where TF
    rc2 = _sfs_saturation_rc2(TF)
    K1 = TF(_SFS_ZETA_K1)
    half = TF(0.5)
    three = TF(3)
    @inbounds for pair_i in 1:n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tcount = cell_ranges[2, target_cell]
        sfirst = cell_ranges[1, source_cell]
        scount = cell_ranges[2, source_cell]
        for i in tfirst:(tfirst + tcount - 1)
            active_row != 0 && iszero(source_bodies[active_row, i]) && continue
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            o1 = zero(TF); o2 = zero(TF); o3 = zero(TF)
            q1 = zero(TF); q2 = zero(TF); q3 = zero(TF)
            for j in sfirst:(sfirst + scount - 1)
                i == j && continue
                active_row != 0 && iszero(source_bodies[active_row, j]) && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                sigma = source_bodies[8, j]
                rho2 = r2 / (sigma * sigma)
                if rho2 <= rc2
                    z = K1 * exp(-half * rho2) / (sigma * sigma * sigma)
                    dz_ = z * (rho2 - three)
                    o1 += dz_ * source_bodies[5, j]
                    o2 += dz_ * source_bodies[6, j]
                    o3 += dz_ * source_bodies[7, j]
                    q1 += dz_ * tg[1, j] + z * dt[1, j]
                    q2 += dz_ * tg[2, j] + z * dt[2, j]
                    q3 += dz_ * tg[3, j] + z * dt[3, j]
                end
            end
            dom[1, i] += o1; dom[2, i] += o2; dom[3, i] += o3
            dq[1, i] += q1; dq[2, i] += q2; dq[3, i] += q3
        end
    end
    return dom
end

# ∂E_i = op(∂J_i) Ω_i + op(J_i) ∂Ω_i − ∂Q_i, formed in place into `dq`
function _host_sfs_form_de!(dq, dj, om, dom, output::AbstractMatrix,
        transposed::Bool, n::Int)
    @inbounds for i in 1:n
        a1, a2, a3 = _sfs_apply_op(
            dj[1, i], dj[2, i], dj[3, i], dj[4, i], dj[5, i], dj[6, i],
            dj[7, i], dj[8, i], dj[9, i], om[1, i], om[2, i], om[3, i], transposed)
        b1, b2, b3 = _sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], dom[1, i], dom[2, i], dom[3, i], transposed)
        dq[1, i] = a1 + b1 - dq[1, i]
        dq[2, i] = a2 + b2 - dq[2, i]
        dq[3, i] = a3 + b3 - dq[3, i]
    end
    return dq
end

# sorted -> global permute of the (L, ∂E) pair into one 6-row per-system buffer
function _scatter_dsfs_host!(buf::AbstractMatrix, dt, de, perm, body_system,
        body_index, isys::Int, n::Int)
    fill!(buf, zero(eltype(buf)))
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        body_system[global_i] == isys || continue
        ibody = body_index[global_i]
        buf[1, ibody] = dt[1, sorted_i]
        buf[2, ibody] = dt[2, sorted_i]
        buf[3, ibody] = dt[3, sorted_i]
        buf[4, ibody] = de[1, sorted_i]
        buf[5, ibody] = de[2, sorted_i]
        buf[6, ibody] = de[3, sorted_i]
    end
    return buf
end

"""
    _run_host_radix_sfs!(state)

Host mirror of the device SFS pass (task 048): TG precompute + ζ pair sweep
over the full direct list. Requires a state built with `sfs=true` (13-row
output). Call after `run_host_radix_lifecycle!` (U/J complete), before
`finalize_radix_sfs_output!`.
"""
function _host_sfs_form_e!(tg, om, q, output::AbstractMatrix,
        transposed::Bool, n::Int)
    @inbounds for i in 1:n
        e1, e2, e3 = _sfs_apply_op(
            output[5, i], output[6, i], output[7, i], output[8, i],
            output[9, i], output[10, i], output[11, i], output[12, i],
            output[13, i], om[1, i], om[2, i], om[3, i], transposed)
        tg[1, i] = e1 - q[1, i]
        tg[2, i] = e2 - q[2, i]
        tg[3, i] = e3 - q[3, i]
    end
    return tg
end

# sorted -> global permute of the 3-row E slab into a per-system buffer
function _scatter_sfs_host!(buf::AbstractMatrix, e, perm, body_system,
        body_index, isys::Int, n::Int)
    fill!(buf, zero(eltype(buf)))
    @inbounds for sorted_i in 1:n
        global_i = perm[sorted_i]
        body_system[global_i] == isys || continue
        ibody = body_index[global_i]
        buf[1, ibody] = e[1, sorted_i]
        buf[2, ibody] = e[2, sorted_i]
        buf[3, ibody] = e[3, sorted_i]
    end
    return buf
end

