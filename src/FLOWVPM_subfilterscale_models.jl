#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence models for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
=###############################################################################

"""
    Model of vortex-stretching SFS contributions evaluated with direct
particle-to-particle interactions. See 20210901 notebook for derivation.
"""
@inline function Estr_direct(target_particle, source_particle, r, zeta, transposed)
    GS = get_Gamma(source_particle)
    JS = get_J(source_particle)
    JT = get_J(target_particle)

    # Stretching term
    if transposed
        # Transposed scheme (Γq⋅∇')(Up - Uq)
        S1 = (JT[1] - JS[1])*GS[1]+(JT[2] - JS[2])*GS[2]+(JT[3] - JS[3])*GS[3]
        S2 = (JT[4] - JS[4])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[6] - JS[6])*GS[3]
        S3 = (JT[7] - JS[7])*GS[1]+(JT[8] - JS[8])*GS[2]+(JT[9] - JS[9])*GS[3]
    else
        # Classic scheme (Γq⋅∇)(Up - Uq)
        S1 = (JT[1] - JS[1])*GS[1]+(JT[4] - JS[4])*GS[2]+(JT[7] - JS[7])*GS[3]
        S2 = (JT[2] - JS[2])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[8] - JS[8])*GS[3]
        S3 = (JT[3] - JS[3])*GS[1]+(JT[6] - JS[6])*GS[2]+(JT[9] - JS[9])*GS[3]
    end

    sigma_inv = 1.0 / get_sigma(source_particle)[]
    zeta_sgm = zeta(r*sigma_inv) * sigma_inv * sigma_inv * sigma_inv

    # Add ζ_σ (Γq⋅∇)(Up - Uq)
    get_SFS(target_particle)[1] += zeta_sgm*S1
    get_SFS(target_particle)[2] += zeta_sgm*S2
    get_SFS(target_particle)[3] += zeta_sgm*S3
end

"""
    gpu_estr_direct!(pfield::ParticleField)

GPU implementation of the O(N²) direct-sum SFS vortex-stretching
contribution used by `Estr_direct!`, dispatched to when `pfield.particles`
is not a plain `Array`. Real implementation is a method of this function
defined in `ext/FLOWVPMGPUExt.jl` (loaded automatically alongside KernelAbstractions and a GPU package).
This stub is only reached if a non-`Array` particle field is used without
KernelAbstractions and a GPU package loaded.
"""
gpu_estr_direct!(pfield::ParticleField) = error(
    "No GPU Estr_direct implementation available for particle arrays of type " *
    "$(typeof(pfield.particles)). Load `using CUDA` alongside FLOWVPM to enable " *
    "the GPU direct SFS stretching contribution for `CuArray`-backed particle fields.")

function Estr_direct!(pfield)
    if pfield.particles isa Array
        if Threads.nthreads() > 1
            Estr_direct_multithreaded(pfield)
        else
            Estr_direct_singlethreaded(pfield)
        end
    else
        gpu_estr_direct!(pfield)
    end
end

function Estr_direct_multithreaded(pfield::ParticleField)
    # An empty field would build a zero-step assignment range below.
    pfield.np > 0 || return nothing
    n_per_thread, rem = divrem(pfield.np, Threads.nthreads())
    n = n_per_thread + (rem > 0)
    assignments = 1:n:pfield.np

    Threads.@threads for i_assignment in eachindex(assignments)
        start_idx = assignments[i_assignment]
        end_idx = min(start_idx + n - 1, pfield.np)

        # Calculate SFS contributions for the assigned particles
        for i_target in start_idx:end_idx
            target_particle = get_particle(pfield, i_target)
            is_static(target_particle) && continue
            tx, ty, tz = target_particle[1], target_particle[2], target_particle[3]

            for source_particle in iterator(pfield)
                sx, sy, sz = source_particle[1], source_particle[2], source_particle[3]

                dx, dy, dz = sx - tx, sy - ty, sz - tz
                r = sqrt(dx * dx + dy * dy + dz * dz)

                Estr_direct(target_particle, source_particle, r, pfield.kernel.zeta, pfield.transposed)
            end
        end
    end
end

function Estr_direct_singlethreaded(pfield::ParticleField)
    for target_particle in iterator(pfield)
        is_static(target_particle) && continue
        tx, ty, tz = target_particle[1], target_particle[2], target_particle[3]

        for source_particle in iterator(pfield)
            sx, sy, sz = source_particle[1], source_particle[2], source_particle[3]

            dx, dy, dz = sx - tx, sy - ty, sz - tz
            r = sqrt(dx * dx + dy * dy + dz * dz)

            Estr_direct(target_particle, source_particle, r, pfield.kernel.zeta, pfield.transposed)
        end
    end
end

function Estr_fmm!(target_pfield::ParticleField, source_pfield::ParticleField, target_tree, source_tree, direct_list;
        i_target_system::Int=1, i_source_system::Int=1)
    if Threads.nthreads() > 1
        Estr_fmm_multithread!(target_pfield, source_pfield, target_tree, source_tree, direct_list;
            i_target_system, i_source_system)
    else
        Estr_fmm_singlethread!(target_pfield, source_pfield, target_tree, source_tree, direct_list;
            i_target_system, i_source_system)
    end
end

function Estr_fmm_multithread!(target_pfield::ParticleField, source_pfield::ParticleField, target_tree, source_tree, direct_list;
        i_target_system::Int=1, i_source_system::Int=1)

    # total number of interactions
    n_interactions = FastMultipole.get_n_interactions(1, target_tree.branches, 1, source_tree.branches, direct_list)
    n_threads = Threads.nthreads()

    # interactions per thread
    n_per_thread, rem = divrem(n_interactions, n_threads)
    rem > 0 && (n_per_thread += 1)
    n_per_thread < 100 && (n_per_thread = 100)

    # create assignments
    assignments = Vector{UnitRange{Int64}}(undef,n_threads)
    for i in eachindex(assignments)
        assignments[i] = 1:0
    end
    FastMultipole.make_direct_assignments!(assignments, 1, target_tree.branches, 1, source_tree.branches, direct_list, n_threads, n_per_thread, nothing)

    Threads.@threads for i_task in eachindex(assignments)
        assignment = assignments[i_task]

        # evaluate
        for i_interaction in assignment
            i_target, i_source = direct_list[i_interaction]
            
            target_index = target_tree.branches[i_target].bodies_index[i_target_system]
            source_index = source_tree.branches[i_source].bodies_index[i_source_system]

            # loop over source particles
            for i_source in source_index
                source_particle = get_particle(source_pfield, source_tree.sort_index_list[i_source_system][i_source])
                is_static(source_particle) && continue

                # source position
                sx, sy, sz = source_particle[1], source_particle[2], source_particle[3]

                # loop over target particles
                for i_target in target_index
                    target_particle = get_particle(target_pfield, target_tree.sort_index_list[i_target_system][i_target])
                    is_static(target_particle) && continue

                    # target position
                    tx, ty, tz = target_particle[1], target_particle[2], target_particle[3]

                    # separation distance
                    dx, dy, dz = sx - tx, sy - ty, sz - tz
                    r = sqrt(dx * dx + dy * dy + dz * dz)

                    # add Estr contribution
                    Estr_direct(target_particle, source_particle, r, source_pfield.kernel.zeta, source_pfield.transposed)

                end
            end
        end
    end
end

function Estr_fmm_singlethread!(target_pfield::ParticleField, source_pfield::ParticleField, target_tree, source_tree, direct_list;
        i_target_system::Int=1, i_source_system::Int=1)

    for (i_target, i_source) in direct_list

        target_index = target_tree.branches[i_target].bodies_index[i_target_system]
        source_index = source_tree.branches[i_source].bodies_index[i_source_system]

        # loop over source particles
        for i_source in source_index
            source_particle = get_particle(source_pfield, source_tree.sort_index_list[i_source_system][i_source])
            is_static(source_particle) && continue

            # source position
            sx, sy, sz = source_particle[1], source_particle[2], source_particle[3]

            # loop over target particles
            for i_target in target_index
                target_particle = get_particle(target_pfield, target_tree.sort_index_list[i_target_system][i_target])
                is_static(target_particle) && continue

                # target position
                tx, ty, tz = target_particle[1], target_particle[2], target_particle[3]

                # separation distance
                dx, dy, dz = sx - tx, sy - ty, sz - tz
                r = sqrt(dx * dx + dy * dy + dz * dz)

                # add Estr contribution
                Estr_direct(target_particle, source_particle, r, source_pfield.kernel.zeta, source_pfield.transposed)

            end
        end
    end
end

"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    UJ_fmm(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end

"""
    SFS model wrapper that hides the static particles from the model in order
to avoid potential numerical instabilities encountered at solid surfaces.
"""
function E_nostaticparticles(pfield, args...; E=Estr_fmm, optargs...)

    @assert pfield.np < pfield.maxparticles "Sorting of particles is needed"*
        " but all pre-allocated memory is already in use"

    org_np = pfield.np
    iaux = pfield.np + 1

    # Fetch auxiliary memory
    paux = get_particle(pfield, iaux; emptyparticle=true)

    # Iterate over particles
    for pi in pfield.np:-1:1

        # Fetch target particles
        p = get_particle(pfield, pi)

        # Case that we found a static particle
        if p.static[1]

            if pi==pfield.np
                nothing

            # Swap this particle with last particle
            else

                # Fetch last particle
                pnp = get_particle(pfield, pfield.np)

                # Store static particle in auxiliary memory
                fmm.overwriteBody(pfield.bodies, iaux-1, pi-1)
                paux.circulation .= p.circulation
                paux.C .= p.C
                paux.static .= p.static

                # Move last particle into the static particle's memory
                fmm.overwriteBody(pfield.bodies, pi-1, pfield.np-1)
                p.circulation .= pnp.circulation
                p.C .= pnp.C
                p.static .= pnp.static

                # Move static particle into the last particle's memory
                fmm.overwriteBody(pfield.bodies, pfield.np-1, iaux-1)
                pnp.circulation .= paux.circulation
                pnp.C .= paux.C
                pnp.static .= paux.static

            end

            # Move "end of array" pointer to hide the static particle
            pfield.np -= 1
        end
    end

    # Call SFS model without the static particles
    E(pfield, args...; optargs...)

    # Restore static particles back to the field
    # pfield.np = org_np

    # NOTE: Here we add the auxiliary memory to the field and then remove it.
    #       This is to make sure that the memory is cleaned and avoid potential
    #       bugs
    pfield.np = org_np + 1
    remove_particle(pfield, pfield.np)

    # # Sort particles to restore the original indexing
    # sort!(iterator(pfield), by = p->p.index[1])

end

################################################################################
# ANALYTIC CORE-SCALING DERIVATIVES (two-level dynamic procedure), all-pairs
# host reference. See FastMultipole/src/translate_batched_resident.jl
# ("analytic core-scaling derivative") for the derivation; this is the
# O(N²) reference the radix sweeps are gated against, and the fallback of
# `dynamicprocedure_twolevel_afterUJ` off the radix path.
################################################################################
"""
    dsigma_direct!(pfield)

Derivatives with respect to a uniform scaling σ → ασ of every particle core,
at α = 1, of the resolved stretching L = (Γ⋅∇)∂U/∂α (into `M[1:3]`) and of the
SFS vortex-stretching estimator ∂E_str/∂α (into `M[4:6]`), by direct
particle-to-particle sums over the whole field (no cutoff). Requires the
current velocity gradient in `J_INDEX`; gaussianerf kernel only. Sources of
∂J include the static particles; the ζ sums follow the radix SFS sweep
(non-static sources, self pair skipped).
"""
function dsigma_direct!(pfield::ParticleField{R}) where R
    pfield.particles isa Array || error("dsigma_direct! is a host (Array) reference")
    pfield.kernel.zeta === zeta_gauserf || error("dsigma_direct!: gaussianerf kernel only")
    np = pfield.np
    P = pfield.particles
    transposed = pfield.transposed
    A = R(sqrt(2 / pi))
    K1 = R((2pi)^(-1.5))
    dJ = zeros(R, 9, np)
    x0 = first(X_INDEX); g0 = first(GAMMA_INDEX); j0 = first(J_INDEX); m0 = first(M_INDEX)
    # ∂J/∂α
    Threads.@threads for i in 1:np
        P[STATIC_INDEX, i] != 0 && continue
        xi, yi, zi = P[x0, i], P[x0 + 1, i], P[x0 + 2, i]
        acc = ntuple(_ -> zero(R), 9)
        for j in 1:np
            j == i && continue
            sigma = P[SIGMA_INDEX, j]
            sigma > 0 || continue
            dx = xi - P[x0, j]; dy = yi - P[x0 + 1, j]; dz = zi - P[x0 + 2, j]
            r2 = dx * dx + dy * dy + dz * dz
            r2 == 0 && continue
            rho2 = r2 / (sigma * sigma)
            invr = inv(sqrt(r2))
            G = A * rho2 * sqrt(rho2) * exp(-rho2 / 2)
            h = fmm._vortex_pair_ugh(dx, dy, dz, r2, invr, P[g0, j], P[g0 + 1, j],
                P[g0 + 2, j], -G, rho2 * G)
            acc = ntuple(k -> acc[k] + h[4 + k], 9)
        end
        for k in 1:9
            dJ[k, i] = acc[k]
        end
    end
    # L = op(∂J)Γ, then ∂E = Σ_j [∂ζ op(J_i − J_j)Γ_j + ζ op(∂J_i − ∂J_j)Γ_j]
    Threads.@threads for i in 1:np
        P[STATIC_INDEX, i] != 0 && continue
        gi = (P[g0, i], P[g0 + 1, i], P[g0 + 2, i])
        L = fmm._sfs_apply_op(dJ[1, i], dJ[2, i], dJ[3, i], dJ[4, i], dJ[5, i], dJ[6, i],
            dJ[7, i], dJ[8, i], dJ[9, i], gi..., transposed)
        xi, yi, zi = P[x0, i], P[x0 + 1, i], P[x0 + 2, i]
        e1 = zero(R); e2 = zero(R); e3 = zero(R)
        for j in 1:np
            j == i && continue
            P[STATIC_INDEX, j] != 0 && continue
            sigma = P[SIGMA_INDEX, j]
            dx = xi - P[x0, j]; dy = yi - P[x0 + 1, j]; dz = zi - P[x0 + 2, j]
            rho2 = (dx * dx + dy * dy + dz * dz) / (sigma * sigma)
            z = K1 * exp(-rho2 / 2) / (sigma * sigma * sigma)
            dz_ = z * (rho2 - 3)
            gj = (P[g0, j], P[g0 + 1, j], P[g0 + 2, j])
            s = fmm._sfs_apply_op(ntuple(k -> P[j0 + k - 1, i] - P[j0 + k - 1, j], 9)...,
                gj..., transposed)
            ds = fmm._sfs_apply_op(ntuple(k -> dJ[k, i] - dJ[k, j], 9)..., gj..., transposed)
            e1 += dz_ * s[1] + z * ds[1]
            e2 += dz_ * s[2] + z * ds[2]
            e3 += dz_ * s[3] + z * ds[3]
        end
        P[m0, i] = L[1]; P[m0 + 1, i] = L[2]; P[m0 + 2, i] = L[3]
        P[m0 + 3, i] = e1; P[m0 + 4, i] = e2; P[m0 + 5, i] = e3
    end
    _sfs_dsigma_delivered!(pfield, true)
    return nothing
end

"""
    dsigma_fmm!(pfield, target_tree, source_tree, direct_list)

`dsigma_direct!` over the octree FMM's direct interaction list (the CPU
`UJ_fmm` path): same sums, restricted to the near-field pairs, which is where
all the σ dependence lives. Called by `UJ_fmm` when the two-level dynamic
procedure has requested the derivatives. ∂J is staged in `pfield.scratch`
rows 1:9 (sorted by particle index), L lands in `M[1:3]`, ∂E in `M[4:6]`.
"""
function dsigma_fmm!(pfield::ParticleField{R}, target_tree, source_tree, direct_list;
        i_target_system::Int=1, i_source_system::Int=1) where R
    pfield.kernel.zeta === zeta_gauserf || error("dsigma_fmm!: gaussianerf kernel only")
    np = pfield.np
    P = pfield.particles
    dJ = view(pfield.scratch, 1:9, 1:np)
    fill!(dJ, zero(R))
    x0 = first(X_INDEX); g0 = first(GAMMA_INDEX); j0 = first(J_INDEX); m0 = first(M_INDEX)
    A = R(sqrt(2 / pi)); K1 = R((2pi)^(-1.5))
    transposed = pfield.transposed
    tsort = target_tree.sort_index_list[i_target_system]
    ssort = source_tree.sort_index_list[i_source_system]
    assignments = _dsigma_assignments(target_tree, source_tree, direct_list)
    # pass 1: ∂J/∂α (static sources included, static targets skipped)
    Threads.@threads for i_task in eachindex(assignments)
        @inbounds for i_interaction in assignments[i_task]
            it, is = direct_list[i_interaction]
            for i_source in source_tree.branches[is].bodies_index[i_source_system]
                j = ssort[i_source]
                sigma = P[SIGMA_INDEX, j]
                sigma > 0 || continue
                gx, gy, gz = P[g0, j], P[g0 + 1, j], P[g0 + 2, j]
                sx, sy, sz = P[x0, j], P[x0 + 1, j], P[x0 + 2, j]
                for i_target in target_tree.branches[it].bodies_index[i_target_system]
                    i = tsort[i_target]
                    (i == j || P[STATIC_INDEX, i] != 0) && continue
                    dx = P[x0, i] - sx; dy = P[x0 + 1, i] - sy; dz = P[x0 + 2, i] - sz
                    r2 = dx * dx + dy * dy + dz * dz
                    r2 == 0 && continue
                    rho2 = r2 / (sigma * sigma)
                    G = A * rho2 * sqrt(rho2) * exp(-rho2 / 2)
                    h = fmm._vortex_pair_ugh(dx, dy, dz, r2, inv(sqrt(r2)), gx, gy, gz, -G, rho2 * G)
                    for k in 1:9
                        dJ[k, i] += h[4 + k]
                    end
                end
            end
        end
    end
    # L = op(∂J)Γ into M[1:3]; ∂E accumulators zeroed
    Threads.@threads for i in 1:np
        @inbounds begin
            L = fmm._sfs_apply_op(dJ[1, i], dJ[2, i], dJ[3, i], dJ[4, i], dJ[5, i], dJ[6, i],
                dJ[7, i], dJ[8, i], dJ[9, i], P[g0, i], P[g0 + 1, i], P[g0 + 2, i], transposed)
            P[m0, i] = L[1]; P[m0 + 1, i] = L[2]; P[m0 + 2, i] = L[3]
            P[m0 + 3, i] = 0; P[m0 + 4, i] = 0; P[m0 + 5, i] = 0
        end
    end
    # pass 2: ∂E = Σ_j [∂ζ op(J_i − J_j)Γ_j + ζ op(∂J_i − ∂J_j)Γ_j] (non-static pairs, as Estr_fmm!)
    Threads.@threads for i_task in eachindex(assignments)
        @inbounds for i_interaction in assignments[i_task]
            it, is = direct_list[i_interaction]
            for i_source in source_tree.branches[is].bodies_index[i_source_system]
                j = ssort[i_source]
                P[STATIC_INDEX, j] != 0 && continue
                sigma = P[SIGMA_INDEX, j]
                gj = (P[g0, j], P[g0 + 1, j], P[g0 + 2, j])
                sx, sy, sz = P[x0, j], P[x0 + 1, j], P[x0 + 2, j]
                for i_target in target_tree.branches[it].bodies_index[i_target_system]
                    i = tsort[i_target]
                    (i == j || P[STATIC_INDEX, i] != 0) && continue
                    dx = P[x0, i] - sx; dy = P[x0 + 1, i] - sy; dz = P[x0 + 2, i] - sz
                    rho2 = (dx * dx + dy * dy + dz * dz) / (sigma * sigma)
                    z = K1 * exp(-rho2 / 2) / (sigma * sigma * sigma)
                    dz_ = z * (rho2 - 3)
                    s = fmm._sfs_apply_op(ntuple(k -> P[j0 + k - 1, i] - P[j0 + k - 1, j], 9)..., gj..., transposed)
                    ds = fmm._sfs_apply_op(ntuple(k -> dJ[k, i] - dJ[k, j], 9)..., gj..., transposed)
                    P[m0 + 3, i] += dz_ * s[1] + z * ds[1]
                    P[m0 + 4, i] += dz_ * s[2] + z * ds[2]
                    P[m0 + 5, i] += dz_ * s[3] + z * ds[3]
                end
            end
        end
    end
    _sfs_dsigma_delivered!(pfield, true)
    return nothing
end

# the interaction ranges of `Estr_fmm_multithread!` (each thread owns whole
# target branches, so the accumulations above never race); one range when
# single-threaded
function _dsigma_assignments(target_tree, source_tree, direct_list)
    n_threads = Threads.nthreads()
    n_threads == 1 && return [1:length(direct_list)]
    n_interactions = FastMultipole.get_n_interactions(1, target_tree.branches, 1, source_tree.branches, direct_list)
    n_per_thread, rem = divrem(n_interactions, n_threads)
    rem > 0 && (n_per_thread += 1)
    n_per_thread < 100 && (n_per_thread = 100)
    assignments = Vector{UnitRange{Int64}}(undef, n_threads)
    for i in eachindex(assignments)
        assignments[i] = 1:0
    end
    FastMultipole.make_direct_assignments!(assignments, 1, target_tree.branches, 1, source_tree.branches, direct_list, n_threads, n_per_thread, nothing)
    return assignments
end
