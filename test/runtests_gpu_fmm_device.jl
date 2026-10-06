# Part B of the radix FMM coupling tests (task 034): the device-resident
# lifecycle on a CuArray-backed ParticleField. Included at runtime by
# runtests_gpu_fmm.jl only when CUDA is functional (this file uses CUDA macros,
# so it must never be macro-expanded on a CUDA-less machine). Requires the
# helpers (fmm034_build, fmm034_pfield, fmm034_uj_errors, FMM034_U_GATE)
# defined by runtests_gpu_fmm.jl.

import CUDA

# GPU copy of a CPU-built field (runtests_gpu.jl pattern), preserving solver
# settings and leaving spare capacity to exercise the capacity contract
function fmm034_to_gpu(cpu_pfield, R=Float64; extra_capacity=256)
    maxp = cpu_pfield.maxparticles + extra_capacity
    gpu = fmm034_pfield(maxp, R; arraytype=CUDA.CuArray, UJ=vpm_fmm.UJ_fmm)
    gpu.np = cpu_pfield.np
    view(gpu.particles, :, 1:cpu_pfield.np) .=
        CUDA.CuArray{R}(Array(cpu_pfield.particles)[:, 1:cpu_pfield.np])
    return gpu
end

@testset "device-resident radix FMM: static U/J vs direct" begin
    for (case, n, R) in (("cube", 20000, Float64), ("wake", 20000, Float64),
                         ("cube", 20000, Float32), ("wake", 20000, Float32))
        cpu = fmm034_build(case, n; R=R)
        gpu = fmm034_to_gpu(cpu, R)
        gpu_ref = fmm034_to_gpu(cpu, R)

        vpm_fmm.UJ_direct(gpu_ref)          # validated direct-sum GPU kernels
        vpm_fmm.UJ_fmm(gpu)                 # routes to the resident lifecycle
        err = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n)
        @info "device radix FMM [$case n=$n $R]" err.u_rel_rms err.j_rel_rms
        @test err.u_rel_rms <= FMM034_U_GATE
        @test err.j_rel_rms < 1e-1

        # 023 counter contract: a device-resident consumer never uploads
        # bodies or copies expansions through the host
        st = FLOWVPM._radix_fmm_couplings[gpu]
        counters = st.cache.state.counters
        @test counters.body_uploads == 0
        @test counters.expansion_host_copies == 0

        # counters stay flat across recurring evaluations (steady state)
        vpm_fmm.UJ_fmm(gpu)
        base = counters.influence_downloads
        alloc = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        @test counters.body_uploads == 0
        @test counters.expansion_host_copies == 0
        @test counters.influence_downloads == base
        @info "device radix FMM [$case n=$n $R] steady-state device alloc (bytes)" alloc

        # coarse warm solve wall time — order-of-magnitude sanity for the 035
        # campaign only, NOT a benchmark (unoptimized settings, single sample)
        CUDA.synchronize()
        t_solve = @elapsed begin
            vpm_fmm.UJ_fmm(gpu)
            CUDA.synchronize()
        end
        @info "device radix FMM [$case n=$n $R] warm U/J solve wall time (s), sanity only" t_solve

        # varying live count below capacity, same cache
        gpu.np -= 100
        gpu_ref.np -= 100
        vpm_fmm.UJ_direct(gpu_ref)
        vpm_fmm.UJ_fmm(gpu)
        @test counters.body_uploads == 0
        errv = fmm034_uj_errors(gpu.particles, gpu_ref.particles, gpu.np)
        @test errv.u_rel_rms <= FMM034_U_GATE
    end
end

@testset "device-resident radix FMM: multi-step RK3 dynamic run" begin
    n = 2000
    nsteps = 5
    dt = 1e-3
    cpu = fmm034_build("wake", n; UJ=vpm_fmm.UJ_direct)
    gpu = fmm034_to_gpu(cpu)   # UJ_fmm

    for _ in 1:nsteps
        # update_U_prev=false: the U_prev bookkeeping loop is scalar-indexed
        # (legacy autotune bookkeeping, unused on the radix path) and not
        # GPU-vectorized; disable on both sides for an identical comparison
        vpm_fmm.nextstep(cpu, dt; update_U_prev=false)
        vpm_fmm.nextstep(gpu, dt; update_U_prev=false)
    end

    A = Array(gpu.particles)
    B = Array(cpu.particles)
    pos_err = maximum(abs.(A[vpm_fmm.X_INDEX, 1:n] .- B[vpm_fmm.X_INDEX, 1:n]))
    err = fmm034_uj_errors(gpu.particles, cpu.particles, n)
    @info "device radix FMM dynamic run (wake n=$n, $nsteps RK3 steps)" pos_err err.u_rel_rms err.j_rel_rms
    @test pos_err <= 5e-5              # positions barely drift at this dt
    @test err.u_rel_rms <= 5e-3        # accumulated over 3*nsteps evaluations

    # counter contract held across the whole dynamic run
    st = FLOWVPM._radix_fmm_couplings[gpu]
    @test st.cache.state.counters.body_uploads == 0
    @test st.cache.state.counters.expansion_host_copies == 0
end

# All-pairs ζ brute force of E_str in the SORTED body frame, from the radix
# slabs themselves: B = packed source bodies (rows 1:3 pos, 5:7 Γ, 8 σ),
# out = lifecycle output (rows 5:13 = J). Accumulated in Float64 regardless of
# the slab precision; framework saturation cutoff matched. O(n²) on host —
# threaded, ~seconds at n = 2e4.
function fmm048_sorted_sfs_brute(B, out, n; transposed=true, active_row=0)
    K1 = 1 / (2pi)^1.5
    rc2 = eltype(B) === Float32 ? 42.25 : 81.0
    E = zeros(3, n)
    Threads.@threads for i in 1:n
        active_row != 0 && iszero(B[active_row, i]) && continue
        xi = Float64(B[1, i]); yi = Float64(B[2, i]); zi = Float64(B[3, i])
        J5 = Float64(out[5, i]); J6 = Float64(out[6, i]); J7 = Float64(out[7, i])
        J8 = Float64(out[8, i]); J9 = Float64(out[9, i]); J10 = Float64(out[10, i])
        J11 = Float64(out[11, i]); J12 = Float64(out[12, i]); J13 = Float64(out[13, i])
        e1 = e2 = e3 = 0.0
        for j in 1:n
            i == j && continue
            active_row != 0 && iszero(B[active_row, j]) && continue
            dx = xi - Float64(B[1, j]); dy = yi - Float64(B[2, j]); dz = zi - Float64(B[3, j])
            sig = Float64(B[8, j])
            rho2 = (dx * dx + dy * dy + dz * dz) / (sig * sig)
            rho2 <= rc2 || continue
            z = K1 * exp(-rho2 / 2) / (sig * sig * sig)
            G1 = Float64(B[5, j]); G2 = Float64(B[6, j]); G3 = Float64(B[7, j])
            K5 = Float64(out[5, j]); K6 = Float64(out[6, j]); K7 = Float64(out[7, j])
            K8 = Float64(out[8, j]); K9 = Float64(out[9, j]); K10 = Float64(out[10, j])
            K11 = Float64(out[11, j]); K12 = Float64(out[12, j]); K13 = Float64(out[13, j])
            if transposed
                e1 += z * ((J5 - K5) * G1 + (J6 - K6) * G2 + (J7 - K7) * G3)
                e2 += z * ((J8 - K8) * G1 + (J9 - K9) * G2 + (J10 - K10) * G3)
                e3 += z * ((J11 - K11) * G1 + (J12 - K12) * G2 + (J13 - K13) * G3)
            else
                e1 += z * ((J5 - K5) * G1 + (J8 - K8) * G2 + (J11 - K11) * G3)
                e2 += z * ((J6 - K6) * G1 + (J9 - K9) * G2 + (J12 - K12) * G3)
                e3 += z * ((J7 - K7) * G1 + (J10 - K10) * G2 + (J13 - K13) * G3)
            end
        end
        E[1, i] = e1; E[2, i] = e2; E[3, i] = e3
    end
    return E
end

fmm048_relrms(A, B) = sqrt(sum(abs2, Float64.(A) .- Float64.(B)) /
                           max(sum(abs2, Float64.(B)), eps()))

# Allocation: steady-state calls must not grow, and the SFS pass must add no
# device allocation. (Fixed byte bands belonged to the removed native CUDA
# lifecycle; the KA path's launch bookkeeping is a different, larger constant.)

@testset "device-resident radix FMM: SFS device pass (task 048)" begin
    specs = vec([("cube", 20000, R, P, rho_t)
                 for R in (Float64, Float32), P in (4, 8),
                     rho_t in (4.211, 4.789)])
    # One wake mechanism case per candidate; the production 210,056-particle
    # wake timing matrix lives in fm048_ab_benchmark.jl.
    append!(specs, [("wake", 20000, Float64, 4, rho_t)
                    for rho_t in (4.211, 4.789)])
    for (case, n, R, P, rho_t) in specs
        cpu = fmm034_build(case, n; R=R)
        static_indices = (2, 17, 201)
        for i in static_indices
            vpm_fmm.set_static(cpu, i, one(R))
            cpu.particles[vpm_fmm.SFS_INDEX, i] .=
                (R(3), R(-2), R(1))
        end
        gpu = fmm034_to_gpu(cpu, R)
        gpu_ref = fmm034_to_gpu(cpu, R)
        FLOWVPM.radix_fmm_settings!(gpu; expansion_order=P, rho_t)
        active_indices = setdiff(collect(1:n), collect(static_indices))
        S_before = Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n]
        Sref_before = Array(gpu_ref.particles)[vpm_fmm.SFS_INDEX, 1:n]

        # reference: validated direct-sum U/J + the direct-sum GPU Estr
        vpm_fmm.UJ_direct(gpu_ref)
        FLOWVPM.Estr_direct!(gpu_ref)       # ext gpu_estr_direct! kernel

        # resident U/J lifecycle followed by the requested SFS pass + delivery.
        # U/J parity excludes static columns: statics' U/J rows are
        # never-reset accumulators (see fmm034_uj_errors), and gpu vs gpu_ref
        # have differing evaluation counts.
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        err = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n;
            skip=static_indices)
        S_after = Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n]
        Sref_after = Array(gpu_ref.particles)[vpm_fmm.SFS_INDEX, 1:n]
        S_delta = S_after .- S_before
        Sref_delta = Sref_after .- Sref_before
        e_sfs = fmm048_relrms(S_delta[:, active_indices],
                              Sref_delta[:, active_indices])
        @test all(iszero, S_delta[:, collect(static_indices)])
        @test all(iszero, Sref_delta[:, collect(static_indices)])
        @test err.u_rel_rms <= FMM034_U_GATE

        # --- mechanical/physics decomposition (mirrors the host testset) ---
        # Download the radix slabs and rebuild the SFS result on host two ways:
        #   E_ulist: the HOST MIRROR of the pass over the SAME device pair
        #            list, from the SAME output J -> e_kernel isolates a
        #            device-kernel defect (gate 1e-6 F64 / 1e-4 F32).
        #   E_full:  all-pairs ζ brute force from the same J -> e_trunc is the
        #            U-list ζ-truncation share at this derived geometry
        #            (recorded, not gated: it is a geometry property, the
        #            design-anticipated default-list gap).
        state = FLOWVPM._radix_fmm_couplings[gpu].cache.state
        nb = state.counts.n_bodies
        @test nb == n
        nd = state.counts.n_direct
        B = Array(view(state.source_bodies, :, 1:nb))
        out = Array(view(state.output, :, 1:nb))
        cr = Array(state.cell_ranges)
        dt = Array(view(state.direct_targets, 1:nd))
        ds = Array(view(state.direct_sources, 1:nd))
        TF = eltype(B)
        tg = zeros(TF, 3, nb); om = zeros(TF, 3, nb); q = zeros(TF, 3, nb)
        vpm_fmm._host_sfs_tg_and_zero!(tg, om, q, out, B, true, nb)
        vpm_fmm._host_sfs_zeta_pairs!(om, q, tg, B, cr, dt, ds, nd, 9)
        E_ulist = zeros(TF, 3, nb)
        vpm_fmm._host_sfs_form_e!(E_ulist, om, q, out, true, nb)
        # sorted -> global permute for comparison with the delivered SFS rows
        perm = Array(view(state.body_perm, 1:nb))
        bsys = Array(view(state.body_system_ids, 1:nb))
        bidx = Array(view(state.body_indices, 1:nb))
        Eg = zeros(TF, 3, nb)
        vpm_fmm._scatter_sfs_host!(Eg, E_ulist, perm, bsys, bidx, 1, nb)
        e_kernel = fmm048_relrms(S_delta[:, active_indices],
                                 Eg[:, active_indices])
        E_full = fmm048_sorted_sfs_brute(B, out, nb; active_row=9)
        sorted_active = findall(!iszero, view(B, 9, :))
        e_trunc = fmm048_relrms(E_ulist[:, sorted_active],
                                E_full[:, sorted_active])
        # Source-mask sensitivity: turning the known-static sources back on
        # must change at least one active target's oracle increment.
        B_all_active = copy(B)
        B_all_active[9, :] .= one(TF)
        E_all_active = fmm048_sorted_sfs_brute(B_all_active, out, nb;
            active_row=9)
        @test E_full[:, sorted_active] != E_all_active[:, sorted_active]
        @info "device radix SFS [$case n=$n P=$P $R rho_t=$rho_t]" err.u_rel_rms err.j_rel_rms e_sfs e_kernel e_trunc
        # kernel-defect isolation: the device pass must reproduce its host
        # mirror over the identical pair list and J — this gate carries the
        # SFS correctness weight
        @test e_kernel <= (R === Float64 ? 1e-6 : 1e-4)
        # Physics vs the exact-erf direct reference is J-ERROR PROPAGATION,
        # not an SFS property: E is a cancellation-dominated difference
        # quantity, so the E/J error ratio is field-dependent — measured on
        # the HOST at exactly this operating point (2026-08-21): cube 2.05,
        # wake 14.1 (host wake e_sfs = 0.0293417 reproduced H200 job 13247540
        # 0.029342 to 5 digits — device == host, no kernel defect). The
        # ζ-truncation share is negligible here (measured 2.1e-6 cube /
        # 8.7e-7 wake). Gate with 20x j_rel headroom: still loud on any
        # mechanism-scale regression (an orientation/index bug puts e_sfs at
        # O(1)), while J regressions are gated by u_rel and e_kernel above.
        e_gate = max(R === Float64 ? 1e-3 : 3e-3,
                     20 * err.j_rel_rms + 2 * e_trunc)
        # The strict eps/2 tail-budget gates (5e-4 F64 / 1e-3 F32) were
        # derived at the host-matrix operating point (n=1500, widened shell,
        # min rho ~ 5.2), where the omitted tail is the ONLY budgeted error.
        # At THIS production-shaped operating point (n=2e4, derived shell)
        # delivered E is J-error-bound (E/J amplification 2.05 cube / 14.1
        # wake, D5/D7) and the strict gate is not attainable at any P here —
        # job 13298230 measured e_sfs = 4.3e-3 cube P4 / 2.9e-2 wake P4 with
        # a mechanically exact SFS pass (e_kernel ~ 1e-15). The strict gates
        # are enforced in their valid regime by the "strict tail-budget
        # operating point" testset below; production-settings accuracy/cost
        # selection is the fm048_tuning_sweep.jl Pareto sweep, gated on the
        # p018 field (user decision 2026-08-22). e_sfs is recorded above for
        # every case; the J-bound gate still fails loudly on any
        # mechanism-scale regression.
        @test e_sfs <= e_gate

        # counter contract unchanged with SFS armed
        st = FLOWVPM._radix_fmm_couplings[gpu]
        counters = st.cache.state.counters
        @test counters.body_uploads == 0
        @test counters.expansion_host_copies == 0

        # steady state: counters flat, allocations not growing
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        base = counters.influence_downloads
        host_alloc_base = @allocated vpm_fmm.UJ_fmm(gpu)
        host_alloc_sfs = @allocated vpm_fmm.UJ_fmm(gpu; sfs=true)
        device_alloc_base = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        device_alloc_sfs = CUDA.@allocated vpm_fmm.UJ_fmm(gpu; sfs=true)
        host_alloc_base2 = @allocated vpm_fmm.UJ_fmm(gpu)
        device_alloc_base2 = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        @test counters.influence_downloads == base
        @info "device radix SFS [$case n=$n P=$P $R rho_t=$rho_t] steady-state allocations (bytes)" host_alloc_base host_alloc_sfs device_alloc_base device_alloc_sfs
        @test host_alloc_base2 <= host_alloc_base + 4096   # no growth
        @test device_alloc_sfs <= device_alloc_base         # SFS adds none
        @test device_alloc_base2 <= device_alloc_base       # no growth

        # Repeat evaluation on the same state: the KA lifecycle and the SFS pass
        # have no order-dependent sums, so the delivered U, J and SFS rows repeat
        # in every bit (active columns; static columns are never-reset
        # accumulators with differing evaluation counts).
        rows = [collect(vpm_fmm.U_INDEX); collect(vpm_fmm.J_INDEX); collect(vpm_fmm.SFS_INDEX)]
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        first_eval = Array(gpu.particles)[rows, active_indices]
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        @test Array(gpu.particles)[rows, active_indices] == first_eval

        # sfs=false must skip the pass itself, not merely delivery.
        S = copy(Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n])
        sfs_ctx = FLOWVPM._radix_fmm_couplings[gpu].sfs[]   # FLOWVPM's SFS buffers
        om_before = copy(Array(view(sfs_ctx.om, :, 1:n)))
        q_before = copy(Array(view(sfs_ctx.q, :, 1:n)))
        vpm_fmm.UJ_fmm(gpu)                 # default sfs=false
        @test Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n] == S
        @test Array(view(sfs_ctx.om, :, 1:n)) == om_before
        @test Array(view(sfs_ctx.q, :, 1:n)) == q_before
    end
end

# Strict delivered-accuracy operating point (2026-08-22): the eps/2
# tail-budget gates (5e-4 F64 / 1e-3 F32) were DERIVED on the host candidate
# matrix at n=1500 with the widened shell (ell=2, near_radius2=20, min rho ~
# 5.2) and the conservative per-pair rho_t candidates — the regime where
# zeta truncation and far-field J truncation are negligible and the only
# budgeted error is the deliberately omitted tail. This testset enforces the
# same gates on the DEVICE path at the same operating point, on both the
# first (uncaptured) and warmed (graph-replayed) evaluations, so the strict
# evidence is preserved where its derivation is valid and a replay defect
# fails it with no error-scaled tolerance.
@testset "device-resident radix SFS: strict tail-budget operating point" begin
    n = 1500
    for R in (Float64, Float32), P in (4, 8), rho_t in (4.211, 4.789)
        cpu = fmm034_build("cube", n; R=R)
        gpu = fmm034_to_gpu(cpu, R)
        gpu_ref = fmm034_to_gpu(cpu, R)
        FLOWVPM.radix_fmm_settings!(gpu; expansion_order=P, ell=2,
            near_radius2=20, rho_t)
        vpm_fmm.UJ_direct(gpu_ref)
        FLOWVPM.Estr_direct!(gpu_ref)
        strict_gate = R === Float64 ? 5e-4 : 1e-3
        vpm_fmm.UJ_fmm(gpu; sfs=true)       # first: uncaptured body
        e_first = fmm034_sfs_relrms(gpu.particles, gpu_ref.particles, n)
        for _ in 1:3                        # warm: record, then replay
            vpm_fmm.UJ_fmm(gpu; sfs=true)
        end
        e_warm = fmm034_sfs_relrms(gpu.particles, gpu_ref.particles, n)
        err = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n)
        @info "device radix SFS strict point [cube n=$n P=$P $R rho_t=$rho_t]" err.u_rel_rms err.j_rel_rms e_first e_warm strict_gate
        @test err.u_rel_rms <= FMM034_U_GATE
        @test e_first <= strict_gate
        @test e_warm <= strict_gate
    end
end

# a 2% tail of 3-4x cores, on the device
function fmm034_tail_field(n, seed, big_scale)
    rng = MersenneTwister(seed)
    cpu = fmm034_pfield(n, Float32)
    s0 = 0.8f0 * (1f0 / n)^(1 / 3)
    for _ in 1:n
        big = rand(rng) < 0.02
        vpm_fmm.add_particle(cpu, rand(rng, Float32, 3), (2 .* rand(rng, Float32, 3) .- 1) ./ n,
            s0 * (big ? big_scale * (1f0 + rand(rng, Float32)) : 1f0 + 0.2f0 * rand(rng, Float32)))
    end
    return fmm034_to_gpu(cpu, Float32)
end

@testset "device large cores out of the leaf cells: no error beyond the tree's" begin
    # the large cores' source copies leave their leaf cells and come back through
    # their groups in the same direct list. Against the direct sum, U and J match
    # the same field with the tail shrunk to ordinary cores (the same grid, no
    # split); a grid sized for the largest core is coarser and so more accurate
    # (and slower), which says nothing about the split. The SFS estimator and the
    # core-scaling channel, built from J, differ from that coarser evaluation only
    # by the two grids' J difference.
    n = 20000
    function run(K; shrink=false)
        gpu = fmm034_tail_field(n, 7, shrink ? 1.1f0 / 2 : 3f0)
        vpm_fmm.radix_fmm_settings!(gpu; oversize_count = K)
        vpm_fmm._sfs_dsigma_request!(gpu, true)
        vpm_fmm.UJ_fmm_gpu!(gpu; reset = true, reset_sfs = true, sfs = true)
        nf = vpm_fmm.fmm.radix_nearfield(FLOWVPM._radix_fmm_couplings[gpu].cache)
        return gpu, Array(gpu.particles[:, 1:n]), nf.n_sources - nf.n_bodies
    end
    ga, A, kA = run(0)
    gc, _, kC = run(0; shrink = true)
    _, B, kB = run(-1)
    @test kA > 0 && kB == 0 && kC == 0
    @test FLOWVPM._radix_fmm_couplings[ga].cache.ell == FLOWVPM._radix_fmm_couplings[gc].cache.ell
    ref_a = fmm034_tail_field(n, 7, 3f0); vpm_fmm.UJ_direct(ref_a)
    ref_c = fmm034_tail_field(n, 7, 1.1f0 / 2); vpm_fmm.UJ_direct(ref_c)
    ea = fmm034_uj_errors(ga.particles, ref_a.particles, n)
    ec = fmm034_uj_errors(gc.particles, ref_c.particles, n)
    @info "large cores vs the same grid without them" kA ea.u_rel_rms ec.u_rel_rms ea.j_rel_rms ec.j_rel_rms
    @test ea.u_rel_rms <= 1.2 * ec.u_rel_rms + 1e-6
    @test ea.j_rel_rms <= 1.2 * ec.j_rel_rms + 1e-6
    rel(rows) = maximum(abs.(A[rows, :] .- B[rows, :])) / maximum(abs.(B[rows, :]))
    dj = rel(vpm_fmm.J_INDEX)
    m0 = first(vpm_fmm.M_INDEX)
    @test rel(vpm_fmm.SFS_INDEX) < 3 * dj + 1e-5
    @test rel(m0:m0+5) < 3 * dj + 1e-5
end

@testset "device radix state restore with large cores keeps the geometry" begin
    n = 20000
    gpu = fmm034_tail_field(n, 3, 4f0)
    vpm_fmm.radix_fmm_settings!(gpu; padding = 1.0, rectangular = true)
    evaluate() = (vpm_fmm._reset_particles(gpu); vpm_fmm.UJ_fmm_gpu!(gpu; reset = true);
                  Array(gpu.particles[:, 1:n]))
    cache() = FLOWVPM._radix_fmm_couplings[gpu].cache
    geometry() = (c = cache(); (c.ell, c.ell_axes, c.policy.near_radius2))
    evaluate(); live = geometry(); A = evaluate()
    nf = vpm_fmm.fmm.radix_nearfield(cache())
    @test nf.n_sources > nf.n_bodies
    vpm_fmm.restore_radix_state!(gpu, vpm_fmm.radix_state(gpu))
    @test geometry() == live
    @test evaluate() == A
end

@testset "device zeta_fmm with grown cores keeps the geometry" begin
    # core-spreading resets evaluate zeta while the grown cores are still in the
    # field: the cache takes them out of the leaf cells and sums them through their
    # coarser-level groups (own term included), so the geometry stays put
    n = 20000
    rng = MersenneTwister(3)
    cpu = fmm034_pfield(n, Float32)
    s0 = 0.8f0 * (1f0 / n)^(1 / 3)
    for _ in 1:n
        big = rand(rng) < 0.02
        vpm_fmm.add_particle(cpu, rand(rng, Float32, 3), (2 .* rand(rng, Float32, 3) .- 1) ./ n,
            s0 * (big ? 4f0 + 4f0 * rand(rng, Float32) : 1f0 + 0.2f0 * rand(rng, Float32)))
    end
    gpu = fmm034_to_gpu(cpu, Float32)
    vpm_fmm.UJ_fmm(gpu)
    geometry() = (c = FLOWVPM._radix_fmm_couplings[gpu].cache; (c.ell, c.ell_axes, c.policy.near_radius2))
    g0 = geometry()
    vpm_fmm.zeta_fmm(gpu)
    @test geometry() == g0
    vpm_fmm.zeta_direct(cpu)
    W = Array(gpu.particles[vpm_fmm.VORTICITY_INDEX, 1:n]); Wref = cpu.particles[vpm_fmm.VORTICITY_INDEX, 1:n]
    @test maximum(abs.(W .- Wref)) / maximum(abs.(Wref)) < 1e-4
end
