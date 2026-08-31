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
        base = (counters.route_uploads, counters.operator_uploads,
                counters.influence_downloads)
        alloc = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        @test counters.body_uploads == 0
        @test counters.expansion_host_copies == 0
        @test (counters.route_uploads, counters.operator_uploads,
               counters.influence_downloads) == base
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

# Allocation contract, asserted at the layer that owns it (2026-08-22 trace;
# integration-api-spec "no per-step allocation" = construction-time buffers
# only, prefixes mutated per step — the framework satisfies this):
#  - the resident LIFECYCLE (graph replay = one CUDA.launch) must be
#    device-allocation-free and within a small host launch-bookkeeping cap;
#  - the full FLOWVPM WRAPPER performs ~25-30 GPU ops/step whose CUDA.jl
#    launch/broadcast bookkeeping allocates ~1-4 KB host each (fixed,
#    launch-count-scaled, NOT n-scaled: any n-sized download would add
#    >= 160 KB at n=2e4), plus <= 512 B device library scratch from two
#    accumulate! prefix scans (counting sort translate_batched_cuda.jl:229,
#    body prefix :6682) and one maximum reduction (geometry gate
#    translate_batched_resident.jl:2085). The wrapper contract is a fixed
#    band, no growth across steps, and SFS adding no device allocation.
const FMM048_HOST_ALLOC_BUDGET = 4096          # lifecycle layer
const FMM048_HOST_WRAPPER_BAND = 160_000       # base wrapper step (measured 98.6-107.2 KB)
const FMM048_HOST_WRAPPER_BAND_SFS = 192_000   # sfs wrapper step (measured 119.1-129.6 KB)
const FMM048_DEVICE_SCRATCH_BAND = 512         # CUDA.jl scan/reduce scratch (measured 272-400 B)

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
        ffmm = FLOWVPM.fmm
        B = Array(view(state.source_bodies, :, 1:nb))
        out = Array(view(state.output, :, 1:nb))
        cr = Array(state.cell_ranges)
        dt = Array(view(state.direct_targets, 1:nd))
        ds = Array(view(state.direct_sources, 1:nd))
        TF = eltype(B)
        tg = zeros(TF, 3, nb); om = zeros(TF, 3, nb); q = zeros(TF, 3, nb)
        ffmm._host_sfs_tg_and_zero!(tg, om, q, out, B, true, nb)
        ffmm._host_sfs_zeta_pairs!(om, q, tg, B, cr, dt, ds, nd, 9)
        E_ulist = zeros(TF, 3, nb)
        ffmm._host_sfs_form_e!(E_ulist, om, q, out, true, nb)
        # sorted -> global permute for comparison with the delivered SFS rows
        perm = Array(view(state.body_perm, 1:nb))
        bsys = Array(view(state.body_system_ids, 1:nb))
        bidx = Array(view(state.body_indices, 1:nb))
        Eg = zeros(TF, 3, nb)
        ffmm._scatter_sfs_host!(Eg, E_ulist, perm, bsys, bidx, 1, nb)
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

        # steady state: counters flat; wrapper-layer allocation band (see the
        # contract comment at FMM048_HOST_ALLOC_BUDGET). The lifecycle-layer
        # zero-allocation assertion runs after the graph identity checks
        # below, where the replay path is certain.
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        base = (counters.route_uploads, counters.operator_uploads,
                counters.influence_downloads)
        host_alloc_base = @allocated vpm_fmm.UJ_fmm(gpu)
        host_alloc_sfs = @allocated vpm_fmm.UJ_fmm(gpu; sfs=true)
        device_alloc_base = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        device_alloc_sfs = CUDA.@allocated vpm_fmm.UJ_fmm(gpu; sfs=true)
        host_alloc_base2 = @allocated vpm_fmm.UJ_fmm(gpu)
        device_alloc_base2 = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        @test (counters.route_uploads, counters.operator_uploads,
               counters.influence_downloads) == base
        @info "device radix SFS [$case n=$n P=$P $R rho_t=$rho_t] steady-state allocations (bytes)" host_alloc_base host_alloc_sfs device_alloc_base device_alloc_sfs
        @test host_alloc_base <= FMM048_HOST_WRAPPER_BAND
        @test host_alloc_sfs <= FMM048_HOST_WRAPPER_BAND_SFS
        @test host_alloc_base2 <= host_alloc_base + 4096   # no growth
        @test device_alloc_base <= FMM048_DEVICE_SCRATCH_BAND
        @test device_alloc_sfs <= device_alloc_base         # SFS adds none
        @test device_alloc_base2 <= device_alloc_base       # no growth

        # Deterministic same-state graph replay: eligibility, executable, and
        # epoch identity are all asserted before and after the replay.
        hctx = state.interaction_list
        @test ffmm._cuda_graph_eligible(state)
        vpm_fmm.UJ_fmm(gpu; sfs=true) # capture if the preceding warm call did not
        @test hctx.graph_exec !== nothing
        @test hctx.graph_epoch == hctx.epoch_id
        graph_exec = hctx.graph_exec
        graph_epoch = hctx.graph_epoch
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        @test hctx.graph_exec === graph_exec
        @test hctx.graph_epoch == graph_epoch == hctx.epoch_id
        # e_replay on active-column deltas only (same pattern as e_sfs
        # above): both fields hold the identical static sentinel, which
        # contributes 0 to the numerator but dominates the denominator at
        # this n (delivered |E| << |sentinel|), deflating a statics-inclusive
        # relrms into vacuousness. sfs=true resets active SFS every call, so
        # the delta from S_before is the per-call delivered E of the replay.
        S_replay = Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n]
        e_replay = fmm048_relrms((S_replay .- S_before)[:, active_indices],
                                 Sref_delta[:, active_indices])
        err_replay = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n;
            skip=static_indices)
        @info "device radix SFS replay parity [$case n=$n P=$P $R rho_t=$rho_t]" e_replay err_replay.u_rel_rms err_replay.j_rel_rms
        # same occupancy epoch (tiny perturbation), so the measured ζ-
        # truncation share and the field's E/J amplification above still apply
        @test e_replay <= max(R === Float64 ? 1e-3 : 3e-3,
                              20 * err_replay.j_rel_rms + 2 * e_trunc)
        # Replayed U and J must not regress vs the first (uncaptured)
        # evaluation: job 13298230 measured replay j_rel_rms ~ 0.11 vs
        # first-call 2e-3 (far-field content decorrelated on replay). The
        # 1.5x headroom covers atomic-order jitter, nothing else.
        @test err_replay.u_rel_rms <= 1.5 * err.u_rel_rms + eps(Float32)
        @test err_replay.j_rel_rms <= 1.5 * err.j_rel_rms + eps(Float32)

        # Lifecycle-layer allocation contract: with the graph live, one step
        # of the resident lifecycle is a single graph launch and must be
        # device-allocation-free within host launch bookkeeping.
        ffmm.run_cuda_radix_lifecycle!(state)
        lifecycle_host = @allocated ffmm.run_cuda_radix_lifecycle!(state)
        lifecycle_dev = CUDA.@allocated ffmm.run_cuda_radix_lifecycle!(state)
        @info "device radix SFS [$case n=$n P=$P $R rho_t=$rho_t] lifecycle-layer allocations (bytes)" lifecycle_host lifecycle_dev
        @test lifecycle_host <= FMM048_HOST_ALLOC_BUDGET
        @test lifecycle_dev == 0

        # Independent same-state replay parity: the uncaptured launch-sequence
        # body on the IDENTICAL state is ground truth; the replayed graph must
        # reproduce the DELIVERED particle U/J with a fixed tolerance that NO
        # measured error can scale. (The e_replay gate above scales with
        # j_rel_rms and is permissive exactly when replay is broken — this
        # gate is not.) The comparison is in GLOBAL particle order: the
        # sorted state.output slab is NOT comparable across calls because the
        # per-step counting-sort scatter assigns intra-cell slots via
        # atomic_add, i.e. the permutation is unstable — diag 13299959
        # measured slab relrms 0.02-0.04 from pure column shuffling on
        # identical input. Static columns are excluded (never-reset
        # accumulators with differing evaluation counts).
        uj_rows = [collect(vpm_fmm.U_INDEX); collect(vpm_fmm.J_INDEX)]
        vpm_fmm.UJ_fmm(gpu; sfs=true)       # replayed graph
        uj_replay = Array(gpu.particles)[uj_rows, active_indices]
        graph_flag_prior = ffmm.radix_setting(:CUDA_GRAPH_LIFECYCLE)
        replay_body_parity = try
            ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, false)
            vpm_fmm.UJ_fmm(gpu; sfs=true)   # uncaptured body, same state
            uj_body = Array(gpu.particles)[uj_rows, active_indices]
            fmm048_relrms(uj_replay, uj_body)
        finally
            ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, graph_flag_prior)
        end
        @info "device radix SFS replay-vs-body parity [$case n=$n P=$P $R rho_t=$rho_t]" replay_body_parity
        @test replay_body_parity <= (R === Float64 ? 1e-10 : 1e-4)

        # sfs=false must skip the pass itself, not merely delivery.
        S = copy(Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n])
        om_before = copy(Array(view(state.sfs.om, :, 1:n)))
        q_before = copy(Array(view(state.sfs.q, :, 1:n)))
        vpm_fmm.UJ_fmm(gpu)                 # default sfs=false
        @test Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n] == S
        @test Array(view(state.sfs.om, :, 1:n)) == om_before
        @test Array(view(state.sfs.q, :, 1:n)) == q_before
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
        # graph_live records whether the warmed calls actually replayed a
        # captured graph at this tiny operating point (if ineligible, e_warm
        # degenerates to re-running the body — recorded, not gated).
        strict_hctx = FLOWVPM._radix_fmm_couplings[gpu].cache.state.interaction_list
        graph_live = strict_hctx.graph_exec !== nothing
        @info "device radix SFS strict point [cube n=$n P=$P $R rho_t=$rho_t]" err.u_rel_rms err.j_rel_rms e_first e_warm strict_gate graph_live
        @test err.u_rel_rms <= FMM034_U_GATE
        @test e_first <= strict_gate
        @test e_warm <= strict_gate
    end
end
