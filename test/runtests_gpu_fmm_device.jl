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
function fmm048_sorted_sfs_brute(B, out, n; transposed=true)
    K1 = 1 / (2pi)^1.5
    rc2 = eltype(B) === Float32 ? 42.25 : 81.0
    E = zeros(3, n)
    Threads.@threads for i in 1:n
        xi = Float64(B[1, i]); yi = Float64(B[2, i]); zi = Float64(B[3, i])
        J5 = Float64(out[5, i]); J6 = Float64(out[6, i]); J7 = Float64(out[7, i])
        J8 = Float64(out[8, i]); J9 = Float64(out[9, i]); J10 = Float64(out[10, i])
        J11 = Float64(out[11, i]); J12 = Float64(out[12, i]); J13 = Float64(out[13, i])
        e1 = e2 = e3 = 0.0
        for j in 1:n
            i == j && continue
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

@testset "device-resident radix FMM: SFS device pass (task 048)" begin
    for (case, n, R) in (("cube", 20000, Float64), ("wake", 20000, Float64),
                         ("cube", 20000, Float32))
        cpu = fmm034_build(case, n; R=R)
        gpu = fmm034_to_gpu(cpu, R)
        gpu_ref = fmm034_to_gpu(cpu, R)

        # reference: validated direct-sum U/J + the direct-sum GPU Estr
        vpm_fmm.UJ_direct(gpu_ref)
        FLOWVPM.Estr_direct!(gpu_ref)       # ext gpu_estr_direct! kernel

        # resident lifecycle with the in-graph SFS pass + delivery
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        err = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n)
        e_sfs = fmm034_sfs_relrms(gpu.particles, gpu_ref.particles, n)
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
        ffmm._host_sfs_zeta_pairs!(om, q, tg, B, cr, dt, ds, nd)
        E_ulist = zeros(TF, 3, nb)
        ffmm._host_sfs_form_e!(E_ulist, om, q, out, true, nb)
        # sorted -> global permute for comparison with the delivered SFS rows
        perm = Array(view(state.body_perm, 1:nb))
        bsys = Array(view(state.body_system_ids, 1:nb))
        bidx = Array(view(state.body_indices, 1:nb))
        Eg = zeros(TF, 3, nb)
        ffmm._scatter_sfs_host!(Eg, E_ulist, perm, bsys, bidx, 1, nb)
        S = Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:nb]
        e_kernel = fmm048_relrms(S, Eg)
        E_full = fmm048_sorted_sfs_brute(B, out, nb)
        e_trunc = fmm048_relrms(E_ulist, E_full)
        @info "device radix SFS [$case n=$n $R]" err.u_rel_rms err.j_rel_rms e_sfs e_kernel e_trunc
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
        @test e_sfs <= e_gate

        # counter contract unchanged with SFS armed
        st = FLOWVPM._radix_fmm_couplings[gpu]
        counters = st.cache.state.counters
        @test counters.body_uploads == 0
        @test counters.expansion_host_copies == 0

        # steady state: counters flat + zero device allocation after warmup
        # (the SFS kernels use only construction-time buffers)
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        base = (counters.route_uploads, counters.operator_uploads,
                counters.influence_downloads)
        alloc_base = CUDA.@allocated vpm_fmm.UJ_fmm(gpu)
        alloc_sfs = CUDA.@allocated vpm_fmm.UJ_fmm(gpu; sfs=true)
        @test (counters.route_uploads, counters.operator_uploads,
               counters.influence_downloads) == base
        @info "device radix SFS [$case n=$n $R] steady-state device alloc (bytes)" alloc_base alloc_sfs
        # the SFS pass + delivery must add no steady-state device allocation
        # over the UJ-only step (construction-time buffers only)
        @test alloc_sfs <= alloc_base

        # graph-replay parity: perturb positions in place (stays inside the
        # box, same occupancy epoch with high probability -> replayed graph)
        # and compare against a fresh direct reference
        dx = R(1e-4) .* (CUDA.rand(R, 3, n) .- R(0.5))
        view(gpu.particles, vpm_fmm.X_INDEX, 1:n) .+= dx
        view(gpu_ref.particles, vpm_fmm.X_INDEX, 1:n) .+= dx
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        FLOWVPM._reset_particles(gpu_ref)
        FLOWVPM._reset_particles_sfs(gpu_ref)
        vpm_fmm.UJ_direct(gpu_ref)
        FLOWVPM.Estr_direct!(gpu_ref)
        e_replay = fmm034_sfs_relrms(gpu.particles, gpu_ref.particles, n)
        err_replay = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n)
        @info "device radix SFS replay parity [$case n=$n $R]" e_replay err_replay.j_rel_rms
        # same occupancy epoch (tiny perturbation), so the measured ζ-
        # truncation share and the field's E/J amplification above still apply
        @test e_replay <= max(R === Float64 ? 1e-3 : 3e-3,
                              20 * err_replay.j_rel_rms + 2 * e_trunc)

        # sfs=false evaluations skip delivery (rows untouched)
        S = copy(Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n])
        vpm_fmm.UJ_fmm(gpu)                 # default sfs=false
        @test Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n] == S
    end
end
