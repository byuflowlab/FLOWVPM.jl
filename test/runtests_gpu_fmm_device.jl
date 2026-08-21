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

@testset "device-resident radix FMM: SFS device pass (task 048)" begin
    for (case, n, R) in (("cube", 20000, Float64), ("wake", 20000, Float64),
                         ("cube", 20000, Float32))
        cpu = fmm034_build(case, n; R=R)
        gpu = fmm034_to_gpu(cpu, R)
        gpu_ref = fmm034_to_gpu(cpu, R)

        # reference: validated direct-sum U/J + the direct-sum GPU Estr
        vpm_fmm.UJ_direct(gpu_ref)
        FLOWVPM.Estr_direct!(gpu_ref)       # ext gpu_estr_direct! kernel

        # resident lifecycle with the in-graph SFS pass + delivery. E is built
        # from J, and the radix J carries the erf-free g/h nearfield
        # approximation plus the far-field regularization deficit
        # (P-independent; see the host SFS testset comment), so the E gate is
        # J-bound: max(base, 3 * j_rel_rms). The host testset carries the
        # tight (1e-6) mechanical-parity gate that isolates the SFS machinery.
        vpm_fmm.UJ_fmm(gpu; sfs=true)
        err = fmm034_uj_errors(gpu.particles, gpu_ref.particles, n)
        e_sfs = fmm034_sfs_relrms(gpu.particles, gpu_ref.particles, n)
        @info "device radix SFS [$case n=$n $R]" err.u_rel_rms err.j_rel_rms e_sfs
        @test err.u_rel_rms <= FMM034_U_GATE
        @test e_sfs <= max(R === Float64 ? 1e-3 : 3e-3, 3 * err.j_rel_rms)

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
        @test e_replay <= max(R === Float64 ? 1e-3 : 3e-3, 3 * err_replay.j_rel_rms)

        # sfs=false evaluations skip delivery (rows untouched)
        S = copy(Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n])
        vpm_fmm.UJ_fmm(gpu)                 # default sfs=false
        @test Array(gpu.particles)[vpm_fmm.SFS_INDEX, 1:n] == S
    end
end
