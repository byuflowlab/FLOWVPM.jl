# Acceptance gate for the FastMultipole KA migration: a whole VPM `UJ` call.
#
# Both arms run over ONE `RadixFMMCache`. The cache is built by the existing
# CUDA path (`RadixFMMCache(device=true)`), the native arm runs
# `_cuda_lifecycle_body!` through it, and the KA arm runs `ka_lifecycle_body!`
# through the same resident state -- selected per call by the runtime radix
# setting `:RADIX_KA_LIFECYCLE`. Nothing is rebuilt between arms, so the two
# see byte-identical inputs and the comparison is of kernels only.
#
# THREE THINGS THIS DOES NOT COMPARE, and must be read alongside any number
# it prints:
#
#   1. M2L window GENERATION is native in both arms. `ka_hierarchical_m2l!`
#      reuses `_cuda_hier_generate_window!` (a CUDA-only flag/scan/compact)
#      and swaps only the per-window apply. The M2L row is an A/B of the
#      rotation/translation math, not of the route bookkeeping.
#   2. The M2L strategy is pinned to `:concat` on BOTH arms. The KA workspace
#      supports only the concatenated plan, and CUDA's fastest H200
#      configuration is the dense fused plan -- so the native column here is
#      NOT native-at-its-best. Run `m2l_strategy=:dense` natively for that
#      number; it is reported below as `native_dense` for context.
#   3. Nearfield shapes differ. FLOWVPM's default `:partitioned` kernel takes
#      CUDA's binned split stream, while the KA arm runs the generic functor
#      pair kernel over the same pair list. Same physics, different schedule
#      -- and the KA kernel drops `_cuda_fast_rsqrt`, so it is the more
#      accurate side (see ka_nearfield_correctness.jl).
#
# Accuracy is reported arm-vs-arm on delivered U and J (the migration
# criterion: does swapping kernels change the answer), plus, at the smallest
# n only, both arms against the O(N^2) `UJ_direct` reference so the shared
# FMM truncation error is visible for scale.
#
# Usage: julia --project=<env with CUDA, KernelAbstractions, FLOWVPM,
#                         FastMultipole@ka-migration> scripts/ka_uj_fmm_ab.jl
# Writes a CSV to $KA_UJ_AB_CSV (default ./ka_uj_fmm_ab_<jobid-or-pid>.csv).

import Random
using Printf
using CUDA
using KernelAbstractions
import FLOWVPM
const fmm = FLOWVPM.fmm

const KAEXT = Base.get_extension(fmm, :FastMultipoleKAExt)
KAEXT === nothing && error("FastMultipoleKAExt is not loaded")

const NS = parse.(Int, split(get(ENV, "KA_UJ_AB_NS", "100000,1000000"), ","))
const REPS = parse(Int, get(ENV, "KA_UJ_AB_REPS", "20"))
const CSV_PATH = get(ENV, "KA_UJ_AB_CSV",
    "ka_uj_fmm_ab_$(get(ENV, "SLURM_JOB_ID", string(getpid()))).csv")

#------- field construction -------#

# The radix/GPU path refuses autotuning (FLOWVPM_fmm_radix.jl:381), and the A/B
# is only meaningful if both arms share p, so both constructors take this.
const AB_FMM = FLOWVPM.FMM(; p=4, autotune_p=false, autotune_ncrit=false,
                           autotune_reg_error=false, default_rho_over_sigma=1.0)

function build_pfield(n, R; seed=1)
    Random.seed!(seed)
    pfield = FLOWVPM.ParticleField(n, R; fmm=AB_FMM)
    for _ in 1:n
        X = rand(R, 3) .* 2 .- 1
        Gamma = rand(R, 3) .* 2 .- 1
        sigma = R(0.1) + R(0.05) * rand(R)
        FLOWVPM.add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

function build_cuda_from(cpu_pfield)
    n = cpu_pfield.maxparticles
    R = eltype(cpu_pfield.particles)
    cfield = FLOWVPM.ParticleField(n, R; arraytype=CUDA.CuArray, np=cpu_pfield.np, fmm=AB_FMM)
    cfield.particles .= CUDA.CuArray(cpu_pfield.particles)
    return cfield
end

# Delivered velocity and velocity gradient, read through the module's own row
# constants so a packed-layout change breaks loudly instead of silently
# comparing the wrong rows.
uj_snapshot(pf) = Array(pf.particles[FLOWVPM.U_INDEX, 1:pf.np]),
                  Array(pf.particles[FLOWVPM.J_INDEX, 1:pf.np])

function relerr(a, b)
    d = maximum(abs.(a .- b))
    s = maximum(abs.(b))
    return s == 0 ? d : d / s
end

#------- timing -------#

# min-of-reps after a discarded warm-up: the KA arm compiles on first call and
# the native arm may record a graph, neither of which belongs in the number.
function timeit(f!, reps)
    f!(); CUDA.synchronize()
    t = Inf
    for _ in 1:reps
        dt = CUDA.@elapsed f!()
        t = min(t, Float64(dt))
    end
    return t * 1e3  # ms
end

arm!(on::Bool) = fmm.set_radix_setting!(:RADIX_KA_LIFECYCLE, on)

#------- per-stage breakdown -------#
#
# Called on the resident state left behind by a completed UJ, so every input
# buffer is already populated. Stages are timed in isolation and DO NOT sum to
# the lifecycle time (no nearfield/far-field stream overlap, and each is
# preceded by a synchronize). Any stage whose driver signature has drifted is
# reported as NaN rather than taking the job down.
function stage_times(state, reps)
    ws = state.scratch
    stages = Pair{String,Tuple{Function,Function}}[
        "nearfield" => (() -> fmm._launch_cuda_nearfield_kernel!(state),
                        () -> KAEXT.ka_launch_nearfield!(state)),
        "b2m"       => (() -> fmm._launch_cuda_b2m!(state),
                        () -> KAEXT.ka_launch_b2m!(state; workgroup=128)),
        "m2m"       => (() -> fmm._launch_cuda_resident_m2m!(state),
                        () -> for g in ws.m2m_groups
                                  KAEXT.ka_resident_stage_group_apply!(
                                      state.multipoles, state.multipoles, g, ws, :m2m)
                              end),
        "m2l"       => (() -> fmm._launch_cuda_resident_m2l!(state),
                        () -> KAEXT.ka_launch_m2l!(state, ws)),
        "l2l"       => (() -> fmm._launch_cuda_resident_l2l!(state),
                        () -> for g in ws.l2l_groups
                                  KAEXT.ka_resident_stage_group_apply!(
                                      state.locals, state.locals, g, ws, :l2l)
                              end),
        "l2b"       => (() -> fmm._launch_cuda_resident_l2b!(state),
                        () -> KAEXT.ka_launch_l2b!(state)),
    ]
    out = Tuple{String,Float64,Float64}[]
    for (name, (native, ka)) in stages
        tn = try timeit(() -> (native(); nothing), reps) catch err
            @warn "native stage $name failed" err; NaN end
        tk = try timeit(() -> (ka(); nothing), reps) catch err
            @warn "KA stage $name failed" err; NaN end
        push!(out, (name, tn, tk))
    end
    return out
end

#------- one problem size -------#

function run_case(io, n)
    @printf("\n===== n = %d =====\n", n)
    cpu = build_pfield(n, Float64)
    pf = build_cuda_from(cpu)
    # :concat on both arms -- the KA workspace supports no other plan.
    FLOWVPM.radix_fmm_settings!(pf; m2l_strategy=:concat)

    arm!(false)
    FLOWVPM.UJ_fmm_gpu!(pf)          # builds the cache; both arms reuse it
    U_n, J_n = uj_snapshot(pf)
    t_native = timeit(() -> FLOWVPM.UJ_fmm_gpu!(pf), REPS)

    arm!(true)
    FLOWVPM.UJ_fmm_gpu!(pf)
    U_k, J_k = uj_snapshot(pf)
    t_ka = timeit(() -> FLOWVPM.UJ_fmm_gpu!(pf), REPS)
    arm!(false)

    eU, eJ = relerr(U_k, U_n), relerr(J_k, J_n)
    @printf("UJ_fmm   native %8.3f ms   KA %8.3f ms   KA/native %.3fx\n",
            t_native, t_ka, t_ka / t_native)
    @printf("delivered relerr vs native:  U %.3e   J %.3e\n", eU, eJ)
    println(io, "uj,$n,$t_native,$t_ka,$(t_ka/t_native),$eU,$eJ")

    # native at its own best configuration, for context on the :concat pin
    pf2 = build_cuda_from(cpu)
    FLOWVPM.radix_fmm_settings!(pf2; m2l_strategy=:dense)
    FLOWVPM.UJ_fmm_gpu!(pf2)
    t_dense = timeit(() -> FLOWVPM.UJ_fmm_gpu!(pf2), REPS)
    @printf("UJ_fmm   native_dense %8.3f ms   (KA/native_dense %.3fx)\n",
            t_dense, t_ka / t_dense)
    println(io, "uj_native_dense,$n,$t_dense,,,,")

    st = FLOWVPM._radix_fmm_couplings[pf]
    for (name, tn, tk) in stage_times(st.cache.state, REPS)
        @printf("  %-10s native %8.3f ms   KA %8.3f ms   %.3fx\n",
                name, tn, tk, tk / tn)
        println(io, "stage_$name,$n,$tn,$tk,$(tk/tn),,")
    end

    if n == minimum(NS)
        ref = build_cuda_from(cpu)
        FLOWVPM.UJ_direct(ref)
        U_d, J_d = uj_snapshot(ref)
        @printf("vs O(N^2) direct:  native U %.3e J %.3e | KA U %.3e J %.3e\n",
                relerr(U_n, U_d), relerr(J_n, J_d),
                relerr(U_k, U_d), relerr(J_k, J_d))
        println(io, "direct,$n,,,,$(relerr(U_k, U_d)),$(relerr(J_k, J_d))")
    end
    return nothing
end

#------- main -------#

CUDA.functional() || error("no functional CUDA device")
println("device: ", CUDA.name(CUDA.device()))
println("reps:   ", REPS)
println("csv:    ", CSV_PATH)

open(CSV_PATH, "w") do io
    println(io, "what,n,native_ms,ka_ms,ka_over_native,relerr_U,relerr_J")
    for n in NS
        run_case(io, n)
    end
end
println("\nka_uj_fmm_ab complete")
