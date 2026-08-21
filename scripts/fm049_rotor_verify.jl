# fm049_rotor_verify.jl — task 049: rotor-field GPU verification on the real
# p018 wake snapshot (step 710, n = 210,056).
#
# CLUSTER driver (H200; written blind, parse-checked locally — no CUDA on the
# dev machine). Usage:
#
#   julia --project=<env> scripts/fm049_rotor_verify.jl <p018_710_particles.bin>
#
# Reads the raw snapshot dump produced by
# FastMultipole/MATRIX_OPERATOR_REFACTOR/scripts/fm049_extract_snapshot.jl
# (two Int64s: nrows=46, np; then 46 x np little-endian Float64, column-major),
# then runs:
#   REF  : full-field GPU direct O(N^2) U/J + Estr reference (the validated
#          048 direct-sum kernels), cross-checked on 2000 seeded sampled
#          targets against a CPU-threaded FastMultipole direct! evaluation.
#   ARM 1: device-resident radix UJ_fmm (no SFS) — parity + median-of-5 wall.
#   ARM 2: UJ_fmm with SFS — parity (mechanical expectation: SFS is exact in
#          the delivered J; delivered-accuracy is J-error-bound, task 048)
#          + marginal SFS cost.
#   ARM 3: full device-resident `nextstep` RK3 steps (exercises the task-049
#          U_prev broadcast fix) — median-of-5 step wall + UJ-vs-rest split
#          via a timed UJ wrapper (RK3 calls UJ three times per step).
#   ARM 4: residency A/B — upload-per-step emulation (H2D of the 46 x np
#          matrix + device UJ + D2H of the U/J/SFS result rows) vs the
#          device-resident numbers, plus the transfer-based host(CPU)-radix
#          path datum.
#
# Outputs (in the working directory the job runs from):
#   fm049_results.csv, fm049_report.txt

import FLOWVPM
import CUDA
using Random: MersenneTwister, randperm
using Statistics: median
using SHA: sha256
using Printf

const vpm = FLOWVPM
const ffmm = FLOWVPM.fmm

# ------------------------------------------------------------------ constants
const N_TARGETS = 2000          # sampled-direct cross-check targets
const SAMPLE_SEED = 49049
const N_REPS = 5                # timing repetitions (after warmup)
const N_STEPS = 5               # ARM 3 RK3 steps
const DT = 1e-5                 # small step: field state barely drifts
const U_GATE = 1e-3             # relative velocity RMS gate (F64)
const TARGET_STEP = 3.3         # s/step budget target (018 campaign)
const CPU_BASELINE = "170-230 s/step (018 CPU baseline, 64 cores)"
const ANCHOR_041A = "041a unitcube GPU best-uniform UJ: 7.40 ms @1e5, 92.3 ms @1e6"

# ------------------------------------------------------------------ load dump
length(ARGS) >= 1 || error("usage: fm049_rotor_verify.jl <p018_710_particles.bin>")
binpath = ARGS[1]
isfile(binpath) || error("snapshot dump not found: $binpath")
data, np = open(binpath, "r") do io
    nrows = read(io, Int64)
    np = read(io, Int64)
    nrows == 46 || error("unexpected row count $nrows != 46")
    A = Matrix{Float64}(undef, 46, np)
    read!(io, A)
    (A, Int(np))
end
digest = bytes2hex(open(sha256, binpath))
@info "snapshot loaded" binpath np sha256=digest
all(isfinite, data) || error("non-finite snapshot data")
all(>(0), view(data, vpm.SIGMA_INDEX, 1:np)) || error("non-positive sigma")

CUDA.functional() || error("CUDA not functional on this node")

# ---------------------------------------------------------------- field setup
fm049_settings() = vpm.FMM(; p=4, ncrit=50, theta=0.4,
    autotune_p=false, autotune_ncrit=false, autotune_reg_error=false)

function make_pfield(A::Matrix{Float64}; arraytype=Matrix, UJ=vpm.UJ_fmm)
    n = size(A, 2)
    pf = vpm.ParticleField(n, Float64;
        formulation=vpm.rVPM,
        kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(),
        SFS=vpm.noSFS,          # UJ arms drive sfs explicitly per call
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=UJ,
        fmm=fm049_settings(),
        arraytype=arraytype)
    copyto!(pf.particles, A)
    pf.np = n
    return pf
end

# relative RMS of selected rows against a reference matrix, over columns cols
function rel_rms(particles, ref_particles, rows, cols)
    A = Array(particles); B = Array(ref_particles)
    err2 = ref2 = 0.0
    for i in cols, r in rows
        d = Float64(A[r, i]) - Float64(B[r, i])
        err2 += d * d
        ref2 += Float64(B[r, i])^2
    end
    return sqrt(err2 / max(ref2, eps()))
end

# median wall time of f() (device-synchronized), after warmup calls
function med_time(f; warmup=1, reps=N_REPS)
    for _ in 1:warmup
        f(); CUDA.synchronize()
    end
    ts = Float64[]
    for _ in 1:reps
        CUDA.synchronize()
        push!(ts, @elapsed begin
            f(); CUDA.synchronize()
        end)
    end
    return median(ts), ts
end

results = Vector{Pair{String,Any}}()
rec!(k, v) = push!(results, k => v)
rec!("np", np)
rec!("sha256", digest)
rec!("julia_threads", Threads.nthreads())

# =========================================================================
# REFERENCE: full-field GPU direct U/J + Estr
# =========================================================================
@info "REF: full-field GPU direct O(N^2) (U/J + Estr)"
gpu_ref = make_pfield(data; arraytype=CUDA.CuArray, UJ=vpm.UJ_direct)
t_ref = @elapsed begin
    vpm.UJ_direct(gpu_ref; sfs=true)   # gpu_direct! + gpu_estr_direct!
    CUDA.synchronize()
end
rec!("t_ref_full_direct_s", t_ref)
@info "REF done" t_ref

# sampled CPU cross-check of the GPU direct reference (seeded)
rng = MersenneTwister(SAMPLE_SEED)
tidx = sort(randperm(rng, np)[1:N_TARGETS])
tgt = make_pfield(data[:, tidx]; arraytype=Matrix, UJ=vpm.UJ_direct)
tgt.particles[vpm.U_INDEX, :] .= 0
tgt.particles[vpm.J_INDEX, :] .= 0
src = make_pfield(data; arraytype=Matrix, UJ=vpm.UJ_direct)
t_cpu_sampled = @elapsed ffmm.direct!(tgt, src;
    scalar_potential=false, gradient=true, hessian=true)
ref_host = Array(gpu_ref.particles)
u_x = rel_rms(tgt.particles, ref_host[:, tidx], vpm.U_INDEX, 1:N_TARGETS)
j_x = rel_rms(tgt.particles, ref_host[:, tidx], vpm.J_INDEX, 1:N_TARGETS)
rec!("t_cpu_sampled_direct_s", t_cpu_sampled)
rec!("xcheck_u_rel_rms", u_x)
rec!("xcheck_j_rel_rms", j_x)
@info "sampled CPU vs GPU direct cross-check" u_x j_x
u_x <= 1e-10 || @warn "GPU-vs-CPU direct cross-check exceeds 1e-10 (F64)" u_x

# =========================================================================
# ARM 1: device-resident radix UJ (no SFS)
# =========================================================================
@info "ARM 1: device UJ_fmm (no SFS)"
gpu = make_pfield(data; arraytype=CUDA.CuArray, UJ=vpm.UJ_fmm)
t_build = @elapsed begin
    vpm.UJ_fmm(gpu)                    # first call builds the RadixFMMCache
    CUDA.synchronize()
end
rec!("t_cache_build_plus_first_uj_s", t_build)
u1s = rel_rms(gpu.particles[:, tidx], ref_host[:, tidx], vpm.U_INDEX, 1:N_TARGETS)
u1f = rel_rms(gpu.particles, ref_host, vpm.U_INDEX, 1:np)
j1f = rel_rms(gpu.particles, ref_host, vpm.J_INDEX, 1:np)
rec!("arm1_u_rel_rms_sampled", u1s)
rec!("arm1_u_rel_rms_full", u1f)
rec!("arm1_j_rel_rms_full", j1f)
arm1_pass = u1s <= U_GATE
rec!("arm1_gate_pass", arm1_pass)
t_uj, ts_uj = med_time(() -> vpm.UJ_fmm(gpu))
rec!("arm1_t_uj_median_s", t_uj)
rec!("arm1_t_uj_all_s", ts_uj)
st = vpm._radix_fmm_couplings[gpu]
rec!("arm1_body_uploads", st.cache.state.counters.body_uploads)
@info "ARM 1 done" u1s u1f j1f t_uj arm1_pass

# =========================================================================
# ARM 2: device UJ with SFS
# =========================================================================
@info "ARM 2: device UJ_fmm (SFS)"
vpm.UJ_fmm(gpu; sfs=true)
CUDA.synchronize()
s2f = rel_rms(gpu.particles, ref_host, vpm.SFS_INDEX, 1:np)
s2s = rel_rms(gpu.particles[:, tidx], ref_host[:, tidx], vpm.SFS_INDEX, 1:N_TARGETS)
rec!("arm2_sfs_rel_rms_full", s2f)
rec!("arm2_sfs_rel_rms_sampled", s2s)
t_ujsfs, _ = med_time(() -> vpm.UJ_fmm(gpu; sfs=true))
rec!("arm2_t_uj_sfs_median_s", t_ujsfs)
rec!("arm2_sfs_marginal_s", t_ujsfs - t_uj)
@info "ARM 2 done (SFS delivered accuracy is J-error-bound; diagnostic only)" s2f t_ujsfs

# =========================================================================
# ARM 3: full device-resident nextstep (RK3, U_prev fix active)
# =========================================================================
@info "ARM 3: device-resident nextstep RK3 x $N_STEPS"
const UJ_STAGE_TIMES = Float64[]
function timed_UJ(pf; optargs...)
    CUDA.synchronize()
    t = @elapsed begin
        vpm.UJ_fmm(pf; optargs...)
        CUDA.synchronize()
    end
    push!(UJ_STAGE_TIMES, t)
    return nothing
end
gpu3 = make_pfield(data; arraytype=CUDA.CuArray, UJ=timed_UJ)
vpm.nextstep(gpu3, DT)                 # warmup: cache build + JIT
CUDA.synchronize()
empty!(UJ_STAGE_TIMES)
step_ts = Float64[]
for _ in 1:N_STEPS
    CUDA.synchronize()
    push!(step_ts, @elapsed begin
        vpm.nextstep(gpu3, DT)         # update_U_prev=true: broadcast fix
        CUDA.synchronize()
    end)
end
t_step = median(step_ts)
n_uj_calls = length(UJ_STAGE_TIMES)
uj_per_step = sum(UJ_STAGE_TIMES) / N_STEPS
rec!("arm3_t_step_median_s", t_step)
rec!("arm3_t_step_all_s", step_ts)
rec!("arm3_uj_calls_per_step", n_uj_calls / N_STEPS)
rec!("arm3_t_uj_per_step_s", uj_per_step)
rec!("arm3_t_rest_per_step_s", t_step - uj_per_step)

# U_prev fix check: row 44 must equal |U| of the final U rows
A3 = Array(gpu3.particles)
uprev_err = maximum(abs.(A3[vpm.U_PREV_INDEX, 1:np] .-
    vec(sqrt.(sum(abs2, A3[vpm.U_INDEX, 1:np]; dims=1)))))
rec!("arm3_uprev_max_abs_err", uprev_err)
uprev_ok = all(isfinite, A3[vpm.U_PREV_INDEX, 1:np]) && uprev_err <= 1e-12
rec!("arm3_uprev_fix_ok", uprev_ok)
@info "ARM 3 done" t_step uj_per_step uprev_err uprev_ok

# =========================================================================
# ARM 4: residency A/B
# =========================================================================
@info "ARM 4: residency A/B (upload-per-step emulation)"
host_state = Array(gpu.particles)                 # 46 x np host matrix
dev_result = CUDA.zeros(Float64, 46, np)          # for the D2H measurement
out_rows = vcat(collect(vpm.U_INDEX), collect(vpm.J_INDEX), collect(vpm.SFS_INDEX))
host_out = Matrix{Float64}(undef, length(out_rows), np)

t_h2d, _ = med_time(() -> copyto!(gpu.particles, host_state))
t_d2h, _ = med_time(() -> copyto!(host_out,
    Array(view(gpu.particles, out_rows, 1:np))))  # download + host gather
t_d2h_full, _ = med_time(() -> copyto!(host_state, gpu.particles))
rec!("arm4_t_h2d_full46_s", t_h2d)
rec!("arm4_t_d2h_ujsfs_rows_s", t_d2h)
rec!("arm4_t_d2h_full46_s", t_d2h_full)
t_upload_step = t_h2d + t_uj + t_d2h
rec!("arm4_t_upload_per_uj_emulated_s", t_upload_step)
transfer_frac_uj = (t_h2d + t_d2h) / t_uj
transfer_frac_step = 3 * (t_h2d + t_d2h) / t_step  # RK3: 3 UJ calls/step
rec!("arm4_transfer_frac_of_uj", transfer_frac_uj)
rec!("arm4_transfer_frac_of_step", transfer_frac_step)

# transfer-based host(CPU)-radix path datum (not GPU; contextual)
t_host_radix = NaN
try
    cpuh = make_pfield(data; arraytype=Matrix, UJ=vpm.UJ_fmm)
    vpm.UJ_fmm_gpu!(cpuh)                          # builds the host cache
    global t_host_radix, _ = med_time(() -> vpm.UJ_fmm_gpu!(cpuh);
        warmup=0, reps=3)
catch err
    @warn "host-radix path datum failed" err
end
rec!("arm4_t_host_radix_uj_s", t_host_radix)
@info "ARM 4 done" t_h2d t_d2h transfer_frac_step t_host_radix

# =========================================================================
# outputs
# =========================================================================
open("fm049_results.csv", "w") do io
    println(io, "key,value")
    for (k, v) in results
        println(io, "$k,\"$(v)\"")
    end
end

recommend_resident = transfer_frac_step > 0.15
open("fm049_report.txt", "w") do io
    println(io, "fm049 rotor field GPU verification — p018 step 710, np=$np")
    println(io, "snapshot sha256: $digest")
    println(io, "")
    println(io, "ACCURACY (reference: full-field GPU direct, CPU cross-check ",
        @sprintf("u=%.2e j=%.2e", u_x, j_x), ")")
    println(io, @sprintf("  ARM1 U rel RMS (2000 sampled) : %.3e  gate %.0e -> %s",
        u1s, U_GATE, arm1_pass ? "PASS" : "FAIL"))
    println(io, @sprintf("  ARM1 U rel RMS (full field)   : %.3e", u1f))
    println(io, @sprintf("  ARM1 J rel RMS (full, diag)   : %.3e", j1f))
    println(io, @sprintf("  ARM2 SFS rel RMS (full, diag) : %.3e  (J-error-bound, 048)", s2f))
    println(io, @sprintf("  ARM3 U_prev row check         : max err %.2e -> %s",
        uprev_err, uprev_ok ? "OK" : "FAIL"))
    println(io, "")
    println(io, "PER-PASS BUDGET (median wall, s)")
    println(io, @sprintf("  cache build + first UJ        : %.4f", t_build))
    println(io, @sprintf("  device UJ (no SFS)            : %.4f", t_uj))
    println(io, @sprintf("  device UJ (SFS)               : %.4f  (marginal %.4f)",
        t_ujsfs, t_ujsfs - t_uj))
    println(io, @sprintf("  full RK3 nextstep             : %.4f  (UJ %.4f + rest %.4f)",
        t_step, uj_per_step, t_step - uj_per_step))
    println(io, @sprintf("  H2D 46 x %d matrix            : %.4f", np, t_h2d))
    println(io, @sprintf("  D2H U/J/SFS rows              : %.4f", t_d2h))
    println(io, @sprintf("  host(CPU)-radix UJ datum      : %.4f", t_host_radix))
    println(io, "")
    println(io, "BUDGET vs TARGET")
    println(io, @sprintf("  step budget target            : %.2f s/step (CPU baseline %s)",
        TARGET_STEP, CPU_BASELINE))
    println(io, @sprintf("  measured device step          : %.4f s/step -> %.1f%% of budget",
        t_step, 100 * t_step / TARGET_STEP))
    println(io, "  anchors: $ANCHOR_041A")
    println(io, @sprintf("  (this field: n = %.2e, between the two anchors)", Float64(np)))
    println(io, "")
    println(io, "RESIDENCY (15%-of-step-time rule)")
    println(io, @sprintf("  per-step transfer cost (3x RK3): %.4f s = %.1f%% of device step",
        3 * (t_h2d + t_d2h), 100 * transfer_frac_step))
    println(io, "  recommendation: ", recommend_resident ?
        "DEVICE-RESIDENT (transfers exceed 15% of step time)" :
        "either (transfers under 15% of step time)")
end
println(read("fm049_report.txt", String))
println("fm049 driver complete")
