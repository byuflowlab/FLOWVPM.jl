# fm051_rect_bench.jl — task 051 stage 1: rectangular (distinct sources ->
# distinct targets) GPU direct-evaluation benchmark on the real p018 wake
# snapshot (step 710, n = 210,056) + a synthetic 36,752-panel rotor-disk set.
#
# CLUSTER driver (H200; written blind, parse-checked locally — no CUDA on the
# dev machine). Usage:
#
#   julia --project=<env> scripts/fm051_rect_bench.jl <p018_710_particles.bin>
#
# Snapshot format (fm049_extract_snapshot.jl): two Int64s (nrows=46, np) then
# 46 x np little-endian Float64 column-major; rows X=1:3, GAMMA=4:6, SIGMA=7.
#
# Passes (the FLOWPanel 018 cross passes, FLOWPanel_simulate.jl:673-712):
#   PASS 1 "points -> targets": gaussianerf vortex particles (np ~ 2.1e5)
#     inducing U (and optionally J) at the 36,752 panel centers.
#     Estimate: 0.02-0.04 s (041k H200 pair rates).
#   PASS 2 "panels -> targets": 36,752 source+vortex-ring tri panels inducing
#     U (and optionally J) at the 210k particle positions, core_size 1e-3
#     (CORE_SIZE_TARGETS). Estimate: 0.4-2 s.
# Each device pass is parity-checked on 2000 sampled targets against the host
# (threaded) FastMultipole.direct_rectangular! reference and timed
# (median of 5). Outputs: fm051_results.csv + report lines.

import FLOWVPM
import CUDA
using Random: MersenneTwister, randperm
using Statistics: median
using Printf

const vpm = FLOWVPM
const fmm = FLOWVPM.fmm

const N_SAMPLE = 2000
const SAMPLE_SEED = 51051
const N_REPS = 5
const PARITY_GATE = 1e-11          # F64 host/device summation-order roundoff
const EST_PASS1 = (0.02, 0.04)
const EST_PASS2 = (0.4, 2.0)
const CORE_SIZE_TARGETS = 1e-3  # 018 driver body->targets offset

# ------------------------------------------------------------------ load dump
length(ARGS) >= 1 || error("usage: fm051_rect_bench.jl <p018_710_particles.bin>")
binpath = ARGS[1]
isfile(binpath) || error("snapshot dump not found: $binpath")
data, np = open(binpath, "r") do io
    nrows = read(io, Int64)
    npv = read(io, Int64)
    nrows == 46 || error("unexpected row count $nrows != 46")
    A = Matrix{Float64}(undef, 46, npv)
    read!(io, A)
    (A, Int(npv))
end
@info "snapshot loaded" binpath np
all(isfinite, data) || error("non-finite snapshot data")
all(>(0), view(data, 7, 1:np)) || error("non-positive sigma")

CUDA.functional() || error("CUDA not functional on this node")
fmm.load_cuda_radix_lifecycle!() || error(
    "CUDA radix lifecycle failed to load: $(fmm.cuda_radix_status()) — " *
    "the rectangular CuMatrix methods ride this include")

# ------------------------------------------------- synthetic rotor-disk panels
# 36,752 tri panels on an annular disk r in [0.012, 0.12] m (DJI9443 R=0.12;
# ~mm panel scale matches the 018 40x40 body mesh areas). Each panel is a
# tag-4 (ConstantSource + VortexRing) element with random strengths, the 018
# body element set (rotor_hover_pressure_comparison.jl:281).
const N_PANELS = 36_752
function build_panels(rng)
    r0, r1 = 0.012, 0.12
    na, nr = 302, 61                       # 2*na*nr = 36,844 >= N_PANELS
    src = zeros(17, N_PANELS)
    centers = zeros(3, N_PANELS)
    q = 0
    for ir in 1:nr, ia in 1:na
        q >= N_PANELS && break
        ra = r0 + (r1 - r0) * (ir - 1) / nr
        rb = r0 + (r1 - r0) * ir / nr
        ta = 2pi * (ia - 1) / na
        tb = 2pi * ia / na
        p1 = (ra*cos(ta), ra*sin(ta), 0.0)
        p2 = (rb*cos(ta), rb*sin(ta), 0.0)
        p3 = (rb*cos(tb), rb*sin(tb), 0.0)
        p4 = (ra*cos(tb), ra*sin(tb), 0.0)
        for verts in ((p1, p2, p3), (p1, p3, p4))
            q >= N_PANELS && break
            q += 1
            src[1, q] = 4                  # ConstantSource + VortexRing
            src[2, q] = 3
            for (iv, v) in enumerate(verts)
                src[3 + 3*(iv-1), q] = v[1]
                src[4 + 3*(iv-1), q] = v[2]
                src[5 + 3*(iv-1), q] = v[3]
            end
            src[15, q] = 1e-3 * randn(rng)     # sigma
            src[16, q] = 1e-3 * randn(rng)     # Gamma
            src[17, q] = CORE_SIZE_TARGETS
            centers[1, q] = (verts[1][1] + verts[2][1] + verts[3][1]) / 3
            centers[2, q] = (verts[1][2] + verts[2][2] + verts[3][2]) / 3
            centers[3, q] = (verts[1][3] + verts[2][3] + verts[3][3]) / 3
        end
    end
    q == N_PANELS || error("panel synthesis bug: q=$q")
    # place the disk near the particle cloud so distances are representative
    com = sum(view(data, 1:3, 1:np); dims=2) ./ np
    for k in 1:3
        src[2 + k, :] .+= com[k]; src[5 + k, :] .+= com[k]; src[8 + k, :] .+= com[k]
        centers[k, :] .+= com[k]
    end
    return src, centers
end

rng = MersenneTwister(SAMPLE_SEED)
panel_src, panel_centers = build_panels(rng)
point_src = data[1:7, 1:np]                    # X 1:3, GAMMA 4:6, SIGMA 7
particle_pos = data[1:3, 1:np]

# ------------------------------------------------------------------- helpers
rel_rms(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))

# worst per-target relative error: a single branch-flipped target (the panel
# functor's absolute thresholds can flip under device FMA contraction) must
# show up rather than being diluted by the global RMS
function per_target_max(a, b)
    m = 0.0
    i_worst = 0
    for i in axes(b, 2)
        nb = sqrt(sum(abs2, view(b, :, i)))
        e = sqrt(sum(abs2, view(a, :, i) .- view(b, :, i))) / max(nb, eps())
        if e > m
            m = e; i_worst = i
        end
    end
    return m, i_worst
end

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
rec!("n_panels", N_PANELS)
rec!("julia_threads", Threads.nthreads())

tidx_p1 = sort(randperm(rng, N_PANELS)[1:N_SAMPLE])   # sampled panel centers
tidx_p2 = sort(randperm(rng, np)[1:N_SAMPLE])         # sampled particles

# one struct per pass config
function run_pass!(label, kernel, sources_h, targets_h, tidx, grad::Bool;
        precision=Float64, u_gate=PARITY_GATE, j_gate=PARITY_GATE)
    T = precision
    n_tgt = size(targets_h, 2)
    nrows = grad ? 12 : 3
    d_src = CUDA.CuMatrix{T}(sources_h)
    d_tgt = CUDA.CuMatrix{T}(targets_h)
    d_out = CUDA.zeros(T, nrows, n_tgt)
    # correctness pass
    fmm.direct_rectangular!(d_out, d_tgt, kernel, d_src; gradient=grad)
    CUDA.synchronize()
    out_h = Array(d_out)
    all(isfinite, out_h) || error("$label: non-finite device output")
    # host reference on the sampled targets (threaded)
    ref = zeros(Float64, nrows, length(tidx))
    fmm.direct_rectangular!(ref, targets_h[:, tidx], kernel,
        Float64.(sources_h); gradient=grad)
    u_err = rel_rms(Float64.(out_h[1:3, tidx]), ref[1:3, :])
    j_err = grad ? rel_rms(Float64.(out_h[4:12, tidx]), ref[4:12, :]) : NaN
    u_max, u_iw = per_target_max(Float64.(out_h[1:3, tidx]), ref[1:3, :])
    j_max, j_iw = grad ? per_target_max(Float64.(out_h[4:12, tidx]), ref[4:12, :]) : (NaN, 0)
    # timing (accumulating into the same out buffer; cost identical)
    t_med, ts = med_time(() -> fmm.direct_rectangular!(d_out, d_tgt, kernel,
        d_src; gradient=grad))
    pass = T === Float64 ? (u_err <= u_gate && u_max <= 100 * u_gate &&
                            (isnan(j_err) || (j_err <= j_gate && j_max <= 100 * j_gate))) :
                           u_err <= 1e-3
    rec!("$(label)_u_rel_rms", u_err)
    rec!("$(label)_j_rel_rms", j_err)
    rec!("$(label)_u_max_pertarget", u_max)
    rec!("$(label)_u_worst_sample_idx", u_iw)
    rec!("$(label)_j_max_pertarget", j_max)
    rec!("$(label)_j_worst_sample_idx", j_iw)
    rec!("$(label)_t_median_s", t_med)
    rec!("$(label)_t_all_s", ts)
    rec!("$(label)_parity_pass", pass)
    @info label u_err j_err t_med pass
    d_src = d_tgt = d_out = nothing
    CUDA.reclaim()
    return t_med, u_err, j_err, pass
end

pt = fmm.RectangularGaussianErfVortex()
pn = fmm.RectangularPanelInfluence()

# ================================================================== PASS 1
@info "PASS 1: particles -> $(N_PANELS) panel centers"
t1u, = run_pass!("pass1_f64_u", pt, point_src, panel_centers, tidx_p1, false)
t1uj, = run_pass!("pass1_f64_uj", pt, point_src, panel_centers, tidx_p1, true)
t1u32, = run_pass!("pass1_f32_u", pt, point_src, panel_centers, tidx_p1, false;
    precision=Float32)
t1uj32, = run_pass!("pass1_f32_uj", pt, point_src, panel_centers, tidx_p1, true;
    precision=Float32)

# ================================================================== PASS 2
@info "PASS 2: $(N_PANELS) panels -> $(np) particle positions"
# panels: J picks up an extra cancellation decade (atan2/log edge terms), and
# real particle targets can sit near blade panels — 049-audit-realistic gates
t2u, = run_pass!("pass2_f64_u", pn, panel_src, particle_pos, tidx_p2, false)
t2uj, = run_pass!("pass2_f64_uj", pn, panel_src, particle_pos, tidx_p2, true;
    j_gate=1e-10)

# ================================================================== PROFILE
# FM051_BENCH_PROFILE=1: timing-only decomposition of where the pass-1/2 cost
# lives (job 13306457 landed slow of both estimate bands). Two questions:
#   (a) pass-2 tag split — the p018 element is tag 4 (ConstantSource +
#       VortexRing); time tag-1-only (3 edge source integrals: atan/log) vs
#       tag-3-only (3 bound-vortex ring segments) copies of the same panels
#       to see which component dominates and what an element-split or
#       ring-specialized kernel could recover.
#   (b) pass-1 occupancy — the points kernel is one-thread-per-target and
#       pass 1 has only 36,752 targets (~144 blocks of 256 on a 132-SM H200,
#       each thread serially walking all 210k sources). If duplicating the
#       target set 2x/8x costs far less than 2x/8x wall time, the kernel is
#       latency/occupancy-bound there and a split-source variant (atomics or
#       per-block partial reduction) is the optimization lever.
if get(ENV, "FM051_BENCH_PROFILE", "0") in ("1", "true")
    @info "PROFILE: pass-2 tag split + pass-1 occupancy probe"
    function time_only!(label, kernel, sources_h, targets_h, grad::Bool)
        d_src = CUDA.CuMatrix{Float64}(sources_h)
        d_tgt = CUDA.CuMatrix{Float64}(targets_h)
        d_out = CUDA.zeros(Float64, grad ? 12 : 3, size(targets_h, 2))
        t_med, = med_time(() -> fmm.direct_rectangular!(d_out, d_tgt, kernel,
            d_src; gradient=grad))
        rec!("profile_$(label)_t_median_s", t_med)
        @info "profile_$label" t_med
        d_src = d_tgt = d_out = nothing
        CUDA.reclaim()
        return t_med
    end
    # (a) tag split at the exact pass-2 shape
    src_srconly = copy(panel_src); src_srconly[1, :] .= 1     # ConstantSource only
    src_ringonly = copy(panel_src); src_ringonly[1, :] .= 3   # VortexRing only (s1=Gamma)
    tp2_src_u  = time_only!("pass2_tag1_source_u",  pn, src_srconly,  particle_pos, false)
    tp2_ring_u = time_only!("pass2_tag3_ring_u",    pn, src_ringonly, particle_pos, false)
    tp2_src_uj  = time_only!("pass2_tag1_source_uj", pn, src_srconly,  particle_pos, true)
    tp2_ring_uj = time_only!("pass2_tag3_ring_uj",   pn, src_ringonly, particle_pos, true)
    # (b) pass-1 occupancy: same source cloud, target set tiled 2x and 8x
    tgt2 = hcat(panel_centers, panel_centers)
    tgt8 = repeat(panel_centers, 1, 8)
    tp1_x1 = t1u
    tp1_x2 = time_only!("pass1_targets_x2_u", pt, point_src, tgt2, false)
    tp1_x8 = time_only!("pass1_targets_x8_u", pt, point_src, tgt8, false)
    println("-" ^ 72)
    println("PROFILE summary")
    @printf("  pass2 tag split (U):   source-only %.4f s | ring-only %.4f s | combined %.4f s\n",
        tp2_src_u, tp2_ring_u, t2u)
    @printf("  pass2 tag split (U+J): source-only %.4f s | ring-only %.4f s | combined %.4f s\n",
        tp2_src_uj, tp2_ring_uj, t2uj)
    @printf("  pass1 occupancy (U): x1 %.4f s | x2 %.4f s (%.2fx) | x8 %.4f s (%.2fx)\n",
        tp1_x1, tp1_x2, tp1_x2 / tp1_x1, tp1_x8, tp1_x8 / tp1_x1)
    println("  (x8 scaling << 8x  =>  pass 1 is occupancy-bound at 36,752 targets;")
    println("   near-linear scaling  =>  it is genuinely arithmetic-bound)")
end

# ================================================================== outputs
open("fm051_results.csv", "w") do io
    println(io, "key,value")
    for (k, v) in results
        println(io, "$k,\"$(v)\"")
    end
end

inband(t, est) = est[1] <= t <= est[2] ? "IN band" : (t < est[1] ? "FASTER than band" : "SLOWER than band")
println("=" ^ 72)
println("fm051 rectangular direct benchmark — p018 step 710, np=$np, panels=$N_PANELS")
@printf("PASS 1 U-only  F64: %.4f s  (estimate %.2f-%.2f s -> %s)\n",
    t1u, EST_PASS1[1], EST_PASS1[2], inband(t1u, EST_PASS1))
@printf("PASS 1 U+J     F64: %.4f s\n", t1uj)
@printf("PASS 1 U-only  F32: %.4f s   U+J F32: %.4f s\n", t1u32, t1uj32)
@printf("PASS 2 U-only  F64: %.4f s  (estimate %.1f-%.1f s -> %s)\n",
    t2u, EST_PASS2[1], EST_PASS2[2], inband(t2u, EST_PASS2))
@printf("PASS 2 U+J     F64: %.4f s\n", t2uj)
println("results written to fm051_results.csv")
println("fm051 driver complete")
