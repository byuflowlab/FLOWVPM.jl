# Task-048 accuracy/efficiency tuning sweep (2026-08-22, user-directed).
#
# Phase 1 (cube n=2e4, F64): grid over P x rho_t x near-shell q, measuring
# delivered-E accuracy against the exact-erf direct reference and warmed
# per-step cost. Phase 2: the Pareto-frontier configs (plus the production
# baselines) run on the real p018 210k-particle snapshot, where the strict
# delivered gate (5e-4 F64) is applied — per user decision the gate binds on
# the PRODUCTION field; the cube grid exists to (a) find the frontier cheaply
# and (b) report the cube-vs-production discrepancy explicitly.
#
# Accuracy is measured twice per config: on the FIRST lifecycle call
# (uncaptured launch-sequence body) and again after warmup (CUDA-graph
# replay). Any drift between the two is an independent replay-parity signal
# that cannot be masked by tolerance scaling. Device-only; cuda_048_run.sh
# captures stdout and hashes artifacts.
using CUDA, Random, Statistics
import FLOWVPM
const vpm = FLOWVPM

CUDA.functional() || error("CUDA is not functional")
const N = parse(Int, get(ENV, "FM048_SWEEP_N", "20000"))
const REPS = parse(Int, get(ENV, "FM048_SWEEP_REPS", "9"))
const SEED = parse(Int, get(ENV, "FM048_SWEEP_SEED", "48048"))
const OUT = get(ENV, "FM048_SWEEP_CSV", "fm048_sweep_results.csv")
const P018_BIN = get(ENV, "FM048_P018_BIN", "")
const STRICT_GATE_F64 = 5e-4
const MAX_P018_ARMS = parse(Int, get(ENV, "FM048_SWEEP_MAX_P018", "8"))

const PS = (4, 6, 8)
const RHOS = (4.211, 4.789)
# near-shell candidates: 0 = derived default (do not pass near_radius2).
# Supported rigid q values cap at 20 (_SUPPORTED_RIGID_NEAR_RADII2); 20 is
# the host-matrix operating point (min rho ~ 5.2 there).
const QS = (0, 14, 17, 20)

function load_snapshot(path)
    isfile(path) || error("FM048_P018_BIN snapshot not found: $path")
    open(path, "r") do io
        nrows = read(io, Int64); n = read(io, Int64)
        nrows == 46 || error("unexpected snapshot row count $nrows")
        A = Matrix{Float64}(undef, nrows, n); read!(io, A)
        # The production snapshot carries LIVE SFS rows (p018_710: every
        # column nonzero, |max| ~ 3.6e5). Estr_direct! ACCUMULATES, so a
        # nonzero starting SFS would contaminate the exact reference while
        # the test arms reset (reset_sfs=true) — making every p018 arm's
        # delivered-E measurement O(1) garbage. Zero at load; the snapshot
        # has no statics, so this equals _reset_particles_sfs.
        A[vpm.SFS_INDEX, :] .= 0.0
        return A
    end
end

function build_field(::Type{R}, P, rho_t, q; UJ=vpm.UJ_fmm, snapshot=nothing) where R
    n = snapshot === nothing ? N : size(snapshot, 2)
    rng = MersenneTwister(SEED)
    sigma = R(2 * n^(-1 / 3))
    pf = vpm.ParticleField(n, R; formulation=vpm.rVPM,
        kernel=vpm.gaussianerf, viscous=vpm.Inviscid(), SFS=vpm.noSFS,
        UJ, arraytype=CuArray,
        fmm=vpm.FMM(; p=P, ncrit=50, theta=0.4, autotune_p=false,
            autotune_ncrit=false, autotune_reg_error=false))
    if snapshot === nothing
        for _ in 1:n
            vpm.add_particle(pf, rand(rng, R, 3),
                (R(2) .* rand(rng, R, 3) .- one(R)) ./ R(n), sigma)
        end
    else
        pf.np = n
        pf.particles .= CUDA.CuArray(R.(snapshot))
    end
    if UJ === vpm.UJ_fmm
        if q == 0
            vpm.radix_fmm_settings!(pf; expansion_order=P, rho_t)
        else
            vpm.radix_fmm_settings!(pf; expansion_order=P, rho_t, near_radius2=q)
        end
    end
    return pf
end

gpu_seconds(f) = (CUDA.synchronize(); t0 = time_ns(); f(); CUDA.synchronize();
                  (time_ns() - t0) / 1e9)

relrms(A, B, idx, n) = begin
    a = Float64.(Array(A)[idx, 1:n]); b = Float64.(Array(B)[idx, 1:n])
    sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))
end

# Exact-erf reference (direct-sum U/J + direct pairwise Estr), one per
# (case, precision) — independent of P/rho_t/q.
function build_reference(::Type{R}; snapshot=nothing) where R
    ref = build_field(R, 4, RHOS[1], 0; UJ=vpm.UJ_direct, snapshot)
    vpm.UJ_direct(ref)
    vpm.Estr_direct!(ref)
    return ref
end

function release!(pf)
    vpm.clear_radix_fmm_cache!(pf)
    return nothing
end

struct SweepRow
    case::String; P::Int; R::String; rho_t::Float64; q::Int
    ell::Int; n::Int; n_direct::Int
    e_u::Float64; e_j::Float64; e_sfs_first::Float64; e_sfs_warm::Float64
    t_uj::Float64; t_ujsfs::Float64
end

function run_config(case, ::Type{R}, P, rho_t, q; ref, snapshot=nothing) where R
    pf = build_field(R, P, rho_t, q; snapshot)
    n = pf.np
    # first evaluation: uncaptured launch-sequence body
    vpm.UJ_fmm(pf; sfs=true, reset=true, reset_sfs=true)
    e_u = relrms(pf.particles, ref.particles, vpm.U_INDEX, n)
    e_j = relrms(pf.particles, ref.particles, vpm.J_INDEX, n)
    e_sfs_first = relrms(pf.particles, ref.particles, vpm.SFS_INDEX, n)
    # warm: JIT/handles, graph record, replay for both call shapes
    for _ in 1:3
        vpm.UJ_fmm(pf; sfs=false)
        vpm.UJ_fmm(pf; sfs=true, reset_sfs=true)
    end
    # replayed-path accuracy: independent replay-parity signal vs e_sfs_first
    vpm.UJ_fmm(pf; sfs=true, reset=true, reset_sfs=true)
    e_sfs_warm = relrms(pf.particles, ref.particles, vpm.SFS_INDEX, n)
    e_j_warm = relrms(pf.particles, ref.particles, vpm.J_INDEX, n)
    tuj = Float64[]; tsfs = Float64[]
    for rep in 1:REPS
        if isodd(rep)
            push!(tuj, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=false)))
            push!(tsfs, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=true, reset_sfs=true)))
        else
            push!(tsfs, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=true, reset_sfs=true)))
            push!(tuj, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=false)))
        end
    end
    coupling = vpm._radix_fmm_couplings[pf]
    st = coupling.cache.state
    row = SweepRow(case, P, string(R), rho_t, q, coupling.cache.ell, n,
        Int(st.counts.n_direct), e_u, e_j, e_sfs_first, e_sfs_warm,
        median(tuj), median(tsfs))
    @info "fm048 sweep config" case P R rho_t q ell=row.ell n row.n_direct e_u e_j e_j_warm e_sfs_first e_sfs_warm t_uj=row.t_uj t_ujsfs=row.t_ujsfs replay_drift=abs(e_sfs_warm - e_sfs_first)
    release!(pf)
    GC.gc(); CUDA.reclaim()
    return row
end

csvline(r::SweepRow) = join((r.case, r.P, r.R, r.rho_t,
    r.q == 0 ? "derived" : string(r.q), r.ell, r.n, r.n_direct,
    r.e_u, r.e_j, r.e_sfs_first, r.e_sfs_warm, r.t_uj, r.t_ujsfs,
    r.t_ujsfs - r.t_uj), ',')

rows = SweepRow[]

# ---------------- phase 1: cube grid (F64) ----------------
@info "fm048 sweep phase 1: cube grid" N REPS SEED PS RHOS QS
ref64 = build_reference(Float64)
for P in PS, rho_t in RHOS, q in QS
    push!(rows, run_config("cube", Float64, P, rho_t, q; ref=ref64))
end

# ---------------- frontier selection ----------------
# Non-dominated set in (t_ujsfs, e_sfs_warm) over the cube F64 grid; the
# production baselines (P=4, derived q, both rho candidates) are always
# retained for the cube-vs-p018 discrepancy report.
cube = [r for r in rows if r.case == "cube"]
isdominated(r, rs) = any(o -> o !== r && o.t_ujsfs <= r.t_ujsfs &&
    o.e_sfs_warm <= r.e_sfs_warm &&
    (o.t_ujsfs < r.t_ujsfs || o.e_sfs_warm < r.e_sfs_warm), rs)
frontier = [r for r in cube if !isdominated(r, cube)]
baselines = [r for r in cube if r.P == 4 && r.q == 0]
selected = unique(vcat(frontier, baselines))
sort!(selected; by=r -> r.t_ujsfs)
if length(selected) > MAX_P018_ARMS
    # keep the baselines plus the most-accurate frontier configs
    keep = unique(vcat(baselines,
        sort([r for r in selected if !(r in baselines)]; by=r -> r.e_sfs_warm)))
    selected = keep[1:min(MAX_P018_ARMS, length(keep))]
end
@info "fm048 sweep frontier" n_frontier=length(frontier) n_selected=length(selected) configs=[(P=r.P, rho_t=r.rho_t, q=r.q) for r in selected]

# F32 spot checks on the frontier configs (cube only)
ref32 = build_reference(Float32)
for r in selected
    push!(rows, run_config("cube", Float32, r.P, r.rho_t, r.q; ref=ref32))
end
ref32 = nothing; GC.gc(); CUDA.reclaim()

# ---------------- phase 2: p018 production arms (F64, gated) ----------------
gate_report = String[]
if !isempty(P018_BIN)
    snap = load_snapshot(P018_BIN)
    @info "fm048 sweep phase 2: p018 arms" n=size(snap, 2) n_arms=length(selected)
    refp = build_reference(Float64; snapshot=snap)
    for r in selected
        row = run_config("p018", Float64, r.P, r.rho_t, r.q; ref=refp, snapshot=snap)
        push!(rows, row)
        pass = row.e_sfs_warm <= STRICT_GATE_F64 && row.e_sfs_first <= STRICT_GATE_F64
        cube_e = r.e_sfs_warm
        push!(gate_report, "p018 P=$(row.P) rho_t=$(row.rho_t) q=$(row.q == 0 ? "derived" : row.q): " *
            "e_sfs=$(row.e_sfs_warm) $(pass ? "PASS" : "FAIL") (gate $STRICT_GATE_F64); " *
            "cube e_sfs=$cube_e, p018/cube ratio=$(row.e_sfs_warm / cube_e), " *
            "t_ujsfs=$(row.t_ujsfs)s (cube $(r.t_ujsfs)s)")
    end
else
    @warn "FM048_P018_BIN unset: skipping the gated p018 production arms"
end

open(OUT, "w") do io
    println(io, "case,p,precision,rho_t,q,ell,n,n_direct_pairs,e_u,e_j," *
        "e_sfs_first,e_sfs_warm,uj_median_s,ujsfs_median_s,sfs_marginal_s")
    foreach(r -> println(io, csvline(r)), rows)
end
println("=== fm048 sweep gate report (strict delivered gate on the production field) ===")
foreach(println, gate_report)
println("FM048_SWEEP_CSV=$OUT")
