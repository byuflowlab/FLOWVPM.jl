# Task-048 graph-replay J-defect diagnostic (2026-08-22). Background: on H200
# job 13298230 the FIRST (uncaptured) radix lifecycle evaluation delivers
# j_rel_rms ~ 2e-3 vs the direct reference, but graph-REPLAYED evaluations
# deliver j_rel_rms ~ 0.1098 (cube) — equal to sqrt(2)*(far-field share of J),
# i.e. the replayed far-field J contribution is full-magnitude but
# decorrelated. This script localizes the failing stage with the four checks
# specified by the 2026-08-22 trace investigation:
#   1. replay u_rel vs j_rel (whole-chain vs J-selective corruption)
#   2. warm-vs-replay output-slab diff per row block (output vs delivery)
#   3. warm-vs-replay multipole/local slab diffs (B2M/M2M vs M2L/L2L vs L2B)
#   4. CUDA_GRAPH_LIFECYCLE=false A/B (graph-execution causality)
# Plus: record-step (capture+instantiate+launch) vs warm and replay-vs-replay
# bit-consistency. Device-only; cheap (single cube n=2e4 P=4 F64 config).
using CUDA, Random
import FLOWVPM
const vpm = FLOWVPM
const ffmm = FLOWVPM.fmm

CUDA.functional() || error("CUDA is not functional")
CUDA.versioninfo()

const N = parse(Int, get(ENV, "FM048_DIAG_N", "20000"))
const SEED = parse(Int, get(ENV, "FM048_DIAG_SEED", "48048"))
const P = 4
const RHO_T = 4.211

function build(::Type{R}; UJ=vpm.UJ_fmm) where R
    rng = MersenneTwister(SEED)
    sigma = R(2 * N^(-1 / 3))
    pf = vpm.ParticleField(N, R; formulation=vpm.rVPM,
        kernel=vpm.gaussianerf, viscous=vpm.Inviscid(), SFS=vpm.noSFS,
        UJ, arraytype=CuArray,
        fmm=vpm.FMM(; p=P, ncrit=50, theta=0.4, autotune_p=false,
            autotune_ncrit=false, autotune_reg_error=false))
    for _ in 1:N
        vpm.add_particle(pf, rand(rng, R, 3),
            (R(2) .* rand(rng, R, 3) .- one(R)) ./ R(N), sigma)
    end
    UJ === vpm.UJ_fmm && vpm.radix_fmm_settings!(pf; expansion_order=P, rho_t=RHO_T)
    return pf
end

relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) /
                    max(sum(abs2, Float64.(b)), eps()))

function particle_errors(pf, ref)
    A = Array(pf.particles); B = Array(ref.particles)
    (u=relrms(A[vpm.U_INDEX, 1:N], B[vpm.U_INDEX, 1:N]),
     j=relrms(A[vpm.J_INDEX, 1:N], B[vpm.J_INDEX, 1:N]))
end

# snapshot the device slabs that localize the failing stage (body_perm rules
# out a sorted-order red herring: output columns are grid-sorted)
function snap(state)
    (out=Array(view(state.output, :, 1:N)),
     mp_phi=Array(state.multipoles.phi), mp_chi=Array(state.multipoles.chi),
     lc_phi=Array(state.locals.phi), lc_chi=Array(state.locals.chi),
     bp=Array(view(state.body_perm, 1:N)))
end

function slabdiff(tag, a, b)
    println("[$tag] perm_stable=", a.bp == b.bp)
    per_row = [relrms(a.out[r, :], b.out[r, :]) for r in 1:size(a.out, 1)]
    println("[$tag] output per-row relrms: ", per_row)
    println("[$tag] output U-block(2:4)=", relrms(a.out[2:4, :], b.out[2:4, :]),
        " J-block(5:13)=", relrms(a.out[5:13, :], b.out[5:13, :]))
    println("[$tag] multipoles phi=", relrms(a.mp_phi, b.mp_phi),
        " chi=", relrms(a.mp_chi, b.mp_chi))
    println("[$tag] locals    phi=", relrms(a.lc_phi, b.lc_phi),
        " chi=", relrms(a.lc_chi, b.lc_chi))
end

ref = build(Float64; UJ=vpm.UJ_direct)
vpm.UJ_direct(ref)
gpu = build(Float64)

# ---- call 1: warm (uncaptured launch-sequence body) ----
vpm.UJ_fmm(gpu)
st = vpm._radix_fmm_couplings[gpu].cache.state
hctx = st.interaction_list
hctx isa ffmm.DeviceHierarchicalM2LContext ||
    error("expected DeviceHierarchicalM2LContext, got $(typeof(hctx)) — graph fields absent")
S1 = snap(st)
e1 = particle_errors(gpu, ref)
println("[call1 warm/uncaptured] u_rel=", e1.u, " j_rel=", e1.j,
    " graph_exec=", hctx.graph_exec !== nothing,
    " eligible=", ffmm._cuda_graph_eligible(st))

# ---- call 2: record + instantiate + launch ----
vpm.UJ_fmm(gpu)
S2 = snap(st)
e2 = particle_errors(gpu, ref)
println("[call2 record+launch] u_rel=", e2.u, " j_rel=", e2.j,
    " graph_exec=", hctx.graph_exec !== nothing,
    " graph_epoch_ok=", hctx.graph_epoch == hctx.epoch_id)
slabdiff("record vs warm", S2, S1)

# ---- calls 3-4: replay ----
exec2 = hctx.graph_exec
vpm.UJ_fmm(gpu)
S3 = snap(st)
e3 = particle_errors(gpu, ref)
println("[call3 replay] u_rel=", e3.u, " j_rel=", e3.j,
    " same_exec=", hctx.graph_exec === exec2)
slabdiff("replay vs warm", S3, S1)
vpm.UJ_fmm(gpu)
S4 = snap(st)
println("[call4 replay] replay-vs-replay bit-consistent=",
    S4.out == S3.out, " (relrms=", relrms(S4.out, S3.out), ")")

# ---- check 4: graph-execution causality via runtime flip ----
ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, false)
vpm.UJ_fmm(gpu)
S5 = snap(st)
e5 = particle_errors(gpu, ref)
println("[graph OFF, launch-sequence body] u_rel=", e5.u, " j_rel=", e5.j)
slabdiff("graphoff vs warm", S5, S1)

ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, true)
vpm.UJ_fmm(gpu)  # replays the previously instantiated exec (same epoch)
S6 = snap(st)
e6 = particle_errors(gpu, ref)
println("[graph back ON, replay] u_rel=", e6.u, " j_rel=", e6.j,
    " matches_prior_replay=", S6.out == S3.out)

# ---- sfs=true call shape shares the same graph: confirm identical behavior ----
vpm.UJ_fmm(gpu; sfs=true, reset_sfs=true)
e7 = particle_errors(gpu, ref)
println("[replay, sfs=true shape] u_rel=", e7.u, " j_rel=", e7.j)

println("fm048 replay diagnostic complete")
