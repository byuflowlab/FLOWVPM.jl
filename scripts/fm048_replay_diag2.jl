# Task-048 graph-replay J-defect diagnostic v2 (2026-08-22). Diag v1 (job
# 13299959) did NOT reproduce the failure: warm, record, replay, graph-off,
# and sfs shapes all delivered j_rel = 1.9087e-3 identically on an A/B-style
# random cube with np == maxparticles, no statics, and no @allocated
# wrappers. Job 13298230's testset measured replay j_rel ~ 0.1098 on its
# FIRST spec iteration (cube n=2e4 F64 P4 rho 4.211), so the trigger is one
# of the in-iteration differences. This script runs a 2^3 factorial over
# exactly those factors, replicating the testset's build (CPU build ->
# extra-capacity GPU copy, testset seed 33025+n) and its exact call sequence,
# printing u/j error after EVERY call:
#   statics   in (false, true)  — 3 static particles w/ seeded SFS rows
#   extracap  in (false, true)  — maxparticles = np + 256 vs np
#   allocwrap in (false, true)  — @allocated / CUDA.@allocated wrappers
using CUDA, Random
import FLOWVPM
const vpm = FLOWVPM
const ffmm = FLOWVPM.fmm

CUDA.functional() || error("CUDA is not functional")

const N = 20000
const P = 4
const RHO_T = 4.211
const SEED = 33025 + N          # FMM034_SEED + n: the testset's exact field
const STATIC_IDXS = (2, 17, 201)

function build_cpu(; UJ=vpm.UJ_fmm, maxparticles=N)
    rng = MersenneTwister(SEED)
    sigma = 2.0 * (1.0 / N)^(1 / 3)
    pf = vpm.ParticleField(maxparticles, Float64; formulation=vpm.rVPM,
        kernel=vpm.gaussianerf, viscous=vpm.Inviscid(), SFS=vpm.noSFS,
        transposed=true, integration=vpm.rungekutta3, UJ,
        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false))
    for _ in 1:N
        vpm.add_particle(pf, rand(rng, 3), (2 .* rand(rng, 3) .- 1) ./ N, sigma)
    end
    return pf
end

# testset's fmm034_to_gpu, extra_capacity parameterized
function to_gpu(cpu; extra_capacity)
    maxp = cpu.maxparticles + extra_capacity
    gpu = vpm.ParticleField(maxp, Float64; formulation=vpm.rVPM,
        kernel=vpm.gaussianerf, viscous=vpm.Inviscid(), SFS=vpm.noSFS,
        transposed=true, integration=vpm.rungekutta3, UJ=vpm.UJ_fmm,
        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false), arraytype=CuArray)
    gpu.np = cpu.np
    view(gpu.particles, :, 1:cpu.np) .=
        CUDA.CuArray{Float64}(Array(cpu.particles)[:, 1:cpu.np])
    return gpu
end

relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) /
                    max(sum(abs2, Float64.(b)), eps()))
function errs(pf, ref)
    A = Array(pf.particles); B = Array(ref.particles)
    (u=relrms(A[vpm.U_INDEX, 1:N], B[vpm.U_INDEX, 1:N]),
     j=relrms(A[vpm.J_INDEX, 1:N], B[vpm.J_INDEX, 1:N]))
end

# one direct-sum reference per statics arm (statics change the field)
function build_ref(statics)
    cpu = build_cpu(; UJ=vpm.UJ_direct)
    statics && seed_statics!(cpu)
    ref = to_gpu(cpu; extra_capacity=256)
    vpm.UJ_direct(ref)
    vpm.Estr_direct!(ref)   # testset fidelity: runs before gpu's first call
    return ref
end

function seed_statics!(pf)
    for i in STATIC_IDXS
        vpm.set_static(pf, i, 1.0)
        pf.particles[vpm.SFS_INDEX, i] .= (3.0, -2.0, 1.0)
    end
    return pf
end

function run_arm(; statics, extracap, allocwrap, ref)
    cpu = build_cpu()
    statics && seed_statics!(cpu)
    gpu = to_gpu(cpu; extra_capacity=extracap ? 256 : 0)
    vpm.radix_fmm_settings!(gpu; expansion_order=P, rho_t=RHO_T)
    tag = "statics=$statics extracap=$extracap allocwrap=$allocwrap"
    report(k) = (e = errs(gpu, ref); println("[$tag] call$k u=$(e.u) j=$(e.j)"))
    vpm.UJ_fmm(gpu; sfs=true); report(1)                       # L181
    vpm.UJ_fmm(gpu; sfs=true); report(2)                       # L266
    if allocwrap
        ha = @allocated vpm.UJ_fmm(gpu); report(3)             # L269
        hs = @allocated vpm.UJ_fmm(gpu; sfs=true); report(4)   # L270
        da = CUDA.@allocated vpm.UJ_fmm(gpu); report(5)        # L271
        ds = CUDA.@allocated vpm.UJ_fmm(gpu; sfs=true); report(6) # L272
        println("[$tag] allocs host=($ha,$hs) device=($da,$ds)")
    else
        vpm.UJ_fmm(gpu); report(3)
        vpm.UJ_fmm(gpu; sfs=true); report(4)
        vpm.UJ_fmm(gpu); report(5)
        vpm.UJ_fmm(gpu; sfs=true); report(6)
    end
    st = vpm._radix_fmm_couplings[gpu].cache.state
    hctx = st.interaction_list
    println("[$tag] eligible=", ffmm._cuda_graph_eligible(st),
        " graph=", hctx.graph_exec !== nothing,
        " epoch_ok=", hctx.graph_epoch == hctx.epoch_id)
    vpm.UJ_fmm(gpu; sfs=true); report(7)                       # L285
    vpm.UJ_fmm(gpu; sfs=true); report(8)                       # L290 (replay measure)
    # ground truth: graph off must restore accuracy if the arm is corrupted
    ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, false)
    vpm.UJ_fmm(gpu; sfs=true); report("9-graphoff")
    ffmm.set_radix_setting!(:CUDA_GRAPH_LIFECYCLE, true)
    vpm.UJ_fmm(gpu; sfs=true); report("10-graphon")
    vpm.clear_radix_fmm_cache!(gpu)
    GC.gc(); CUDA.reclaim()
    return nothing
end

for statics in (false, true)
    ref = build_ref(statics)
    for extracap in (false, true), allocwrap in (false, true)
        run_arm(; statics, extracap, allocwrap, ref)
    end
    GC.gc(); CUDA.reclaim()
end
println("fm048 replay diagnostic v2 complete")
