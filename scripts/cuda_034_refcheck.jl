# Task 034 closing gate: the device-resident radix FMM coupling vs the
# checksummed 033 sampled-direct references.
#
# Builds the EXACT 033 case constructions (benchmark_033_common.jl: seeds,
# sigma conventions, FMM settings p=4/ncrit=50/theta=0.4, autotuning off),
# runs the coupled U/J solve through `UJ_fmm` on a CuArray-backed field
# (FastMultipole resident radix lifecycle, zero per-step body transfer), and
# compares sampled U (gate) and J (diagnostic) at the reference sample
# indices against the 033 CPU sampled-direct references
# (direct_reference_<case>_n<n>.csv; schema/index checks and the sha256 of
# each file are enforced by fm033_read_reference — the manifest itself is
# verified by sha256sum -c in cuda_034_run.sh before this script runs).
#
# Gate: Float64 u_rel_rms <= 1e-3 for every case x n (Integration Phase
# velocity tolerance). Float32 is reported alongside, not gated.
#
# Usage (GPU node, see cuda_034_run.sh):
#   julia --project=$ENVDIR scripts/cuda_034_refcheck.jl <FMDIR> [n ...]
# where <FMDIR> is the FastMultipole tree holding
# MATRIX_OPERATOR_REFACTOR/{scripts,data/flowvpm_baseline/references}.
# Default n grid: 10000 100000.
#
# Harness self-check without a GPU (host-resident transfer path, Float64
# only):  FM034_REFCHECK_HOST=1 julia --project=<env> scripts/cuda_034_refcheck.jl <FMDIR> 10000

const FM034_HOST_MODE = get(ENV, "FM034_REFCHECK_HOST", "0") == "1"

if !FM034_HOST_MODE
    import CUDA
    CUDA.functional() || error("CUDA is not functional on this node")
end

isempty(ARGS) && error("usage: cuda_034_refcheck.jl <FastMultipole-dir> [n ...]")
const FM034_FMDIR = abspath(ARGS[1])

# 033 harness: case builders, reference reader (schema + deterministic-index
# validation + per-file sha256), shared constants. Defines `vpm = FLOWVPM`.
include(joinpath(FM034_FMDIR, "MATRIX_OPERATOR_REFACTOR", "scripts",
    "benchmark_033_common.jl"))

const FM034_REFDIR = joinpath(FM034_FMDIR, "MATRIX_OPERATOR_REFACTOR",
    "data", "flowvpm_baseline", "references")
const FM034_NS = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [10_000, 100_000]
const FM034_U_GATE = 1e-3
# Gate scope: the historical 034 shipped-defaults contract covers cube+wake.
# Exploration cases added later to FM033_CASES (e.g. 037b's "rotor") are
# accuracy-gated per-row inside their own campaign sweeps against their
# checksummed references, not by this abort-on-fail preflight; include them
# here explicitly via FM034_REFCHECK_CASES when that gate is wanted.
const FM034_CASES = Tuple(split(get(ENV, "FM034_REFCHECK_CASES", "cube,wake"), ','))

vpm._FMM_HAS_RADIX || error("installed FastMultipole lacks the radix device interface")

# CuArray-backed copy of a CPU-built 033 field at precision R, identical
# solver settings (runtests_gpu_fmm_device.jl pattern)
function fm034_to_gpu(cpu_pfield, R)
    n = vpm.get_np(cpu_pfield)
    gpu = vpm.ParticleField(n, R;
        formulation=vpm.rVPM,
        kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(),
        SFS=vpm.noSFS,
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=fm033_settings(),
        arraytype=Main.CUDA.CuArray)
    gpu.np = n
    gpu.particles .= Main.CUDA.CuArray{R}(Array(cpu_pfield.particles)[:, 1:n])
    return gpu
end

# Sampled U/J errors of a (possibly device-backed) field against a reference
# from fm033_read_reference, computed on a host download (no scalar indexing
# of CuArrays; everything in Float64)
function fm034_sampled_errors(pfield, reference)
    A = Array(pfield.particles)
    u_err2 = u_ref2 = j_err2 = j_ref2 = 0.0
    u_max = 0.0
    for (si, bi) in enumerate(reference.indices)
        uref = reference.U[si]
        e2 = 0.0
        for k in 1:3
            d = Float64(A[vpm.U_INDEX[k], bi]) - uref[k]
            e2 += d * d
            u_ref2 += uref[k]^2
        end
        u_err2 += e2
        u_max = max(u_max, sqrt(e2))
        jref = reference.J[si]
        for k in 1:9
            d = Float64(A[vpm.J_INDEX[k], bi]) - jref[k]
            j_err2 += d * d
            j_ref2 += jref[k]^2
        end
    end
    return (
        u_rel_rms=sqrt(u_err2 / max(u_ref2, eps(Float64))),
        u_abs_rms=sqrt(u_err2 / reference.samples),
        u_max_err=u_max,
        j_rel_rms=sqrt(j_err2 / max(j_ref2, eps(Float64))),
    )
end

println("=== 034 reference comparison: device-resident radix FMM vs 033 " *
    "checksummed sampled-direct references")
println("mode: $(FM034_HOST_MODE ? "HOST (transfer path, harness self-check)" : "DEVICE (CuArray-resident)")")
println("cases: $(join(FM034_CASES, ", "))  n: $(join(FM034_NS, ", "))  gate: " *
    "Float64 u_rel_rms <= $(FM034_U_GATE)")

results = NamedTuple[]
for case in FM034_CASES, n in FM034_NS
    t_build = @elapsed cpu = fm033_build(case, n)
    n_actual = vpm.get_np(cpu)
    refpath = fm033_reference_path(FM034_REFDIR, case, n)
    reference = fm033_read_reference(refpath, case, n, n_actual)
    println("[$case n=$n] built (n_actual=$n_actual, $(round(t_build, digits=1))s); " *
        "reference $(basename(refpath)) samples=$(reference.samples) " *
        "sha256=$(reference.checksum)")

    precisions = FM034_HOST_MODE ? (Float64,) : (Float64, Float32)
    for R in precisions
        local err, t_solve
        if FM034_HOST_MODE
            vpm.UJ_fmm_gpu!(cpu)     # transfer-based host radix path
            err = fm034_sampled_errors(cpu, reference)
            t_solve = NaN
        else
            gpu = fm034_to_gpu(cpu, R)
            vpm.UJ_fmm(gpu)          # routes to the resident device lifecycle
            Main.CUDA.synchronize()
            t_solve = @elapsed begin  # warm re-solve, coarse sanity only
                vpm.UJ_fmm(gpu)
                Main.CUDA.synchronize()
            end
            err = fm034_sampled_errors(gpu, reference)
            # 023 counter contract on the device path
            st = vpm._radix_fmm_couplings[gpu]
            c = st.cache.state.counters
            (c.body_uploads == 0 && c.expansion_host_copies == 0) ||
                error("device counter contract violated [$case n=$n $R]: " *
                    "body_uploads=$(c.body_uploads) " *
                    "expansion_host_copies=$(c.expansion_host_copies)")
            vpm.clear_radix_fmm_cache!(gpu)
        end
        push!(results, (; case, n, R, err..., t_solve))
        println("[$case n=$n $R] u_rel_rms=$(err.u_rel_rms) " *
            "u_abs_rms=$(err.u_abs_rms) u_max_err=$(err.u_max_err) " *
            "j_rel_rms=$(err.j_rel_rms) warm_solve_s=$(t_solve)")
    end
end

println("\ncase,n,precision,u_rel_rms,u_abs_rms,u_max_err,j_rel_rms,warm_solve_s")
for r in results
    println(join((r.case, r.n, r.R, r.u_rel_rms, r.u_abs_rms, r.u_max_err,
        r.j_rel_rms, r.t_solve), ','))
end

failures = [r for r in results if r.R == Float64 && !(r.u_rel_rms <= FM034_U_GATE)]
if !isempty(failures)
    for r in failures
        println("GATE FAIL: [$(r.case) n=$(r.n) Float64] u_rel_rms=$(r.u_rel_rms) > $(FM034_U_GATE)")
    end
    error("034 reference-comparison gate failed for $(length(failures)) case(s)")
end
println("\n034 reference-comparison gate PASSED " *
    "(all Float64 u_rel_rms <= $(FM034_U_GATE))")
