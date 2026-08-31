# H200 gate check for the KernelAbstractions generalization (metal-testing
# side-track): does a backend-agnostic KA kernel regress FLOWVPMCUDAExt.jl's
# hand-written, H200-validated CUDA kernels? Standalone script -- does NOT
# touch FLOWVPM.gpu_direct!/gpu_zeta_direct!/gpu_estr_direct! dispatch for
# CuArray (same "de-risk before wiring" pattern as
# test/metal_env/ka_direct_bench.jl used for Metal), so it cannot regress
# production CUDA runs just by being present in the repo.
#
# Two different questions bundled here, because the two CUDA kernel families
# have different risk profiles:
#   1. gpu_direct! (UJ_direct): CUDA's `gpu_atomic_square!` is TILED
#      (shared-memory blocking + atomic reduction) -- a genuinely different,
#      more optimized algorithm than the brute-force KA kernel proven on
#      Metal. This is the real gate: if KA regresses meaningfully at
#      production N (1e5-1e6, per the H200-validated range in CLAUDE.md),
#      it should NOT replace gpu_atomic_square!.
#   2. gpu_zeta_direct!/gpu_estr_direct!: CUDA's existing kernels are ALSO
#      brute-force (no tiling) -- same algorithm as the KA port. There's no
#      algorithmic regression risk here, only "does KA itself add overhead
#      on CUDA the way it doesn't on Metal" -- lower-stakes, but still worth
#      confirming on real hardware before wiring.
#
# Usage: julia --project=<env with CUDA, KernelAbstractions, FLOWVPM, FastMultipole> scripts/ka_cuda_bench.jl
# Writes a CSV to $KA_CUDA_BENCH_CSV (default ./ka_cuda_bench_<jobid-or-pid>.csv).

using FLOWVPM
using CUDA
using KernelAbstractions
import Random

const KA = KernelAbstractions

CUDA.functional() || error("CUDA not functional on this node")

# --- KA kernels: verbatim ports of ext/FLOWVPMKAMetalExt.jl's kernels
# (already H200-... no, Metal-validated for correctness/speed there),
# backend-obtained via KA.get_backend() so the same code runs on CUDA here
# without depending on a CUDA-specific KA symbol name. ---

@kernel function ka_direct_kernel!(out, @Const(s), n::Int32, kernel)
    j_target = @index(Global)
    if j_target <= n
        T = eltype(s)
        @inbounds tx = s[1, j_target]
        @inbounds ty = s[2, j_target]
        @inbounds tz = s[3, j_target]

        U1, U2, U3 = zero(T), zero(T), zero(T)
        J1, J2, J3, J4, J5, J6, J7, J8, J9 = zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T)

        const4 = T(0.25 / pi)

        i::Int32 = 1
        while i <= n
            @inbounds dX1 = tx - s[1, i]
            @inbounds dX2 = ty - s[2, i]
            @inbounds dX3 = tz - s[3, i]
            r2 = dX1^2 + dX2^2 + dX3^2
            r = sqrt(r2)

            @inbounds sigma = s[7, i]

            if r2 > zero(T) && abs(sigma) > zero(T)
                c4 = -const4 / (r*r2)
                @inbounds gam1 = c4 * s[4, i]
                @inbounds gam2 = c4 * s[5, i]
                @inbounds gam3 = c4 * s[6, i]

                g_sgm, dg_sgmdr = kernel(r/sigma)

                aux = dg_sgmdr/(sigma*r) - 3*g_sgm/r2

                crss1 = dX2*gam3 - dX3*gam2
                crss2 = dX3*gam1 - dX1*gam3
                crss3 = dX1*gam2 - dX2*gam1

                U1 += g_sgm * crss1
                U2 += g_sgm * crss2
                U3 += g_sgm * crss3

                gam1 *= g_sgm; gam2 *= g_sgm; gam3 *= g_sgm
                dX1 *= aux; dX2 *= aux; dX3 *= aux

                J1 += crss1 * dX1
                J2 += crss2 * dX1 - gam3
                J3 += crss3 * dX1 + gam2
                J4 += crss1 * dX2 + gam3
                J5 += crss2 * dX2
                J6 += crss3 * dX2 - gam1
                J7 += crss1 * dX3 - gam2
                J8 += crss2 * dX3 + gam1
                J9 += crss3 * dX3
            end
            i += Int32(1)
        end

        @inbounds out[1, j_target] = U1
        @inbounds out[2, j_target] = U2
        @inbounds out[3, j_target] = U3
        @inbounds out[4, j_target] = J1
        @inbounds out[5, j_target] = J2
        @inbounds out[6, j_target] = J3
        @inbounds out[7, j_target] = J4
        @inbounds out[8, j_target] = J5
        @inbounds out[9, j_target] = J6
        @inbounds out[10, j_target] = J7
        @inbounds out[11, j_target] = J8
        @inbounds out[12, j_target] = J9
    end
end

function ka_gpu_direct!(pfield, backend; workgroup=256)
    n = pfield.np
    n == 0 && return nothing
    P = pfield.particles
    T = eltype(P)
    s = view(P, 1:7, 1:n)
    out = KA.zeros(backend, T, 12, n)
    ka_direct_kernel!(backend, workgroup)(out, s, Int32(n), pfield.kernel.g_dgdr; ndrange=n)
    KA.synchronize(backend)
    view(P, FLOWVPM.U_INDEX, 1:n) .+= view(out, 1:3, :)
    view(P, FLOWVPM.J_INDEX, 1:n) .+= view(out, 4:12, :)
    return nothing
end

@kernel function ka_zeta_direct_kernel!(out, @Const(s), n::Int32, zeta)
    j_target = @index(Global)
    if j_target <= n
        @inbounds tx = s[1, j_target]
        @inbounds ty = s[2, j_target]
        @inbounds tz = s[3, j_target]
        T = eltype(s)
        acc1, acc2, acc3 = zero(T), zero(T), zero(T)
        i::Int32 = 1
        while i <= n
            @inbounds dX1 = tx - s[1, i]
            @inbounds dX2 = ty - s[2, i]
            @inbounds dX3 = tz - s[3, i]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)
            @inbounds sigma = s[7, i]
            zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)
            @inbounds acc1 += s[4, i]*zeta_sgm
            @inbounds acc2 += s[5, i]*zeta_sgm
            @inbounds acc3 += s[6, i]*zeta_sgm
            i += Int32(1)
        end
        @inbounds out[1, j_target] += acc1
        @inbounds out[2, j_target] += acc2
        @inbounds out[3, j_target] += acc3
    end
end

function ka_gpu_zeta_direct!(pfield, backend; workgroup=256)
    n = pfield.np
    n == 0 && return nothing
    P = pfield.particles
    T = eltype(P)
    s = view(P, 1:7, 1:n)
    out = KA.zeros(backend, T, 3, n)
    ka_zeta_direct_kernel!(backend, workgroup)(out, s, Int32(n), pfield.kernel.zeta; ndrange=n)
    KA.synchronize(backend)
    view(P, FLOWVPM.VORTICITY_INDEX, 1:n) .= view(out, 1:3, :)
    return nothing
end

@kernel function ka_estr_direct_kernel!(sfs_out, @Const(P), n::Int32, zeta, transposed::Bool,
                                         static_row::Int32, j1::Int32, j2::Int32, j3::Int32,
                                         j4::Int32, j5::Int32, j6::Int32, j7::Int32, j8::Int32, j9::Int32)
    j_target = @index(Global)
    T = eltype(P)
    if j_target <= n
        @inbounds target_is_static = P[static_row, j_target]
        if target_is_static == 0
            @inbounds tx = P[1, j_target]
            @inbounds ty = P[2, j_target]
            @inbounds tz = P[3, j_target]
            @inbounds JT1 = P[j1, j_target]; @inbounds JT2 = P[j2, j_target]; @inbounds JT3 = P[j3, j_target]
            @inbounds JT4 = P[j4, j_target]; @inbounds JT5 = P[j5, j_target]; @inbounds JT6 = P[j6, j_target]
            @inbounds JT7 = P[j7, j_target]; @inbounds JT8 = P[j8, j_target]; @inbounds JT9 = P[j9, j_target]
            acc1, acc2, acc3 = zero(T), zero(T), zero(T)
            i::Int32 = 1
            while i <= n
                @inbounds source_is_static = P[static_row, i]
                if source_is_static == 0
                    @inbounds sx = P[1, i]
                    @inbounds sy = P[2, i]
                    @inbounds sz = P[3, i]
                    dX1 = tx - sx
                    dX2 = ty - sy
                    dX3 = tz - sz
                    r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)
                    @inbounds sigma = P[7, i]
                    zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)
                    @inbounds GS1 = P[4, i]; @inbounds GS2 = P[5, i]; @inbounds GS3 = P[6, i]
                    @inbounds JS1 = P[j1, i]; @inbounds JS2 = P[j2, i]; @inbounds JS3 = P[j3, i]
                    @inbounds JS4 = P[j4, i]; @inbounds JS5 = P[j5, i]; @inbounds JS6 = P[j6, i]
                    @inbounds JS7 = P[j7, i]; @inbounds JS8 = P[j8, i]; @inbounds JS9 = P[j9, i]
                    if transposed
                        S1 = (JT1-JS1)*GS1 + (JT2-JS2)*GS2 + (JT3-JS3)*GS3
                        S2 = (JT4-JS4)*GS1 + (JT5-JS5)*GS2 + (JT6-JS6)*GS3
                        S3 = (JT7-JS7)*GS1 + (JT8-JS8)*GS2 + (JT9-JS9)*GS3
                    else
                        S1 = (JT1-JS1)*GS1 + (JT4-JS4)*GS2 + (JT7-JS7)*GS3
                        S2 = (JT2-JS2)*GS1 + (JT5-JS5)*GS2 + (JT8-JS8)*GS3
                        S3 = (JT3-JS3)*GS1 + (JT6-JS6)*GS2 + (JT9-JS9)*GS3
                    end
                    acc1 += zeta_sgm*S1
                    acc2 += zeta_sgm*S2
                    acc3 += zeta_sgm*S3
                end
                i += Int32(1)
            end
            @inbounds sfs_out[1, j_target] += acc1
            @inbounds sfs_out[2, j_target] += acc2
            @inbounds sfs_out[3, j_target] += acc3
        end
    end
end

function ka_gpu_estr_direct!(pfield, backend; workgroup=256)
    n = pfield.np
    n == 0 && return nothing
    P = pfield.particles
    T = eltype(P)
    out = KA.zeros(backend, T, 3, n)
    jrows = Int32.(FLOWVPM.J_INDEX)
    ka_estr_direct_kernel!(backend, workgroup)(
        out, P, Int32(n), pfield.kernel.zeta, pfield.transposed,
        Int32(FLOWVPM.STATIC_INDEX),
        jrows[1], jrows[2], jrows[3], jrows[4], jrows[5], jrows[6], jrows[7], jrows[8], jrows[9];
        ndrange=n)
    KA.synchronize(backend)
    view(P, FLOWVPM.SFS_INDEX, 1:n) .+= view(out, 1:3, :)
    return nothing
end

# --- harness ---

function build_pfield(n, R; seed=1)
    Random.seed!(seed)
    pfield = FLOWVPM.ParticleField(n, R)
    for i in 1:n
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
    cfield = FLOWVPM.ParticleField(n, R; arraytype=CUDA.CuArray, np=cpu_pfield.np)
    cfield.particles .= CUDA.CuArray(cpu_pfield.particles)
    return cfield
end

relerr(a, b) = (a = Array(a); b = Array(b); maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b))))

println("=== Correctness: KA-CUDA vs CPU reference (UJ_direct + zeta_direct + Estr_direct!) ===")
for n in (64, 200, 1000)
    cpu = build_pfield(n, Float32)
    cfield = build_cuda_from(cpu)
    backend = KA.get_backend(cfield.particles)

    FLOWVPM.UJ_direct(cpu; sfs=true, reset_sfs=true)
    ka_gpu_direct!(cfield, backend)
    ka_gpu_estr_direct!(cfield, backend)

    eU = relerr(view(cpu.particles, FLOWVPM.U_INDEX, :), view(cfield.particles, FLOWVPM.U_INDEX, :))
    eJ = relerr(view(cpu.particles, FLOWVPM.J_INDEX, :), view(cfield.particles, FLOWVPM.J_INDEX, :))
    eSFS = relerr(view(cpu.particles, FLOWVPM.SFS_INDEX, :), view(cfield.particles, FLOWVPM.SFS_INDEX, :))

    cpu2 = build_pfield(n, Float32)
    cfield2 = build_cuda_from(cpu2)
    FLOWVPM.zeta_direct(cpu2)
    ka_gpu_zeta_direct!(cfield2, backend)
    eZ = relerr(view(cpu2.particles, FLOWVPM.VORTICITY_INDEX, :), view(cfield2.particles, FLOWVPM.VORTICITY_INDEX, :))

    println("n=$n  U relerr=$eU  J relerr=$eJ  SFS(Estr) relerr=$eSFS  vorticity(zeta) relerr=$eZ")
end

const NS = (1_000, 5_000, 20_000, 50_000, 100_000, 300_000)

function timeit(f!, n; reps=2)
    t = Inf
    for _ in 1:reps
        cpu = build_pfield(n, Float32)
        cfield = build_cuda_from(cpu)
        f!(cfield)  # warm up (compile)
        cpu = build_pfield(n, Float32)
        cfield = build_cuda_from(cpu)
        dt = @elapsed (f!(cfield); CUDA.synchronize())
        t = min(t, dt)
    end
    return t
end

println()
println("=== Speed gate: CUDA tiled gpu_atomic_square! (wired) vs KA brute-force, CuArray, best-of-3 ===")
println("n, cuda_tiled_direct_s, ka_direct_s, ka/tiled, cuda_zeta_s (brute-force), ka_zeta_s, ka/cuda_zeta, cuda_estr_s (brute-force), ka_estr_s, ka/cuda_estr")

backend = KA.get_backend(CUDA.CuArray(zeros(Float32, 1)))

results = NamedTuple[]
for n in NS
    t_tiled = timeit(pf -> FLOWVPM.UJ_direct(pf), n)                 # wired dispatch: hand-written tiled gpu_atomic_square!
    t_ka    = timeit(pf -> ka_gpu_direct!(pf, backend), n)           # standalone KA brute-force

    t_cuda_zeta = timeit(pf -> FLOWVPM.zeta_direct(pf), n)           # wired dispatch: hand-written brute-force gpu_zeta_direct_kernel!
    t_ka_zeta   = timeit(pf -> ka_gpu_zeta_direct!(pf, backend), n)  # standalone KA brute-force

    t_cuda_estr = timeit(pf -> FLOWVPM.Estr_direct!(pf), n)          # wired dispatch: hand-written brute-force gpu_estr_direct_kernel!
    t_ka_estr   = timeit(pf -> ka_gpu_estr_direct!(pf, backend), n)  # standalone KA brute-force

    row = (n=n, cuda_tiled_direct_s=t_tiled, ka_direct_s=t_ka, ka_over_tiled=t_ka/t_tiled,
           cuda_zeta_s=t_cuda_zeta, ka_zeta_s=t_ka_zeta, ka_over_cuda_zeta=t_ka_zeta/t_cuda_zeta,
           cuda_estr_s=t_cuda_estr, ka_estr_s=t_ka_estr, ka_over_cuda_estr=t_ka_estr/t_cuda_estr)
    push!(results, row)
    println("$n, $(round(t_tiled,sigdigits=4)), $(round(t_ka,sigdigits=4)), $(round(t_ka/t_tiled,sigdigits=3))x, " *
            "$(round(t_cuda_zeta,sigdigits=4)), $(round(t_ka_zeta,sigdigits=4)), $(round(t_ka_zeta/t_cuda_zeta,sigdigits=3))x, " *
            "$(round(t_cuda_estr,sigdigits=4)), $(round(t_ka_estr,sigdigits=4)), $(round(t_ka_estr/t_cuda_estr,sigdigits=3))x")
end

csv_path = get(ENV, "KA_CUDA_BENCH_CSV", "ka_cuda_bench_$(get(ENV, "SLURM_JOB_ID", string(getpid()))).csv")
open(csv_path, "w") do io
    println(io, "n,cuda_tiled_direct_s,ka_direct_s,ka_over_tiled,cuda_zeta_s,ka_zeta_s,ka_over_cuda_zeta,cuda_estr_s,ka_estr_s,ka_over_cuda_estr")
    for r in results
        println(io, "$(r.n),$(r.cuda_tiled_direct_s),$(r.ka_direct_s),$(r.ka_over_tiled),$(r.cuda_zeta_s),$(r.ka_zeta_s),$(r.ka_over_cuda_zeta),$(r.cuda_estr_s),$(r.ka_estr_s),$(r.ka_over_cuda_estr)")
    end
end
println()
println("wrote $csv_path")

println()
println("=== Gate readout ===")
# Threshold: KA may replace CUDA's tiled gpu_atomic_square! only if it costs
# no more than 15% over the hand-written kernel at production scale
# (n=1e5-1e6, the H200-validated range per CLAUDE.md). Below that range,
# launch-overhead noise makes the ratio meaningless either way, so it isn't
# gated. Override via KA_CUDA_BENCH_THRESHOLD (e.g. "1.10").
const THRESHOLD = parse(Float64, get(ENV, "KA_CUDA_BENCH_THRESHOLD", "1.15"))
production_rows = filter(r -> r.n >= 100_000, results)
gate_direct_pass = all(r -> r.ka_over_tiled <= THRESHOLD, production_rows)
println("gpu_direct! (tiled) gate: threshold ka_over_tiled <= $(THRESHOLD)x at n>=100000.")
for r in production_rows
    verdict = r.ka_over_tiled <= THRESHOLD ? "PASS" : "FAIL"
    println("  n=$(r.n): ka_over_tiled=$(round(r.ka_over_tiled,sigdigits=3))x -> $verdict")
end
println("gpu_direct! overall: ", gate_direct_pass ? "PASS -- KA may replace gpu_atomic_square! on CUDA." :
                                                      "FAIL -- KEEP the hand-written tiled kernel; do not wire KA for gpu_direct! on CUDA.")
println()
println("gpu_zeta_direct!/gpu_estr_direct! gate: same algorithm (brute-force) on both sides already, so")
println("this is a pure KA-overhead check, not an algorithm-regression risk -- same threshold applied for consistency.")
gate_zeta_pass = all(r -> r.ka_over_cuda_zeta <= THRESHOLD, production_rows)
gate_estr_pass = all(r -> r.ka_over_cuda_estr <= THRESHOLD, production_rows)
println("gpu_zeta_direct! overall: ", gate_zeta_pass ? "PASS -- KA may replace gpu_zeta_direct_kernel! on CUDA." : "FAIL -- keep hand-written.")
println("gpu_estr_direct! overall: ", gate_estr_pass ? "PASS -- KA may replace gpu_estr_direct_kernel! on CUDA." : "FAIL -- keep hand-written.")
