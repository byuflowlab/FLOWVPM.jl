# Decision-gate benchmark for the Metal.jl side-track (see
# .claude/plans/we-need-to-make-idempotent-spark.md). Compares the real
# production CPU path (UJ_fmm) against the only GPU path that exists on this
# branch (UJ_direct, brute-force @metal kernel in ext/FLOWVPMMetalExt.jl) --
# NOT an apples-to-apples same-algorithm comparison, since FMM+Metal doesn't
# exist. The question this answers: does GPU direct beat CPU FMM at
# production particle counts, i.e. is there any reason to keep going?
using FLOWVPM
using Metal
import Random

const NS = (1_000, 5_000, 20_000, 50_000, 100_000)

# Apple GPUs have no Float64 support whatsoever (not a bug -- hardware
# limitation), so the Metal side runs Float32 while the CPU-FMM side stays
# at the production dtype (Float64). Not same-precision, but it's the
# realistic comparison: production CPU-FMM vs the only GPU path that exists.
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

function build_metal_from(cpu_pfield)
    n = cpu_pfield.maxparticles
    R = eltype(cpu_pfield.particles)
    mfield = FLOWVPM.ParticleField(n, R; arraytype=Metal.MtlArray, np=cpu_pfield.np)
    mfield.particles .= Metal.MtlArray(cpu_pfield.particles)
    return mfield
end

function timeit_cpu_fmm(n; reps=3)
    t = Inf
    for _ in 1:reps
        pfield = build_pfield(n, Float64)
        dt = @elapsed FLOWVPM.UJ_fmm(pfield)
        t = min(t, dt)
    end
    return t
end

function timeit_metal_direct(n; reps=3)
    t = Inf
    for _ in 1:reps
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        FLOWVPM.UJ_direct(mfield) # warm up (compile)
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        dt = @elapsed (FLOWVPM.UJ_direct(mfield); Metal.synchronize())
        t = min(t, dt)
    end
    return t
end

println("n, CPU-FMM (s), Metal-direct (s), speedup (FMM/Metal)")
for n in NS
    t_cpu = timeit_cpu_fmm(n)
    t_metal = timeit_metal_direct(n)
    speedup = t_cpu / t_metal
    println("$n, $(round(t_cpu, sigdigits=4)), $(round(t_metal, sigdigits=4)), $(round(speedup, sigdigits=3))x")
end
