# Decision-gate benchmark using REAL simulation particle data (not random
# uniform particles), per user request -- random-particle CPU-FMM timings
# looked suspiciously high. Data: ~/Downloads/for_ryan/NREL_50_36_2_1.125/,
# a real FLOWUnsteady run's saved pfield snapshots. step_timing.csv gives
# n_particles per saved step; picks the steps whose particle count spans the
# same range as the earlier random-particle sweep (1e3-1e5), using whatever
# real snapshots are actually available (saved every 36 steps here).
using FLOWVPM
using Metal
import HDF5
import Random

const DATA_DIR = joinpath(homedir(), "Downloads", "for_ryan", "NREL_50_36_2_1.125")
const CASE = "NREL_50_36_2_1.125"

# steps chosen from step_timing.csv (only steps with saved pfield.<n>.h5
# exist -- every 36 steps): n_particles ~ 16k/47k/91k/127k/171k/238k
const STEPS = (36, 108, 216, 324, 468, 684)

function load_real_pfield(step; R=Float64)
    h5fname = "$(CASE)_pfield.$(step).h5"
    h5 = HDF5.h5open(joinpath(DATA_DIR, h5fname), "r")
    np = HDF5.read(h5["np"])
    close(h5)
    pfield = FLOWVPM.ParticleField(np, R)
    FLOWVPM.read!(pfield, h5fname; path=DATA_DIR)
    return pfield
end

function build_metal_from(cpu_pfield, R32)
    n = cpu_pfield.maxparticles
    mfield = FLOWVPM.ParticleField(n, R32; arraytype=Metal.MtlArray, np=cpu_pfield.np)
    mfield.particles .= Metal.MtlArray(Float32.(cpu_pfield.particles))
    return mfield
end

function timeit_cpu_fmm(step; reps=2)
    t = Inf
    n = 0
    for _ in 1:reps
        pfield = load_real_pfield(step)
        n = pfield.np
        dt = @elapsed FLOWVPM.UJ_fmm(pfield)
        t = min(t, dt)
    end
    return n, t
end

function timeit_metal_direct(step; reps=2)
    t = Inf
    n = 0
    for _ in 1:reps
        cpu = load_real_pfield(step)
        n = cpu.np
        mfield = build_metal_from(cpu, Float32)
        FLOWVPM.UJ_direct(mfield) # warm up
        cpu = load_real_pfield(step)
        mfield = build_metal_from(cpu, Float32)
        dt = @elapsed (FLOWVPM.UJ_direct(mfield); Metal.synchronize())
        t = min(t, dt)
    end
    return n, t
end

println("n (real), CPU-FMM (s), Metal-direct (s), speedup (FMM/Metal)")
flush(stdout)
for step in STEPS
    println("loading step=$step ..."); flush(stdout)
    n_cpu, t_cpu = timeit_cpu_fmm(step)
    println("  CPU-FMM done: n=$n_cpu t=$(round(t_cpu, sigdigits=4))s"); flush(stdout)
    n_gpu, t_metal = timeit_metal_direct(step)
    println("  Metal-direct done: n=$n_gpu t=$(round(t_metal, sigdigits=4))s"); flush(stdout)
    speedup = t_cpu / t_metal
    println("step=$step n=$n_cpu, $(round(t_cpu, sigdigits=4)), $(round(t_metal, sigdigits=4)), $(round(speedup, sigdigits=3))x")
    flush(stdout)
end
