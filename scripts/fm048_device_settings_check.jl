# Task 047 device wiring check (run inside the 048 H200 job): a
# construction-locked radix setting flipped AFTER the device cache is built
# must error loudly at the next device step (verify_locked_radix_settings at
# _radix_cache_device_step! entry), and restoring the value must clear it.
using Test, Random, CUDA
import FLOWVPM
const vpm = FLOWVPM
const fmm = FLOWVPM.fmm

CUDA.functional() || error("CUDA not functional on this node")

n = 4000
rng = MersenneTwister(4048)
sigma = 2.0 * (1.0 / n)^(1 / 3)
pfield = vpm.ParticleField(n, Float64;
    formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
    SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
    UJ=vpm.UJ_fmm,
    fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4,
        autotune_p=false, autotune_ncrit=false, autotune_reg_error=false),
    arraytype=CuArray)
for _ in 1:n
    vpm.add_particle(pfield, rand(rng, 3), (2 .* rand(rng, 3) .- 1) ./ n, sigma)
end

@testset "047 device construction-lock (late flip errors loudly)" begin
    # first evaluation builds the device cache (snapshot taken at construction)
    vpm.UJ_fmm(pfield; reset=true, autotune=false)
    old = fmm.radix_setting(:CUDA_NEARFIELD_GH_MODE)
    flipped = old === :shipped ? :fp32 : :shipped
    fmm.set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, flipped)
    err = try
        vpm.UJ_fmm(pfield; reset=true, autotune=false)
        nothing
    catch e
        e
    end
    fmm.set_radix_setting!(:CUDA_NEARFIELD_GH_MODE, old)
    @test err !== nothing
    msg = err === nothing ? "" : sprint(showerror, err)
    @test occursin("construction-locked", msg)
    @test occursin("CUDA_NEARFIELD_GH_MODE", msg)
    # restored value steps cleanly again
    vpm.UJ_fmm(pfield; reset=true, autotune=false)
    @test true
end
println("fm048 device settings check complete")
