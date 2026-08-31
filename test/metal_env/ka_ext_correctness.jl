# Correctness check for the wired-in ext/FLOWVPMKAMetalExt.jl dispatch
# (FLOWVPM.gpu_direct!/gpu_zeta_direct!/gpu_estr_direct! for MtlArray via
# KernelAbstractions) against the CPU reference -- NOT the standalone
# ka_direct_bench.jl script, which only exercised gpu_direct! pre-wiring.
using FLOWVPM
using Metal
using KernelAbstractions
import Random

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

relerr(a, b) = (a = Array(a); b = Array(b); maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b))))

println("=== Correctness: ext dispatch (KA-Metal) vs CPU reference, UJ_direct + Estr_direct! ===")
for n in (64, 200, 1000)
    cpu = build_pfield(n, Float32)
    mfield = build_metal_from(cpu)

    FLOWVPM.UJ_direct(cpu; sfs=true, reset_sfs=true)
    FLOWVPM.UJ_direct(mfield; sfs=true, reset_sfs=true)

    eU = relerr(view(cpu.particles, FLOWVPM.U_INDEX, :), view(mfield.particles, FLOWVPM.U_INDEX, :))
    eJ = relerr(view(cpu.particles, FLOWVPM.J_INDEX, :), view(mfield.particles, FLOWVPM.J_INDEX, :))
    eSFS = relerr(view(cpu.particles, FLOWVPM.SFS_INDEX, :), view(mfield.particles, FLOWVPM.SFS_INDEX, :))
    println("n=$n  max U relerr=$eU  max J relerr=$eJ  max SFS(Estr) relerr=$eSFS")
end

println()
println("=== Correctness: ext dispatch (KA-Metal) vs CPU reference, zeta_direct ===")
for n in (64, 200, 1000)
    cpu = build_pfield(n, Float32)
    mfield = build_metal_from(cpu)

    FLOWVPM.zeta_direct(cpu)
    FLOWVPM.zeta_direct(mfield)

    eZ = relerr(view(cpu.particles, FLOWVPM.VORTICITY_INDEX, :), view(mfield.particles, FLOWVPM.VORTICITY_INDEX, :))
    println("n=$n  max vorticity(zeta) relerr=$eZ")
end
