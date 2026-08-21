# GPU regression test for the direct (no-FMM) N-body kernels in
# ext/FLOWVPMCUDAExt.jl. Only included by runtests.jl when a functional
# CUDA-capable GPU is present in the current environment (CUDA is an
# optional/weak dependency of FLOWVPM, so this is a no-op on CPU-only CI).
using Test
import Random
vpm = FLOWVPM

function gpu_test_build_cpu_pfield(n, R=Float64; seed=1, static_frac=0.0)
    Random.seed!(seed)
    pfield = vpm.ParticleField(n, R)
    for i in 1:n
        X = rand(R, 3) .* 2 .- 1
        Gamma = rand(R, 3) .* 2 .- 1
        sigma = R(0.1) + R(0.05) * rand(R)
        vpm.add_particle(pfield, X, Gamma, sigma; static=(rand() < static_frac))
    end
    return pfield
end

function gpu_test_build_gpu_from(cpu_pfield, R=Float64)
    n = cpu_pfield.maxparticles
    gpu_pfield = vpm.ParticleField(n, R; arraytype=CUDA.CuArray, np=cpu_pfield.np)
    gpu_pfield.particles .= CUDA.CuArray(cpu_pfield.particles)
    return gpu_pfield
end

function gpu_test_relerr(a, b)
    a, b = Array(a), Array(b)
    return maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))
end

@testset "GPU direct-sum kernels (gpu_direct!/zeta_direct/Estr_direct!)" begin
    for R in (Float64, Float32), n in (64, 1024, 4096), static_frac in (0.0, 0.3)
        tol = R == Float64 ? 1e-8 : 1e-2

        cpu = gpu_test_build_cpu_pfield(n, R; static_frac=static_frac)
        gpu = gpu_test_build_gpu_from(cpu, R)
        vpm.UJ_direct(cpu)
        vpm.UJ_direct(gpu)
        @test gpu_test_relerr(view(cpu.particles, vpm.U_INDEX, :), view(gpu.particles, vpm.U_INDEX, :)) < tol
        @test gpu_test_relerr(view(cpu.particles, vpm.J_INDEX, :), view(gpu.particles, vpm.J_INDEX, :)) < tol

        cpu2 = gpu_test_build_cpu_pfield(n, R; static_frac=static_frac)
        gpu2 = gpu_test_build_gpu_from(cpu2, R)
        vpm.zeta_direct(cpu2)
        vpm.zeta_direct(gpu2)
        @test gpu_test_relerr(view(cpu2.particles, vpm.J_INDEX[1:3], :), view(gpu2.particles, vpm.J_INDEX[1:3], :)) < tol

        cpu3 = gpu_test_build_cpu_pfield(n, R; static_frac=static_frac)
        gpu3 = gpu_test_build_gpu_from(cpu3, R)
        vpm.UJ_direct(cpu3)
        vpm.UJ_direct(gpu3)
        vpm.Estr_direct!(cpu3)
        vpm.Estr_direct!(gpu3)
        @test gpu_test_relerr(view(cpu3.particles, vpm.SFS_INDEX, :), view(gpu3.particles, vpm.SFS_INDEX, :)) < tol
    end
end

function gpu_test_build_gpu_pfield_directly(n, R=Float64; seed=1, static_frac=0.0)
    Random.seed!(seed)
    pfield = vpm.ParticleField(n, R; arraytype=CUDA.CuArray)
    for i in 1:n
        X = rand(R, 3) .* 2 .- 1
        Gamma = rand(R, 3) .* 2 .- 1
        sigma = R(0.1) + R(0.05) * rand(R)
        vpm.add_particle(pfield, X, Gamma, sigma; static=(rand() < static_frac))
    end
    return pfield
end

@testset "add_particle builds a GPU-backed field directly (no CPU bulk-copy)" begin
    for R in (Float64, Float32), n in (1, 64, 1024), static_frac in (0.0, 0.3)
        tol = R == Float64 ? 1e-8 : 1e-2

        cpu = gpu_test_build_cpu_pfield(n, R; static_frac=static_frac)
        gpu = gpu_test_build_gpu_pfield_directly(n, R; static_frac=static_frac)

        @test gpu.np == cpu.np
        @test gpu_test_relerr(view(cpu.particles, vpm.X_INDEX, :), view(gpu.particles, vpm.X_INDEX, :)) < tol
        @test gpu_test_relerr(view(cpu.particles, vpm.GAMMA_INDEX, :), view(gpu.particles, vpm.GAMMA_INDEX, :)) < tol
        @test gpu_test_relerr(view(cpu.particles, vpm.SIGMA_INDEX:vpm.SIGMA_INDEX, :), view(gpu.particles, vpm.SIGMA_INDEX:vpm.SIGMA_INDEX, :)) < tol
        @test Array(view(gpu.particles, vpm.STATIC_INDEX:vpm.STATIC_INDEX, :)) == view(cpu.particles, vpm.STATIC_INDEX:vpm.STATIC_INDEX, :)

        # and the physics on a directly-built GPU field still matches the CPU reference
        vpm.UJ_direct(cpu)
        vpm.UJ_direct(gpu)
        @test gpu_test_relerr(view(cpu.particles, vpm.U_INDEX, :), view(gpu.particles, vpm.U_INDEX, :)) < tol
        @test gpu_test_relerr(view(cpu.particles, vpm.J_INDEX, :), view(gpu.particles, vpm.J_INDEX, :)) < tol
    end
end
