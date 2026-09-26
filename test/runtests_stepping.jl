# Host stepping unit tests: the per-particle CPU loops and the broadcast
# implementations of the same update agree; the freestream enters once; core
# spreading grows sigma as sqrt(sigma^2 + 2 nu dt); particle removal keeps
# the field consistent. Written 2026-09-26.
using Test, Random, StaticArrays, LinearAlgebra
import FLOWVPM
const vpm = FLOWVPM

function random_field(formulation; n = 64, seed = 1, kwargs...)
    Random.seed!(seed)
    pf = vpm.ParticleField(n + 4; formulation, kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct, kwargs...)
    for i in 1:n
        vpm.add_particle(pf, SVector{3,Float64}(randn(3)), SVector{3,Float64}(randn(3)), 0.05 + 0.1 * rand())
    end
    pf.particles[vpm.U_INDEX, 1:n] .= randn(3, n)
    pf.particles[vpm.J_INDEX, 1:n] .= 0.1 .* randn(9, n)
    pf.particles[vpm.M_INDEX, 1:n] .= 0.01 .* randn(length(vpm.M_INDEX), n)
    pf.particles[vpm.SFS_INDEX, 1:n] .= 0.01 .* randn(3, n)
    pf.particles[vpm.C_INDEX[1], 1:n] .= 0.5
    return pf
end
same(a, b) = maximum(abs.(a .- b)) <= 64 * eps() * max(1.0, maximum(abs.(b)))

@testset "CPU loop and broadcast RK3/Euler updates agree" begin
    zeta0 = vpm.kernel_gaussianerf.zeta(0.0); Uinf = SVector(1.0, -0.5, 0.25); dt = 1e-3
    for (form, cpu!, bc!) in ((vpm.formulation_rVPM, vpm.update_particle_states_cpu_reformulated!, vpm.update_particle_states_broadcast_reformulated!),
                             (vpm.ClassicVPM{Float64}(), vpm.update_particle_states_cpu_classic!, vpm.update_particle_states_broadcast_classic!))
        f = form isa vpm.ReformulatedVPM ? form.f : 0.0; g = form isa vpm.ReformulatedVPM ? form.g : 0.0
        for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
            A = random_field(form); B = random_field(form)
            cpu!(A, a, b, dt, Uinf, f, g, zeta0); bc!(B, a, b, dt, Uinf, f, g, zeta0)
            @test same(A.particles[:, 1:A.np], B.particles[:, 1:B.np])
        end
    end
    A = random_field(vpm.formulation_rVPM); B = random_field(vpm.formulation_rVPM)
    f, g = vpm.formulation_rVPM.f, vpm.formulation_rVPM.g
    vpm._euler_cpu_reformulated!(A, dt, Uinf, f, g, zeta0); vpm._euler_broadcast_reformulated!(B, dt, Uinf, f, g, zeta0)
    @test same(A.particles[:, 1:A.np], B.particles[:, 1:B.np])
    A = random_field(vpm.ClassicVPM{Float64}()); B = random_field(vpm.ClassicVPM{Float64}())
    vpm._euler_cpu_classic!(A, dt, Uinf, zeta0); vpm._euler_broadcast_classic!(B, dt, Uinf, zeta0)
    @test same(A.particles[:, 1:A.np], B.particles[:, 1:B.np])
end

@testset "a lone particle convects with the freestream exactly once" begin
    for integration in (vpm.euler, vpm.rungekutta3)
        pf = vpm.ParticleField(4; formulation = vpm.formulation_rVPM, kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct,
                               Uinf = t -> SVector(2.0, 0.0, -1.0), integration)
        vpm.add_particle(pf, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1e-3), 0.1)
        vpm.nextstep(pf, 0.01)
        @test vpm.get_X(pf, 1) ≈ [0.02, 0.0, -0.01] atol = 1e-12
        @test vpm.get_Gamma(pf, 1) ≈ [0.0, 0.0, 1e-3] atol = 1e-12      # no self-stretching
    end
end

@testset "core spreading grows sigma by sqrt(sigma^2 + 2 nu dt)" begin
    nu = 1.5e-5; dt = 0.02
    pf = vpm.ParticleField(8; formulation = vpm.formulation_rVPM, kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct,
                           viscous = vpm.CoreSpreading(nu, 0.1; beta = 1e9), integration = vpm.euler)
    for i in 1:4; vpm.add_particle(pf, SVector(0.3i, 0.0, 0.0), SVector(0.0, 0.0, 1e-3), 0.05 + 0.01i); end
    s0 = copy(pf.particles[vpm.SIGMA_INDEX, 1:4])
    vpm.viscousdiffusion(pf, pf.viscous, dt)
    @test pf.particles[vpm.SIGMA_INDEX, 1:4] ≈ sqrt.(s0.^2 .+ 2nu * dt) atol = 1e-15
end

@testset "remove_particle moves the last particle into the freed slot" begin
    pf = random_field(vpm.formulation_rVPM; n = 10)
    before = copy(pf.particles[:, 1:10])
    vpm.remove_particle(pf, 3)
    @test vpm.get_np(pf) == 9
    @test pf.particles[vpm.X_INDEX, 3] == before[vpm.X_INDEX, 10] && pf.particles[vpm.GAMMA_INDEX, 3] == before[vpm.GAMMA_INDEX, 10]
    @test pf.particles[:, 1:2] == before[:, 1:2] && pf.particles[:, 4:9] == before[:, 4:9]
    @test_throws ErrorException vpm.remove_particle(pf, 10)
    @test_throws ErrorException vpm.remove_particle(pf, 0)
end
