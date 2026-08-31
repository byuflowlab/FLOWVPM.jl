using Test
using LinearAlgebra
import FLOWVPM

# BRAINSTORM 020 Phase 2R: frozen-gradient geometric rVPM integrator.

function expint_field(; Gamma=(1.0, 0.0, 0.0), sigma=0.1, J=zeros(9),
                        integration=FLOWVPM.euler_exp,
                        formulation=FLOWVPM.rVPM,
                        viscous=FLOWVPM.Inviscid(), transposed=true)
    pfield = FLOWVPM.ParticleField(4; integration, formulation, viscous,
                                   transposed)
    FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), Gamma, sigma)
    FLOWVPM.set_J(pfield, 1, J)
    return pfield
end

function frozen_operator(J, transposed)
    return transposed ?
        [J[1] J[2] J[3]; J[4] J[5] J[6]; J[7] J[8] J[9]] :
        [J[1] J[4] J[7]; J[2] J[5] J[8]; J[3] J[6] J[9]]
end

function geometric_expected(Gamma, sigma, J, dt, g, transposed)
    q = exp(dt*frozen_operator(J, transposed))*collect(Gamma)
    ratio = norm(q)/norm(Gamma)
    return q*ratio^(-3g), sigma*ratio^(-g), g*log(ratio)/dt
end

@testset "Frozen-gradient geometric integrator (euler_exp)" begin

    @testset "aligned strain retains physical amplification" begin
        for dtZ in (0.5, 1.0, 2.0, 5.0, 50.0)
            dt = 1.0
            J = zeros(9); J[1] = 5dtZ
            pf = expint_field(; J)
            FLOWVPM._euler_exp(pf, dt)
            p = FLOWVPM.get_particle(pf, 1)
            @test FLOWVPM.get_Gamma(p)[1] ≈ exp(2dtZ) rtol=2e-13
            @test FLOWVPM.get_sigma(p)[] ≈ 0.1exp(-dtZ) rtol=2e-13
            @test FLOWVPM.get_sigma(p)[] > 0
        end
    end

    @testset "general constant gradient, both operator conventions" begin
        Gamma = (0.7, -0.2, 0.4); sigma = 0.13; dt = 0.37
        J = [0.8, -0.3, 0.2, 0.5, -0.4, 0.1, -0.6, 0.7, -0.4]
        for transposed in (true, false)
            pf = expint_field(; Gamma, sigma, J, transposed)
            Gref, sref, zref = geometric_expected(Gamma, sigma, J, dt,
                                                   1/5, transposed)
            FLOWVPM._euler_exp(pf, dt)
            p = FLOWVPM.get_particle(pf, 1)
            @test collect(FLOWVPM.get_Gamma(p)) ≈ Gref rtol=2e-13 atol=2e-15
            @test FLOWVPM.get_sigma(p)[] ≈ sref rtol=2e-13
            @test FLOWVPM.get_M(p)[9] ≈ zref rtol=2e-13 atol=2e-15
        end
    end

    @testset "first-order agreement with Euler in the smooth limit" begin
        J = [0.8, -0.3, 0.2, 0.5, -0.4, 0.1, -0.6, 0.7, -0.4]
        Gamma = (0.7, -0.2, 0.4)
        errors = Float64[]
        for dt in (1e-3, 5e-4)
            pe = expint_field(; Gamma, J, integration=FLOWVPM.euler)
            pg = expint_field(; Gamma, J)
            FLOWVPM._euler(pe, dt); FLOWVPM._euler_exp(pg, dt)
            Ge = collect(FLOWVPM.get_Gamma(FLOWVPM.get_particle(pe, 1)))
            Gg = collect(FLOWVPM.get_Gamma(FLOWVPM.get_particle(pg, 1)))
            push!(errors, norm(Gg-Ge))
        end
        @test errors[2] < 0.27errors[1] # local difference is O(dt^2)
    end

    @testset "first-order convergence for a time-varying gradient" begin
        # Aligned L(t)=diag(a+b*t,0,0) has a closed-form reference. Updating
        # the frozen gradient at each left endpoint isolates its time error.
        a = 0.4; b = 0.7; tf = 0.8
        exact_log_r = a*tf + 0.5b*tf^2
        Gref = exp((1 - 3/5)*exact_log_r)
        sref = 0.1exp(-(1/5)*exact_log_r)
        errors = Float64[]
        for nsteps in (20, 40, 80)
            dt = tf/nsteps
            pf = expint_field()
            for n in 0:nsteps-1
                J = zeros(9)
                J[1] = a + b*n*dt
                FLOWVPM.set_J(pf, 1, J)
                FLOWVPM._euler_exp(pf, dt)
            end
            p = FLOWVPM.get_particle(pf, 1)
            push!(errors, hypot(FLOWVPM.get_Gamma(p)[1] - Gref,
                                FLOWVPM.get_sigma(p)[] - sref))
        end
        @test errors[2] < 0.51errors[1]
        @test errors[3] < 0.51errors[2]
        @test errors[3] > 0
    end

    @testset "coupled CoreSpreading map" begin
        nu = 1e-5; sigma0 = 0.1; dt = 1.0; dtZ = 1.0
        J = zeros(9); J[1] = 5dtZ
        visc = FLOWVPM.CoreSpreading(nu, sigma0, FLOWVPM.zeta_fmm)
        pf = expint_field(; sigma=sigma0, J, viscous=visc)
        FLOWVPM._euler_exp(pf, dt)
        ypred = sigma0^2*exp(-2dtZ) + nu/dtZ*(1-exp(-2dtZ))
        @test FLOWVPM.get_sigma(FLOWVPM.get_particle(pf, 1))[] ≈ sqrt(ypred) rtol=2e-13

        # Exact Z -> 0 limit recovers ordinary CoreSpreading.
        pf0 = expint_field(; sigma=sigma0, J=zeros(9), viscous=visc)
        FLOWVPM._euler_exp(pf0, dt)
        @test FLOWVPM.get_sigma(FLOWVPM.get_particle(pf0, 1))[] ≈
              sqrt(sigma0^2 + 2nu*dt) rtol=2e-13
    end

    @testset "unsupported f is rejected explicitly" begin
        pf = expint_field(; formulation=FLOWVPM.ReformulatedVPM{Float64}(0.1, 0.2))
        @test_throws ArgumentError FLOWVPM._euler_exp(pf, 0.1)
    end

end
