using Test
using LinearAlgebra
import FLOWVPM
const vpm026 = FLOWVPM

# BRAINSTORM 026 Phase 0 (W2): persistent per-particle Δσ² attribution
# accumulators in SplittingState. Invariant under test: for any sequence of
# accepted steps with no split/merge/RBF-reset,
#     σ²(t) − σ²(t₀) ≈ dsigma2_visc + dsigma2_rvpm    (per particle)
# with each side accumulated at its source (viscous scheme vs rVPM
# compression), including guard/clamp interventions (applied delta, not the
# formula).

function accum_field(; np=3, integration=vpm026.euler,
                       viscous=vpm026.Inviscid(),
                       formulation=vpm026.rVPM, transposed=true, maxp=20)
    pf = vpm026.ParticleField(maxp; integration, formulation, viscous,
                              transposed)
    Jbase = [0.8, -0.3, 0.2, 0.5, -0.4, 0.1, -0.6, 0.7, -0.4]
    for i in 1:np
        vpm026.add_particle(pf, (0.1i, 0.05i, 0.0), (0.7, -0.2, 0.4 + 0.1i),
                            0.1 + 0.01i)
        vpm026.set_J(pf, i, (0.5 + 0.1i) .* Jbase)
    end
    return pf
end

sigma2_of(pf, i) = vpm026.get_sigma(vpm026.get_particle(pf, i))[]^2

function assert_conservation(pf, sigma2_0; rtol=1e-12)
    st = pf.splitting_state
    for i in 1:pf.np
        @test sigma2_of(pf, i) - sigma2_0[i] ≈
              st.dsigma2_visc[i] + st.dsigma2_rvpm[i] rtol=rtol atol=1e-15
    end
end

@testset "dsigma2 attribution accumulators (026 W2)" begin

    @testset "euler + Inviscid: all Δσ² is rVPM" begin
        pf = accum_field()
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:5
            vpm026._euler(pf, 1e-2)
        end
        st = pf.splitting_state
        @test all(iszero, st.dsigma2_visc[1:pf.np])
        @test any(!iszero, st.dsigma2_rvpm[1:pf.np])
        assert_conservation(pf, s0)
    end

    @testset "euler + sigma_guard clamp: applied delta attributed" begin
        pf = accum_field()
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        # Tight ceil forces the clamp to engage; the accumulator must record
        # the applied (clamped) delta, keeping the invariant exact.
        guard = (; floor=0.05, ceil=0.105)
        for _ in 1:5
            vpm026._euler(pf, 1e-2; sigma_guard=guard)
        end
        assert_conservation(pf, s0)
    end

    @testset "euler + CoreSpreading: viscous side is exactly 2ν·dt per step" begin
        nu = 1e-3
        pf = accum_field(; viscous=vpm026.CoreSpreading(nu, 0.1,
                                        vpm026.zeta_direct; beta=1e6))
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        nsteps, dt = 4, 1e-2
        for _ in 1:nsteps
            vpm026._euler(pf, dt)
        end
        st = pf.splitting_state
        for i in 1:pf.np
            @test st.dsigma2_visc[i] ≈ nsteps * 2 * nu * dt rtol=1e-12
        end
        assert_conservation(pf, s0)
    end

    @testset "euler_exp + Inviscid: geometric contraction attributed to rVPM" begin
        pf = accum_field(; integration=vpm026.euler_exp)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:5
            vpm026._euler_exp(pf, 1e-2)
        end
        st = pf.splitting_state
        @test all(iszero, st.dsigma2_visc[1:pf.np])
        assert_conservation(pf, s0)
    end

    @testset "euler_exp + CoreSpreading: blended split conserves" begin
        nu = 1e-3
        pf = accum_field(; integration=vpm026.euler_exp,
                           viscous=vpm026.CoreSpreading(nu, 0.1,
                                        vpm026.zeta_direct; beta=1e6))
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:4
            vpm026._euler_exp(pf, 1e-2)
        end
        st = pf.splitting_state
        @test all(>(0), st.dsigma2_visc[1:pf.np])
        assert_conservation(pf, s0)
    end

    @testset "rk3 stages: per-stage applied Δσ² conserves" begin
        pf = accum_field(; integration=vpm026.rungekutta3)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        f = pf.formulation.f; g = pf.formulation.g
        zeta0 = pf.kernel.zeta(0.0)
        Uinf = zeros(3)
        # One full RK3 step: fresh field ⇒ M already zero (matches
        # _reset_M_storage! precondition).
        for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
            vpm026.update_particle_states_cpu_reformulated!(
                pf, a, b, 1e-2, Uinf, f, g, zeta0)
        end
        st = pf.splitting_state
        @test any(!iszero, st.dsigma2_rvpm[1:pf.np])
        assert_conservation(pf, s0)
    end

    @testset "lockstep: add/remove/split/merge bookkeeping" begin
        pf = accum_field(; np=4)
        st = pf.splitting_state
        for i in 1:4
            st.dsigma2_visc[i] = 10.0 + i
            st.dsigma2_rvpm[i] = -(20.0 + i)
        end

        # remove_particle: swap-with-last copies, vacated tail slot zeroed
        vpm026.remove_particle(pf, 2)
        @test st.dsigma2_visc[2] == 14.0
        @test st.dsigma2_rvpm[2] == -24.0
        @test st.dsigma2_visc[4] == 0.0
        @test st.dsigma2_rvpm[4] == 0.0

        # add_particle: fresh slot zero-initialized
        vpm026.add_particle(pf, (9.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2)
        @test st.dsigma2_visc[pf.np] == 0.0
        @test st.dsigma2_rvpm[pf.np] == 0.0

        # split: both children start with fresh accumulators
        st.dsigma2_visc[1] = 99.0
        st.dsigma2_rvpm[1] = -99.0
        opts = vpm026.SplitOptions(;
            trigger=vpm026.SigmaShrinkTrigger(0.5), max_fraction=1.0)
        np_before = pf.np
        vpm026._do_split!(pf, st, 1, 1.0, 0.0, 0.0, opts)
        @test pf.np == np_before + 1
        @test st.dsigma2_visc[1] == 0.0
        @test st.dsigma2_rvpm[1] == 0.0
        @test st.dsigma2_visc[pf.np] == 0.0
        @test st.dsigma2_rvpm[pf.np] == 0.0

        # merge: representative's splitting state reset wholesale,
        # sigma_0 re-anchored to the merged σ
        pfm = vpm026.ParticleField(10)
        vpm026.add_particle(pfm, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        vpm026.add_particle(pfm, (0.05, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        stm = pfm.splitting_state
        stm.dsigma2_visc[1] = 5.0; stm.dsigma2_rvpm[1] = -5.0
        stm.H_chi[1] = 3.0; stm.hold_counter[1] = 7
        removed = vpm026.merge_particles!(pfm; r_merge=0.5,
                                          sigma_relative=false)
        @test removed > 0
        @test pfm.np == 1
        @test stm.dsigma2_visc[1] == 0.0
        @test stm.dsigma2_rvpm[1] == 0.0
        @test stm.H_chi[1] == 0.0
        @test stm.hold_counter[1] == 0
        @test stm.sigma_0[1] ≈ vpm026.get_sigma(vpm026.get_particle(pfm, 1))[]
    end
end
