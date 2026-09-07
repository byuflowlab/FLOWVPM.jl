using Test
using LinearAlgebra
import FLOWVPM
const vpmrs = FLOWVPM

# BRAINSTORM 026 Phase 2: resolution-preserving splitting
# (`ResolutionSplitState`/`ResolutionSplitOpts`, src/FLOWVPM_resolution_split.jl).
# Test numbering follows the implementation plan's Stage 8.1.

function rsplit_field(; np=3, integration=vpmrs.euler,
                        viscous=vpmrs.Inviscid(),
                        formulation=vpmrs.rVPM, transposed=true, maxp=20)
    pf = vpmrs.ParticleField(maxp; integration, formulation, viscous,
                             transposed)
    Jbase = [0.8, -0.3, 0.2, 0.5, -0.4, 0.1, -0.6, 0.7, -0.4]
    for i in 1:np
        vpmrs.add_particle(pf, (0.1i, 0.05i, 0.0), (0.7, -0.2, 0.4 + 0.1i),
                           0.1 + 0.01i)
        vpmrs.set_J(pf, i, (0.5 + 0.1i) .* Jbase)
    end
    return pf
end

"Fill slot i of rs with distinct recognizable values keyed by `key`."
function stamp_slot!(rs, i, key)
    rs.sigma_0[i] = 1.0key
    rs.axis[1, i] = 2.0key; rs.axis[2, i] = 3.0key; rs.axis[3, i] = 4.0key
    rs.weight[i] = 5.0key
    rs.exposure[i] = 6.0key
    rs.dvisc[i] = 7.0key
    rs.drvpm[i] = 8.0key
    return nothing
end

slot_values(rs, i) = (rs.sigma_0[i], rs.axis[1, i], rs.axis[2, i],
                      rs.axis[3, i], rs.weight[i], rs.exposure[i],
                      rs.dvisc[i], rs.drvpm[i])

@testset "resolution split (026 Phase 2)" begin

# --------------------------------------------------------------------------
# 8.1 test 6 — lockstep + disabled-state no-op
# --------------------------------------------------------------------------
@testset "t6: lockstep and nothing-state no-op" begin

    @testset "state === nothing: dynamics bit-identical" begin
        pf_off = rsplit_field()
        pf_on = rsplit_field()
        vpmrs.enable_resolution_split!(pf_on)
        @test pf_off.resolution_split === nothing
        @test pf_on.resolution_split isa vpmrs.ResolutionSplitState
        for _ in 1:3
            vpmrs._euler(pf_off, 1e-2)
            vpmrs._euler(pf_on, 1e-2)
        end
        # accumulation must not perturb the dynamics in any bit
        @test pf_off.particles == pf_on.particles
        # and the disabled field must work through removal unchanged
        vpmrs.remove_particle(pf_off, 2)
        @test pf_off.np == 2
    end

    @testset "enable seeds sigma_0 from current σ" begin
        pf = rsplit_field()
        rs = vpmrs.enable_resolution_split!(pf)
        for i in 1:pf.np
            @test rs.sigma_0[i] == vpmrs.get_sigma(pf, i)[]
            @test rs.weight[i] == 0 && rs.exposure[i] == 0
        end
        # idempotent: re-enabling returns the same state object
        @test vpmrs.enable_resolution_split!(pf) === rs
    end

    @testset "add_particle initializes the new slot" begin
        pf = rsplit_field()
        rs = vpmrs.enable_resolution_split!(pf)
        stamp_slot!(rs, pf.np + 1, 9.0)  # dirty the incoming slot
        vpmrs.add_particle(pf, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), 0.25)
        i = pf.np
        @test rs.sigma_0[i] == 0.25
        @test slot_values(rs, i)[2:end] == (0, 0, 0, 0, 0, 0, 0)
    end

    @testset "remove_particle swap-with-last + tail zero" begin
        pf = rsplit_field(; np=4)
        rs = vpmrs.enable_resolution_split!(pf)
        for i in 1:4
            stamp_slot!(rs, i, Float64(i))
        end
        expected = slot_values(rs, 4)
        vpmrs.remove_particle(pf, 2)
        @test pf.np == 3
        @test slot_values(rs, 2) == expected          # slot 2 ← old slot 4
        @test slot_values(rs, 1) == slot_values(rs, 1)
        @test all(iszero, slot_values(rs, 4))         # vacated tail zeroed
        # removing the last slot only zeroes the tail
        vpmrs.remove_particle(pf, 3)
        @test pf.np == 2 && all(iszero, slot_values(rs, 3))
    end
end

# --------------------------------------------------------------------------
# 8.1 test 7 — sign-invariant averaging, exposure, reset, integrator wiring
# --------------------------------------------------------------------------
@testset "t7: sign-invariant axis averaging + exposure" begin

    @testset "alternating ±S converges where raw sum cancels" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        dt = 1e-2
        raw = zeros(3)
        for k in 1:10
            s = (k % 2 == 0 ? -1.0 : 1.0) .* (2.0, 0.0, 0.0)
            raw .+= dt .* s
            vpmrs._rsplit_accumulate!(rs, 1, dt, s[1], s[2], s[3], 0.0, 0.0, 1.0)
        end
        @test norm(raw) < 1e-14                       # raw sum cancels
        axn = norm(rs.axis[:, 1])
        @test axn ≈ 10 * dt * 2.0 rtol = 1e-12        # sign-aligned sum doesn't
        @test axn / rs.weight[1] ≈ 1.0 rtol = 1e-12   # coherence → 1
    end

    @testset "incoherent samples give coherence < 1" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        dt = 1e-2
        for k in 1:10
            s = k % 2 == 0 ? (1.0, 0.0, 0.0) : (0.0, 1.0, 0.0)
            vpmrs._rsplit_accumulate!(rs, 1, dt, s[1], s[2], s[3], 0.0, 0.0, 1.0)
        end
        @test norm(rs.axis[:, 1]) / rs.weight[1] ≈ sqrt(2)/2 rtol = 1e-12
    end

    @testset "exposure: signed λ along Γ̂" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        dt, lam = 1e-2, 0.7
        G = (0.0, 0.0, 2.0)
        for _ in 1:5   # stretch along Γ̂: λ > 0
            vpmrs._rsplit_accumulate!(rs, 1, dt, lam*G[1], lam*G[2], lam*G[3],
                                      G...)
        end
        @test rs.exposure[1] ≈ 5 * dt * lam rtol = 1e-12
        for _ in 1:5   # compression along Γ̂: λ < 0 subtracts (signed)
            vpmrs._rsplit_accumulate!(rs, 1, dt, -lam*G[1], -lam*G[2], -lam*G[3],
                                      G...)
        end
        @test abs(rs.exposure[1]) < 1e-14
        # ...while the axis kept accumulating sign-invariantly the whole time
        @test norm(rs.axis[:, 1]) ≈ rs.weight[1] rtol = 1e-12
    end

    @testset "reset-on-split zeroes accumulators, restamps sigma_0" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        stamp_slot!(rs, 1, 3.0)
        vpmrs._rsplit_reset_slot!(rs, 1, 0.042)
        @test rs.sigma_0[1] == 0.042
        @test slot_values(rs, 1)[2:end] == (0, 0, 0, 0, 0, 0, 0)
    end

    @testset "integrator wiring: euler / euler_exp / rk3 final stage" begin
        # euler: one step accumulates axis+weight and mirrors dsigma2
        pf = rsplit_field()
        rs = vpmrs.enable_resolution_split!(pf)
        for _ in 1:3
            vpmrs._euler(pf, 1e-2)
        end
        st = pf.splitting_state
        for i in 1:pf.np
            @test rs.weight[i] > 0
            @test rs.dvisc[i] == 0   # Inviscid
            @test rs.drvpm[i] ≈ st.dsigma2_rvpm[i] rtol = 1e-12
        end

        # euler_exp: same wiring through the frozen-gradient path
        pf = rsplit_field(; integration=vpmrs.euler_exp)
        rs = vpmrs.enable_resolution_split!(pf)
        for _ in 1:3
            vpmrs._euler_exp(pf, 1e-2)
        end
        for i in 1:pf.np
            @test rs.weight[i] > 0
            @test rs.drvpm[i] ≈ pf.splitting_state.dsigma2_rvpm[i] rtol = 1e-12
        end

        # rk3: S sampled ONLY on the final stage (b == 8/15)
        pf = rsplit_field(; integration=vpmrs.rungekutta3)
        rs = vpmrs.enable_resolution_split!(pf)
        f = pf.formulation.f; g = pf.formulation.g
        zeta0 = pf.kernel.zeta(0.0)
        Uinf = zeros(3)
        vpmrs.update_particle_states_cpu_reformulated!(
            pf, 0.0, 1/3, 1e-2, Uinf, f, g, zeta0)
        @test all(iszero, rs.weight[1:pf.np])          # stage 1: no sample
        vpmrs.update_particle_states_cpu_reformulated!(
            pf, -5/9, 15/16, 1e-2, Uinf, f, g, zeta0)
        @test all(iszero, rs.weight[1:pf.np])          # stage 2: no sample
        vpmrs.update_particle_states_cpu_reformulated!(
            pf, -153/128, 8/15, 1e-2, Uinf, f, g, zeta0)
        @test all(>(0), rs.weight[1:pf.np])            # final stage samples
        for i in 1:pf.np
            @test rs.drvpm[i] ≈ pf.splitting_state.dsigma2_rvpm[i] rtol = 1e-12
        end
    end
end

end # outer testset
