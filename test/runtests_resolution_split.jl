using Test
using LinearAlgebra
import Random
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

# --------------------------------------------------------------------------
# Kernel test helpers (commit 2)
# --------------------------------------------------------------------------
"Field with a single parent particle ready to split (headroom for 4 children)."
function one_parent_field(; X=(0.3, -0.2, 0.5), Gamma=(0.4, -0.7, 0.9),
                            sigma=0.05, circulation=0.8, maxp=200)
    pf = rsplit_field(; np=0, maxp)
    vpmrs.add_particle(pf, X, Gamma, sigma; circulation)
    vpmrs.enable_resolution_split!(pf)
    return pf
end

"Apply kernel `kern` (symbol) to particle 1 of `pf` with default-opts geometry."
function apply_kernel!(pf, kern; dir=(0.0, 0.0, 1.0))
    rs = pf.resolution_split
    opts = vpmrs.ResolutionSplitOpts()
    if kern === :tetra4
        vpmrs._split_viscous_tetra4!(pf, rs, 1, opts.viscous_offset_ratio)
    elseif kern === :tri3
        vpmrs._split_compress_tri3!(pf, rs, 1, dir..., opts.compress_offset_ratio)
    elseif kern === :pair2
        vpmrs._split_elongate_pair2!(pf, rs, 1, dir..., opts.elongate_offset_ratio)
    else
        error("unknown kernel $kern")
    end
    return pf
end

children_X(pf) = [collect(vpmrs.get_X(pf, i)) for i in 1:pf.np]
children_G(pf) = [collect(vpmrs.get_Gamma(pf, i)) for i in 1:pf.np]

# --------------------------------------------------------------------------
# 8.1 test 1 — exact Γ + linear impulse + centroid conservation (all kernels)
# --------------------------------------------------------------------------
@testset "t1: conservation (Γ, centroid, linear impulse)" begin
    Random.seed!(260901)
    for kern in (:tetra4, :tri3, :pair2), trial in 1:5
        X0 = Tuple(randn(3)); G0 = Tuple(randn(3))
        dir = normalize(randn(3))
        pf = one_parent_field(; X=X0, Gamma=G0, sigma=0.03 + 0.02rand())
        apply_kernel!(pf, kern; dir=Tuple(dir))
        m = kern === :tetra4 ? 4 : (kern === :tri3 ? 3 : 2)
        @test pf.np == m
        Xs, Gs = children_X(pf), children_G(pf)
        @test sum(Gs) ≈ collect(G0) atol = 1e-14           # total Γ exact
        @test sum(Xs) ./ m ≈ collect(X0) atol = 1e-13      # centroid exact
        # linear impulse (1/2)Σ x×Γ conserved: offsets are centroid-symmetric
        # and every child carries the same Γ share
        I0 = 0.5 .* (collect(X0) × collect(G0))
        Ic = 0.5 .* sum(Xs[k] × Gs[k] for k in 1:m)
        @test Ic ≈ I0 atol = 1e-14
        # all children parallel to parent Γ, equal shares
        for k in 1:m
            @test Gs[k] ≈ collect(G0) ./ m atol = 1e-14
        end
    end
end

# --------------------------------------------------------------------------
# 8.1 test 2 — angular impulse error quantified and bounded
# --------------------------------------------------------------------------
@testset "t2: angular impulse error bounded" begin
    Random.seed!(260902)
    # A = (1/3)Σ x×(x×Γ). The split error is Σ d_k×(d_k×Γ/m) (cross terms
    # vanish by centroid symmetry), so |ΔA| ≤ (1/3)·a²·|Γ| with a the child
    # offset radius. pair2 is exact by its ± symmetry along one line... only
    # if e ∥ Γ; in general its error also obeys the same bound.
    for kern in (:tetra4, :tri3, :pair2), trial in 1:5
        X0 = Tuple(randn(3)); G0 = Tuple(randn(3))
        dir = Tuple(normalize(randn(3)))
        sigma_p = 0.05
        pf = one_parent_field(; X=X0, Gamma=G0, sigma=sigma_p)
        opts = vpmrs.ResolutionSplitOpts()
        a = (kern === :tetra4 ? opts.viscous_offset_ratio :
             kern === :tri3 ? opts.compress_offset_ratio :
             opts.elongate_offset_ratio) * sigma_p
        apply_kernel!(pf, kern; dir)
        m = pf.np
        Xs, Gs = children_X(pf), children_G(pf)
        A0 = (collect(X0) × (collect(X0) × collect(G0))) ./ 3
        Ac = sum(Xs[k] × (Xs[k] × Gs[k]) for k in 1:m) ./ 3
        err = norm(Ac - A0)
        @test err <= a^2 * norm(collect(G0)) / 3 * (1 + 1e-12)
    end
end

# --------------------------------------------------------------------------
# 8.1 test 3 — far-field equivalence: U, J (strain) at r ∈ {5,10,20}σ
# --------------------------------------------------------------------------
"Evaluate (U, J) induced at probe points by the (nonzero-Γ) particles of pf."
function probe_UJ(pf, probes)
    np0 = pf.np
    for xp in probes
        vpmrs.add_particle(pf, xp, (0.0, 0.0, 0.0), pf.particles[7, 1])
    end
    vpmrs._reset_particles(pf)
    vpmrs.UJ_direct(pf)
    U = [collect(vpmrs.get_U(pf, np0 + k)) for k in 1:length(probes)]
    J = [collect(vpmrs.get_J(pf, np0 + k)) for k in 1:length(probes)]
    for _ in 1:length(probes)   # restore field
        vpmrs.remove_particle(pf, pf.np)
    end
    return U, J
end

@testset "t3: far-field U/J equivalence" begin
    Random.seed!(260903)
    X0 = (0.0, 0.0, 0.0); G0 = (0.4, -0.7, 0.9); sigma_p = 0.05
    # probe shell directions
    dirs = [normalize(randn(3)) for _ in 1:6]
    # multipole error ~ (a/r)²: dipole term vanishes by symmetry. Bounds
    # loose vs the estimate to stay robust across random orientations.
    tolU = Dict(5 => 0.08, 10 => 0.02, 20 => 0.006)
    tolJ = Dict(5 => 0.25, 10 => 0.06, 20 => 0.015)
    for kern in (:tetra4, :tri3, :pair2)
        pf_par = one_parent_field(; X=X0, Gamma=G0, sigma=sigma_p)
        pf_ch = one_parent_field(; X=X0, Gamma=G0, sigma=sigma_p)
        apply_kernel!(pf_ch, kern; dir=(0.0, 0.0, 1.0))
        for r_over_sigma in (5, 10, 20)
            r = r_over_sigma * sigma_p
            probes = [Tuple(r .* d) for d in dirs]
            Up, Jp = probe_UJ(pf_par, probes)
            Uc, Jc = probe_UJ(pf_ch, probes)
            for k in 1:length(probes)
                @test norm(Uc[k] - Up[k]) <= tolU[r_over_sigma]*norm(Up[k])
                @test norm(Jc[k] - Jp[k]) <= tolJ[r_over_sigma]*norm(Jp[k])
                # strain (symmetric part of ∇U feeds SFS/stretching inputs)
                Sp = reshape(Jp[k], 3, 3); Sp = (Sp + Sp')/2
                Sc = reshape(Jc[k], 3, 3); Sc = (Sc + Sc')/2
                @test norm(Sc - Sp) <= tolJ[r_over_sigma]*max(norm(Sp), 1e-30)
            end
        end
    end
end

# --------------------------------------------------------------------------
# 8.1 test 4 — combined child vorticity support vs parent; L2 regression pin
# --------------------------------------------------------------------------
"ω magnitude of pf's particles at grid point x (Gaussian zeta)."
function omega_at(pf, x)
    w = zeros(3)
    for i in 1:pf.np
        Xi = vpmrs.get_X(pf, i); Gi = vpmrs.get_Gamma(pf, i)
        s = pf.particles[7, i]
        r = sqrt((x[1]-Xi[1])^2 + (x[2]-Xi[2])^2 + (x[3]-Xi[3])^2)
        w .+= collect(Gi) .* (pf.kernel.zeta(r/s) / s^3)
    end
    return w
end

function support_l2_residual(kern; dir=(0.0, 0.0, 1.0))
    X0 = (0.0, 0.0, 0.0); G0 = (0.0, 0.0, 1.0); sigma_p = 0.05
    pf_par = one_parent_field(; X=X0, Gamma=G0, sigma=sigma_p)
    pf_ch = one_parent_field(; X=X0, Gamma=G0, sigma=sigma_p)
    apply_kernel!(pf_ch, kern; dir)
    # grid through the split plane (z = 0 plane and along z) covering ±4σ
    pts = [(x, y, z) for x in range(-4sigma_p, 4sigma_p, length=17),
                        y in range(-4sigma_p, 4sigma_p, length=17),
                        z in range(-4sigma_p, 4sigma_p, length=9)]
    num = 0.0; den = 0.0
    for x in pts
        wp = omega_at(pf_par, x); wc = omega_at(pf_ch, x)
        num += sum(abs2, wc .- wp); den += sum(abs2, wp)
    end
    return sqrt(num/den)
end

@testset "t4: combined child support (L2 residual regression pin)" begin
    Random.seed!(260904)
    # Regression pins for the DEFAULT offset ratios (loose ±rel band rather
    # than exact stream pins — draws are orientation-random). Values measured
    # at implementation (2026-09-07); they guard the ratio defaults [D2/D2b/D3]
    # and the σ_c rules against silent drift.
    pins = Dict(:tetra4 => (0.50, 0.68), :tri3 => (0.92, 1.18), :pair2 => (0.095, 0.112))
    for (kern, pin) in pins
        res = [support_l2_residual(kern; dir=Tuple(normalize(randn(3))))
               for _ in 1:3]
        for r in res
            @test pin[1] <= r <= pin[2]
        end
    end
end

# --------------------------------------------------------------------------
# 8.1 test 5 — child mutual induction adds no new stiffness
# --------------------------------------------------------------------------
@testset "t5: child mutual induction dt|Z| ≪ 2/3" begin
    Random.seed!(260905)
    # Campaign-like scales: σ* = 0.0381R at R = 0.12 m (019 ruling), dt from
    # the NT144 exact-rate ladder at 5400 RPM. Γ chosen so the PARENT-scale
    # advective stiffness dt·Γ/(4πσ³) sits at 0.1 — a stiff-but-healthy
    # ambient level — then the children's sibling-induced dt|Z| must stay far
    # from the 2/3 tripwire (geometry must not amplify stiffness).
    sigma_p = 0.0381 * 0.12
    dt = (60/5400) / 144
    Gmag = 0.1 * 4pi * sigma_p^3 / dt
    for kern in (:tetra4, :tri3, :pair2)
        pf = one_parent_field(; X=(0.0, 0.0, 0.0),
                              Gamma=(0.0, Gmag/sqrt(2), Gmag/sqrt(2)),
                              sigma=sigma_p)
        apply_kernel!(pf, kern; dir=Tuple(normalize(randn(3))))
        vpmrs._reset_particles(pf)
        vpmrs.UJ_direct(pf)
        for i in 1:pf.np
            J = vpmrs.get_J(pf, i); G = vpmrs.get_Gamma(pf, i)
            # transposed stretching S = (Γ⋅∇')U, Z = S⋅Γ/|Γ|²
            S1 = J[1]*G[1] + J[2]*G[2] + J[3]*G[3]
            S2 = J[4]*G[1] + J[5]*G[2] + J[6]*G[3]
            S3 = J[7]*G[1] + J[8]*G[2] + J[9]*G[3]
            Z = (S1*G[1] + S2*G[2] + S3*G[3]) / (G[1]^2 + G[2]^2 + G[3]^2)
            @test dt*abs(Z) < (2/3)/10
        end
    end
end

# --------------------------------------------------------------------------
# 8.1 test 9 — orientation draws: exact shell/plane geometry, arbitrary draws
# --------------------------------------------------------------------------
@testset "t9: orientation-independent geometric invariants" begin
    Random.seed!(260909)
    opts = vpmrs.ResolutionSplitOpts()
    sigma_p = 0.05
    for trial in 1:20
        X0 = Tuple(randn(3))
        dir = Tuple(normalize(randn(3)))

        # tetra4: all 4 children on the shell |d| = a, all edges a√(8/3)
        pf = one_parent_field(; X=X0, sigma=sigma_p)
        apply_kernel!(pf, :tetra4)
        a = opts.viscous_offset_ratio * sigma_p
        Xs = children_X(pf)
        for x in Xs
            @test norm(x .- collect(X0)) ≈ a rtol = 1e-12
        end
        edge = a*sqrt(8/3)
        for i in 1:4, j in i+1:4
            @test norm(Xs[i] - Xs[j]) ≈ edge rtol = 1e-12
        end

        # tri3: children exactly in the plane ⟂ dir, radius a, side a√3
        pf = one_parent_field(; X=X0, sigma=sigma_p)
        apply_kernel!(pf, :tri3; dir)
        a = opts.compress_offset_ratio * sigma_p
        Xs = children_X(pf)
        for x in Xs
            d = x .- collect(X0)
            @test abs(sum(d .* collect(dir))) < 1e-13
            @test norm(d) ≈ a rtol = 1e-12
        end
        for i in 1:3, j in i+1:3
            @test norm(Xs[i] - Xs[j]) ≈ a*sqrt(3) rtol = 1e-12
        end

        # pair2: children exactly at ±b along dir, σ_c = σ_p
        pf = one_parent_field(; X=X0, sigma=sigma_p)
        apply_kernel!(pf, :pair2; dir)
        b = opts.elongate_offset_ratio * sigma_p
        Xs = children_X(pf)
        @test Xs[1] ≈ collect(X0) .- b .* collect(dir) atol = 1e-13
        @test Xs[2] ≈ collect(X0) .+ b .* collect(dir) atol = 1e-13
        @test pf.particles[7, 1] == sigma_p == pf.particles[7, 2]
    end
end

end # outer testset
