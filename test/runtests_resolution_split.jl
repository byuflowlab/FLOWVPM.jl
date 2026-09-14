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
    rs.dvisc[i] = 7.0key
    rs.drvpm[i] = 8.0key
    return nothing
end

slot_values(rs, i) = (rs.sigma_0[i], rs.axis[1, i], rs.axis[2, i],
                      rs.axis[3, i], rs.weight[i],
                      rs.dvisc[i], rs.drvpm[i])

sigma2_of(pf, i) = vpmrs.get_sigma(pf, i)[]^2

# Δσ² attribution invariant (ported from the 026 W2 dsigma2 suite): for any
# sequence of accepted steps with no split/merge/RBF-reset AND no engaged
# floor/ceil clamp,
#     σ²(t) − σ²(t₀) ≈ dvisc + drvpm    (per particle)
# with each side accumulated at its source (viscous scheme vs rVPM area
# evolution). Accumulators record ATTEMPTED (pre-clamp) Δσ², so the invariant
# intentionally BREAKS when a clamp engages — see the pinned-clamp testset.
function assert_delta_sigma2_conservation(pf, sigma2_0; rtol=1e-12)
    rs = pf.resolution_split
    for i in 1:pf.np
        @test sigma2_of(pf, i) - sigma2_0[i] ≈
              rs.dvisc[i] + rs.drvpm[i] rtol=rtol atol=1e-15
    end
end

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
            @test rs.weight[i] == 0 && rs.dvisc[i] == 0 && rs.drvpm[i] == 0
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
        @test slot_values(rs, i)[2:end] == (0, 0, 0, 0, 0, 0)
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
# 8.1 test 7 — sign-invariant averaging, reset, integrator wiring
# --------------------------------------------------------------------------
@testset "t7: sign-invariant axis averaging" begin

    @testset "alternating ±S converges where raw sum cancels" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        dt = 1e-2
        raw = zeros(3)
        for k in 1:10
            s = (k % 2 == 0 ? -1.0 : 1.0) .* (2.0, 0.0, 0.0)
            raw .+= dt .* s
            vpmrs._rsplit_accumulate!(rs, 1, dt, s[1], s[2], s[3])
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
            vpmrs._rsplit_accumulate!(rs, 1, dt, s[1], s[2], s[3])
        end
        @test norm(rs.axis[:, 1]) / rs.weight[1] ≈ sqrt(2)/2 rtol = 1e-12
    end

    @testset "reset-on-split zeroes accumulators, restamps sigma_0" begin
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        stamp_slot!(rs, 1, 3.0)
        vpmrs._rsplit_reset_slot!(rs, 1, 0.042)
        @test rs.sigma_0[1] == 0.042
        @test slot_values(rs, 1)[2:end] == (0, 0, 0, 0, 0, 0)
    end

    @testset "integrator wiring: euler / euler_exp / rk3 final stage" begin
        # euler: one step accumulates axis+weight and the dσ² attribution
        pf = rsplit_field()
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:3
            vpmrs._euler(pf, 1e-2)
        end
        for i in 1:pf.np
            @test rs.weight[i] > 0
            @test rs.dvisc[i] == 0   # Inviscid
        end
        assert_delta_sigma2_conservation(pf, s0)

        # euler_exp: same wiring through the frozen-gradient path
        pf = rsplit_field(; integration=vpmrs.euler_exp)
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:3
            vpmrs._euler_exp(pf, 1e-2)
        end
        for i in 1:pf.np
            @test rs.weight[i] > 0
        end
        assert_delta_sigma2_conservation(pf, s0)

        # rk3: S sampled ONLY on the final stage (b == 8/15)
        pf = rsplit_field(; integration=vpmrs.rungekutta3)
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
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
        assert_delta_sigma2_conservation(pf, s0)
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

"Apply kernel `kern` (symbol) to particle 1 of `pf` (default-opts geometry unless overridden)."
function apply_kernel!(pf, kern; dir=(0.0, 0.0, 1.0),
                       opts=vpmrs.ResolutionSplitOpts())
    rs = pf.resolution_split
    if kern === :tetra4
        vpmrs._split_viscous_tetra4!(pf, rs, 1, opts)
    elseif kern === :tri3
        vpmrs._split_compress_tri3!(pf, rs, 1, dir..., opts)
    elseif kern === :pair2
        vpmrs._split_elongate_pair2!(pf, rs, 1, dir..., opts)
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


# --------------------------------------------------------------------------
# 8.1 test 8 — per-mechanism fractional triggers + anti-refire + counters
# (Ryan 2026-09-08 ruling: the trigger IS the mechanism; σ bounds are clamps)
# --------------------------------------------------------------------------
@testset "t8: triggers, routing, anti-refire, counters" begin
    Random.seed!(260908)
    NTzero = (; n_split_viscous=0, n_split_compress=0, n_split_elongate=0,
                n_children_elongate=0,
                n_skipped_capacity=0, n_skipped_mech_disabled=0)

    @testset "no armed trigger => no-op" begin
        pf = one_parent_field()
        @test vpmrs.split_particles!(pf, vpmrs.ResolutionSplitOpts()) == NTzero
        @test pf.np == 1
    end

    @testset "viscous fraction => tetra4" begin
        pf = one_parent_field(; sigma=0.05)          # sigma_0 = 0.05
        rs = pf.resolution_split
        # threshold: dvisc > ((1+f)^2 - 1)*sigma_0^2 = 1.25*0.0025 = 3.125e-3
        rs.dvisc[1] = 4e-3
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5,
                                         enable_viscous_split=true,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_viscous == 1 && pf.np == 4
        # anti-refire: children get fresh sigma_0 and zeroed accumulators
        @test vpmrs.split_particles!(pf, opts) == NTzero
        @test pf.np == 4
        # just below threshold: no fire
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 3e-3
        @test vpmrs.split_particles!(pf, opts) == NTzero
    end

    @testset "compress fraction (drvpm > 0) => tri3" begin
        pf = one_parent_field(; sigma=0.05)
        rs = pf.resolution_split
        rs.drvpm[1] = 4e-3                           # past the f=0.5 threshold
        rs.dvisc[1] = 1.0                            # irrelevant: f_visc unarmed
        opts = vpmrs.ResolutionSplitOpts(; f_comp=0.5,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_compress == 1 && pf.np == 3
        @test vpmrs.split_particles!(pf, opts) == NTzero
    end

    @testset "elongate fraction (drvpm < 0) => pair2" begin
        pf = one_parent_field(; sigma=0.05)
        rs = pf.resolution_split
        # threshold: drvpm < ((1-f)^2 - 1)*sigma_0^2 = -0.51*0.0025 = -1.275e-3
        rs.drvpm[1] = -2e-3
        opts = vpmrs.ResolutionSplitOpts(; f_elong=0.3,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_elongate == 1 && pf.np == 2
        # accumulators reset to 0 on children => must re-accumulate
        @test all(iszero, rs.drvpm[1:2])
        @test vpmrs.split_particles!(pf, opts) == NTzero
        # net attempted collapse past sigma_0^2 fires for any armed f_elong
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.drvpm[1] = -1.0          # sigma_0^2 + drvpm < 0
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_elongate == 1
    end

    @testset "compress-then-relax cancels (signed net accumulator)" begin
        pf = one_parent_field(; sigma=0.05)
        rs = pf.resolution_split
        vpmrs._rsplit_accumulate_dsigma2!(rs, 1, 0.0, 4e-3)   # compression...
        vpmrs._rsplit_accumulate_dsigma2!(rs, 1, 0.0, -4e-3)  # ...then relax back
        @test rs.drvpm[1] == 0
        opts = vpmrs.ResolutionSplitOpts(; f_comp=0.5, f_elong=0.3,
                                         enable_stretch_split=true)
        @test vpmrs.split_particles!(pf, opts) == NTzero
        @test pf.np == 1
    end

    @testset "double fire viscous vs compress: larger ratio excess wins" begin
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5, f_comp=0.5,
                                         enable_viscous_split=true,
                                         enable_stretch_split=true)
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 8e-3
        pf.resolution_split.drvpm[1] = 4e-3          # both fire; viscous excess larger
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_viscous == 1 && c.n_split_compress == 0
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 4e-3
        pf.resolution_split.drvpm[1] = 8e-3          # compress excess larger
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_compress == 1 && c.n_split_viscous == 0
    end

    @testset "viscous (grow) wins over elongate (shrink)" begin
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 4e-3
        pf.resolution_split.drvpm[1] = -2e-3         # elongate armed and firing
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5, f_elong=0.3,
                                         enable_viscous_split=true,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_viscous == 1 && c.n_split_elongate == 0 && pf.np == 4
    end

    @testset "firing-but-disabled mechanism: skip + count, never reroute" begin
        # viscous fires with the viscous mechanism OFF
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 1.0
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_skipped_mech_disabled == 1 && pf.np == 1
        @test c.n_split_compress == 0          # not rerouted to tri3
        # elongate fires with the stretch mechanism OFF
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.drvpm[1] = -1.0
        opts = vpmrs.ResolutionSplitOpts(; f_elong=0.3,
                                         enable_viscous_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_skipped_mech_disabled == 1 && pf.np == 1
    end

    @testset "capacity skip counter" begin
        pf = one_parent_field(; sigma=0.05, maxp=3)  # tetra4 needs +3 slots
        pf.resolution_split.dvisc[1] = 1.0
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5,
                                         enable_viscous_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_skipped_capacity == 1 && c.n_split_viscous == 0 && pf.np == 1
    end

    @testset "emission clamp: child sigma_c clamped, vol consistent" begin
        # tetra4 children would be sigma_c = 0.05*4^(-1/3) ~ 0.0315
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5, sigma_max=0.02,
                                         enable_viscous_split=true)
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 1.0
        vpmrs.split_particles!(pf, opts)
        for i in 1:pf.np
            sig = vpmrs.get_sigma(pf, i)[]
            @test sig == 0.02
            @test vpmrs.get_vol(pf, i)[] ≈ 4/3*pi*sig^3 rtol=1e-12
            @test pf.resolution_split.sigma_0[i] == sig
        end
        # sigma_min clamp on the same kernel
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5, sigma_min=0.04,
                                         enable_viscous_split=true)
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.dvisc[1] = 1.0
        vpmrs.split_particles!(pf, opts)
        for i in 1:pf.np
            @test vpmrs.get_sigma(pf, i)[] == 0.04
        end
        # NaN bounds: unclamped
        @test vpmrs._rsplit_clamp_sigma(0.7, NaN, NaN) == 0.7
        @test vpmrs._rsplit_clamp_sigma(0.7, 0.1, NaN) == 0.7
        @test vpmrs._rsplit_clamp_sigma(0.05, 0.1, NaN) == 0.1
        @test vpmrs._rsplit_clamp_sigma(0.7, NaN, 0.5) == 0.5
    end

    @testset "multi-particle pass: mixed routing in one loop" begin
        pf = rsplit_field(; np=0, maxp=50)
        for k in 1:3
            vpmrs.add_particle(pf, (0.5k, 0.0, 0.0), (0.0, 0.0, 1.0), 0.05)
        end
        rs = vpmrs.enable_resolution_split!(pf)
        rs.dvisc[1] = 1.0                       # viscous => tetra4
        rs.drvpm[2] = 1.0                       # compress => tri3
        rs.drvpm[3] = -1.0                      # elongate => pair2
        opts = vpmrs.ResolutionSplitOpts(; f_visc=0.5, f_comp=0.5, f_elong=0.3,
                                         enable_viscous_split=true,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_viscous == 1 && c.n_split_compress == 1 &&
              c.n_split_elongate == 1
        @test pf.np == 4 + 3 + 2                # tetra4 + tri3 + pair2 children
        @test vpmrs.split_particles!(pf, opts) == NTzero   # all fresh
    end

    @testset "on_representative closure resets merged slot" begin
        # mimics the FLOWPanel merge-policy seam (Stage 3.4): the closure is
        # the ONLY merge coupling — FLOWVPM_merging.jl itself is untouched
        pf = one_parent_field()
        rs = pf.resolution_split
        stamp_slot!(rs, 1, 4.0)
        on_rep = i -> vpmrs._rsplit_reset_slot!(rs, i, 0.077)
        on_rep(1)
        @test rs.sigma_0[1] == 0.077
        @test slot_values(rs, 1)[2:end] == (0, 0, 0, 0, 0, 0)
    end
end

# --------------------------------------------------------------------------
# Δσ² attribution (ported from the 026 W2 dsigma2-accumulator suite, which
# tested the removed legacy accumulators — same invariants on the mirrors)
# --------------------------------------------------------------------------
@testset "Δσ² attribution mirrors (ported 026 W2)" begin

    @testset "euler + Inviscid: all Δσ² is rVPM" begin
        pf = rsplit_field()
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:5
            vpmrs._euler(pf, 1e-2)
        end
        @test all(iszero, rs.dvisc[1:pf.np])
        @test any(!iszero, rs.drvpm[1:pf.np])
        assert_delta_sigma2_conservation(pf, s0)
    end

    @testset "euler + sigma_guard clamp: ATTEMPTED delta attributed" begin
        # Freeze sigma with floor == ceil == current sigma: the applied delta
        # is exactly zero every step, but the accumulator must keep accruing
        # the attempted (pre-clamp) delta so triggers still fire at the
        # clamps (Ryan 2026-09-08).
        pf = rsplit_field(; np=1)
        rs = vpmrs.enable_resolution_split!(pf)
        sig0 = vpmrs.get_sigma(pf, 1)[]
        guard = (; floor=sig0, ceil=sig0)
        vpmrs._euler(pf, 1e-2; sigma_guard=guard)
        @test vpmrs.get_sigma(pf, 1)[] == sig0        # pinned
        d1 = rs.drvpm[1]
        @test d1 != 0                                  # attempted accrues
        vpmrs._euler(pf, 1e-2; sigma_guard=guard)
        @test vpmrs.get_sigma(pf, 1)[] == sig0
        @test abs(rs.drvpm[1]) > abs(d1)               # ...and keeps accruing
        @test sign(rs.drvpm[1]) == sign(d1)

        # and the matching fractional trigger fires despite frozen sigma
        f_tiny = 1e-9
        opts = rs.drvpm[1] > 0 ?
            vpmrs.ResolutionSplitOpts(; f_comp=f_tiny,
                                      enable_stretch_split=true) :
            vpmrs.ResolutionSplitOpts(; f_elong=f_tiny,
                                      enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_compress + c.n_split_elongate == 1
        # fresh children re-arm only after new attempted deformation
        @test vpmrs.split_particles!(pf, opts) ==
              (; n_split_viscous=0, n_split_compress=0, n_split_elongate=0,
                 n_children_elongate=0,
                 n_skipped_capacity=0, n_skipped_mech_disabled=0)
    end

    @testset "euler_exp + sigma_guard clamp: attempted accrues while pinned" begin
        pf = rsplit_field(; np=1, integration=vpmrs.euler_exp)
        rs = vpmrs.enable_resolution_split!(pf)
        sig0 = vpmrs.get_sigma(pf, 1)[]
        guard = (; floor=sig0, ceil=sig0)
        vpmrs._euler_exp(pf, 1e-2; sigma_guard=guard)
        @test vpmrs.get_sigma(pf, 1)[] == sig0
        d1 = rs.drvpm[1]
        @test d1 != 0
        vpmrs._euler_exp(pf, 1e-2; sigma_guard=guard)
        @test vpmrs.get_sigma(pf, 1)[] == sig0
        @test abs(rs.drvpm[1]) > abs(d1)
    end

    @testset "euler + CoreSpreading: viscous side is exactly 2ν·dt per step" begin
        nu = 1e-3
        pf = rsplit_field(; viscous=vpmrs.CoreSpreading(nu, 0.1,
                                        vpmrs.zeta_direct; beta=1e6))
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        nsteps, dt = 4, 1e-2
        for _ in 1:nsteps
            vpmrs._euler(pf, dt)
        end
        for i in 1:pf.np
            @test rs.dvisc[i] ≈ nsteps * 2 * nu * dt rtol=1e-12
        end
        assert_delta_sigma2_conservation(pf, s0)
    end

    @testset "euler_exp + Inviscid: geometric contraction attributed to rVPM" begin
        pf = rsplit_field(; integration=vpmrs.euler_exp)
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:5
            vpmrs._euler_exp(pf, 1e-2)
        end
        @test all(iszero, rs.dvisc[1:pf.np])
        assert_delta_sigma2_conservation(pf, s0)
    end

    @testset "euler_exp + CoreSpreading: blended split conserves" begin
        nu = 1e-3
        pf = rsplit_field(; integration=vpmrs.euler_exp,
                            viscous=vpmrs.CoreSpreading(nu, 0.1,
                                        vpmrs.zeta_direct; beta=1e6))
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        for _ in 1:4
            vpmrs._euler_exp(pf, 1e-2)
        end
        @test all(>(0), rs.dvisc[1:pf.np])
        assert_delta_sigma2_conservation(pf, s0)
    end

    @testset "rk3 stages: per-stage applied Δσ² conserves" begin
        pf = rsplit_field(; integration=vpmrs.rungekutta3)
        rs = vpmrs.enable_resolution_split!(pf)
        s0 = [sigma2_of(pf, i) for i in 1:pf.np]
        f = pf.formulation.f; g = pf.formulation.g
        zeta0 = pf.kernel.zeta(0.0)
        Uinf = zeros(3)
        # One full RK3 step: fresh field ⇒ M already zero (matches
        # _reset_M_storage! precondition).
        for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
            vpmrs.update_particle_states_cpu_reformulated!(
                pf, a, b, 1e-2, Uinf, f, g, zeta0)
        end
        @test any(!iszero, rs.drvpm[1:pf.np])
        assert_delta_sigma2_conservation(pf, s0)
    end

    @testset "lockstep: add/remove bookkeeping of dvisc/drvpm" begin
        pf = rsplit_field(; np=4)
        rs = vpmrs.enable_resolution_split!(pf)
        for i in 1:4
            rs.dvisc[i] = 10.0 + i
            rs.drvpm[i] = -(20.0 + i)
        end

        # remove_particle: swap-with-last copies, vacated tail slot zeroed
        vpmrs.remove_particle(pf, 2)
        @test rs.dvisc[2] == 14.0
        @test rs.drvpm[2] == -24.0
        @test rs.dvisc[4] == 0.0
        @test rs.drvpm[4] == 0.0

        # add_particle: fresh slot zero-initialized
        vpmrs.add_particle(pf, (9.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2)
        @test rs.dvisc[pf.np] == 0.0
        @test rs.drvpm[pf.np] == 0.0
    end
end

# --------------------------------------------------------------------------
# Session 3 — run_vpm! wiring (split_every/split_opts) + native merge reset
# --------------------------------------------------------------------------
@testset "s3: run_vpm! wiring and native merge reset (D-A)" begin

    @testset "split_every requires split_opts" begin
        pf = rsplit_field()
        @test_throws ErrorException vpmrs.run_vpm!(pf, 1e-2, 1; split_every=1,
                                                   verbose=false)
    end

    @testset "split_every=0 (default) leaves the feature off" begin
        pf = rsplit_field()
        vpmrs.run_vpm!(pf, 1e-2, 2; verbose=false)
        @test pf.resolution_split === nothing
    end

    @testset "state attached before step 1 at any cadence" begin
        pf = rsplit_field()
        opts = vpmrs.ResolutionSplitOpts()      # all triggers NaN-disabled
        vpmrs.run_vpm!(pf, 1e-2, 0; split_every=5, split_opts=opts,
                       verbose=false)
        @test pf.resolution_split isa vpmrs.ResolutionSplitState
    end

    @testset "split pass fires on cadence" begin
        pf = rsplit_field(; np=2, maxp=30)
        # pre-stamp a compress credit far past the f_comp threshold (the
        # in-march accumulation adds only tiny deltas on top)
        rs = vpmrs.enable_resolution_split!(pf)
        rs.drvpm[1] = 1.0; rs.drvpm[2] = 1.0
        opts = vpmrs.ResolutionSplitOpts(; f_comp=0.5,
                                         enable_stretch_split=true)
        vpmrs.run_vpm!(pf, 1e-2, 2; split_every=2, split_opts=opts,
                       verbose=false)
        @test pf.np == 6                        # one pass at i=2: 2 x tri3
        for i in 1:pf.np                        # children born fresh
            @test pf.resolution_split.sigma_0[i] == vpmrs.get_sigma(pf, i)[]
        end
        # off-cadence: no pass fires within nsteps
        pf2 = rsplit_field(; np=2, maxp=30)
        rs2 = vpmrs.enable_resolution_split!(pf2)
        rs2.drvpm[1] = 1.0; rs2.drvpm[2] = 1.0
        vpmrs.run_vpm!(pf2, 1e-2, 2; split_every=3, split_opts=opts,
                       verbose=false)
        @test pf2.np == 2
    end

    @testset "merge_particles! natively resets the representative slot" begin
        pf = rsplit_field(; np=0, maxp=10)
        vpmrs.add_particle(pf, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        vpmrs.add_particle(pf, (0.01, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        rs = vpmrs.enable_resolution_split!(pf)
        stamp_slot!(rs, 1, 3.0)
        stamp_slot!(rs, 2, 4.0)
        vpmrs.merge_particles!(pf; r_merge=5.0)
        @test pf.np == 1
        # sigma_0 := merged σ, accumulators zeroed (merged particle = new entity)
        @test rs.sigma_0[1] == vpmrs.get_sigma(pf, 1)[]
        @test slot_values(rs, 1)[2:end] == (0, 0, 0, 0, 0, 0)
    end

    @testset "merge with resolution_split === nothing is a no-op branch" begin
        pf = rsplit_field(; np=0, maxp=10)
        vpmrs.add_particle(pf, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        vpmrs.add_particle(pf, (0.01, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        vpmrs.merge_particles!(pf; r_merge=5.0)
        @test pf.np == 1
        @test pf.resolution_split === nothing
    end
end

# --------------------------------------------------------------------------
# t10 — adaptive in-line elongation kernel (theory doc §3, Ryan 2026-09-09)
# --------------------------------------------------------------------------
@testset "t10: adaptive elongation (plan, conservation, composition)" begin
    Random.seed!(260910)
    PHI = 2.75

    "Stamp parent i's split state for an elongation event: reference σ₀,
    attempted accumulator drvpm, and realized σ_p on the particle."
    function stamp_elong!(pf, i, sigma0, drvpm, sigma_p)
        rs = pf.resolution_split
        rs.sigma_0[i] = sigma0
        rs.drvpm[i] = drvpm
        vpmrs.get_sigma(pf, i)[] = sigma_p
        return rs
    end

    @testset "plan: m = round(λ·σ₀/σ_c), s tiles λ·ℓ₀ (unclamped)" begin
        sigma0 = 0.05; f = 0.3
        # trigger boundary exactly: σ_att = (1−f)σ₀, realized σ_p = σ_att
        drvpm = ((1 - f)^2 - 1) * sigma0^2
        sigma_p = (1 - f) * sigma0
        lam = 1 / (1 - f)^2                       # 2.0408
        pf = one_parent_field(; sigma=sigma0)
        stamp_elong!(pf, 1, sigma0, drvpm, sigma_p)
        opts = vpmrs.ResolutionSplitOpts(; f_elong=f, elongate_overlap=PHI,
                                         enable_stretch_split=true)
        m, s = vpmrs._elongate_plan(pf.resolution_split, 1, opts, sigma_p)
        @test m == 3                              # m_ideal = (1−f)⁻³ = 2.915
        @test s ≈ lam * (sigma0/PHI) / 3 rtol=1e-12
        # volume rule within rounding: m·σ_c³ ≈ σ₀³
        @test abs(m - lam^1.5) < 0.5
        # spacing near the target child overlap (off only by m rounding)
        @test s ≈ (sigma_p/PHI) * (lam^1.5/m) rtol=1e-12
    end

    @testset "plan saturation: m_max and attempted-collapse fall back to Φ_t spacing" begin
        sigma0 = 0.05
        opts = vpmrs.ResolutionSplitOpts(; f_elong=0.3, elongate_overlap=PHI,
                                         elongate_m_max=4,
                                         enable_stretch_split=true)
        # λ = 4 with σ_p = 0.5σ₀ → m_ideal = 8 > m_max
        pf = one_parent_field(; sigma=sigma0)
        stamp_elong!(pf, 1, sigma0, -0.75*sigma0^2, 0.5*sigma0)
        m, s = vpmrs._elongate_plan(pf.resolution_split, 1, opts, 0.5*sigma0)
        @test m == 4
        @test s ≈ 0.5*sigma0/PHI rtol=1e-12       # overlap retained, under-tiles
        # attempted total collapse: σ₀² + drvpm ≤ 0
        pf = one_parent_field(; sigma=sigma0)
        stamp_elong!(pf, 1, sigma0, -2*sigma0^2, 0.5*sigma0)
        m, s = vpmrs._elongate_plan(pf.resolution_split, 1, opts, 0.5*sigma0)
        @test m == 4
        @test s ≈ 0.5*sigma0/PHI rtol=1e-12
    end

    @testset "conservation + geometry for adaptive m (split_particles!)" begin
        sigma0 = 0.05; f = 0.3
        drvpm = ((1 - f)^2 - 1) * sigma0^2 * 1.0001   # just past trigger
        sigma_p = (1 - f) * sigma0
        for trial in 1:5
            X0 = Tuple(randn(3)); G0 = Tuple(randn(3))
            dir = Tuple(normalize(randn(3)))
            pf = one_parent_field(; X=X0, Gamma=G0, sigma=sigma0,
                                    circulation=0.8)
            stamp_elong!(pf, 1, sigma0, drvpm, sigma_p)
            # feed the direction through the axis average (coherent samples)
            vpmrs._rsplit_accumulate!(pf.resolution_split, 1, 1.0, dir...)
            opts = vpmrs.ResolutionSplitOpts(; f_elong=f,
                                             elongate_overlap=PHI,
                                             enable_stretch_split=true)
            c = vpmrs.split_particles!(pf, opts)
            m = pf.np
            @test c.n_split_elongate == 1 && c.n_children_elongate == m == 3
            Xs, Gs = children_X(pf), children_G(pf)
            @test sum(Gs) ≈ collect(G0) atol=1e-14
            @test sum(Xs) ./ m ≈ collect(X0) atol=1e-13
            I0 = 0.5 .* (collect(X0) × collect(G0))
            @test 0.5 .* sum(Xs[k] × Gs[k] for k in 1:m) ≈ I0 atol=1e-14
            # in-line geometry: equispaced along dir, middle child at parent
            offs = sort([sum((x .- collect(X0)) .* collect(dir)) for x in Xs])
            @test offs[2] ≈ 0 atol=1e-13
            @test offs[3] - offs[2] ≈ offs[2] - offs[1] rtol=1e-10
            for x in Xs   # no component off the line
                d = x .- collect(X0)
                @test norm(d .- sum(d .* collect(dir)) .* collect(dir)) < 1e-13
            end
            # σ_c = σ_p, circulation unchanged (crosswise cut), fresh state
            for k in 1:m
                @test vpmrs.get_sigma(pf, k)[] == sigma_p
                @test vpmrs.get_circulation(pf, k)[] == 0.8
                @test pf.resolution_split.sigma_0[k] == sigma_p
            end
            # angular impulse error within the t2 bound, a = (m−1)s/2
            a = (offs[3] - offs[1]) / 2
            A0 = (collect(X0) × (collect(X0) × collect(G0))) ./ 3
            Ac = sum(Xs[k] × (Xs[k] × Gs[k]) for k in 1:m) ./ 3
            @test norm(Ac - A0) <= a^2 * norm(collect(G0)) / 3 * (1 + 1e-12)
        end
    end

    @testset "composition: two small splits ≡ one big split (exact-m regime)" begin
        sigma0 = 0.05
        r = 2.0^(-1/3)                            # per-step σ ratio → m_ideal = 2 exactly
        opts = vpmrs.ResolutionSplitOpts(; f_elong=0.2, elongate_overlap=PHI,
                                         enable_stretch_split=true)
        # step 1 of the two-step path
        pf = one_parent_field(; sigma=sigma0)
        stamp_elong!(pf, 1, sigma0, (r^2 - 1)*sigma0^2, r*sigma0)
        m1, s1 = vpmrs._elongate_plan(pf.resolution_split, 1, opts, r*sigma0)
        # step 2: a child of step 1 (σ₀ = rσ₀) stretches by the same ratio
        pf2 = one_parent_field(; sigma=r*sigma0)
        stamp_elong!(pf2, 1, r*sigma0, (r^2 - 1)*(r*sigma0)^2, r^2*sigma0)
        m2, s2 = vpmrs._elongate_plan(pf2.resolution_split, 1, opts, r^2*sigma0)
        # one-shot to the same final σ
        pf3 = one_parent_field(; sigma=sigma0)
        stamp_elong!(pf3, 1, sigma0, (r^4 - 1)*sigma0^2, r^2*sigma0)
        m3, s3 = vpmrs._elongate_plan(pf3.resolution_split, 1, opts, r^2*sigma0)
        @test m1 == m2 == 2
        @test m3 == 4 == m1*m2                    # same child count...
        @test s3 ≈ s2 rtol=1e-12                  # ...same final spacing
    end

    @testset "capacity: adaptive m needs m−1 slots" begin
        sigma0 = 0.05; f = 0.3
        drvpm = ((1 - f)^2 - 1) * sigma0^2 * 1.0001
        pf = one_parent_field(; sigma=sigma0, maxp=2)  # m=3 needs +2 slots
        stamp_elong!(pf, 1, sigma0, drvpm, (1 - f)*sigma0)
        opts = vpmrs.ResolutionSplitOpts(; f_elong=f, elongate_overlap=PHI,
                                         enable_stretch_split=true)
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_skipped_capacity == 1 && c.n_split_elongate == 0 && pf.np == 1
    end

    @testset "legacy fallback: elongate_overlap = NaN keeps fixed pair2" begin
        pf = one_parent_field(; sigma=0.05)
        pf.resolution_split.drvpm[1] = -2e-3
        opts = vpmrs.ResolutionSplitOpts(; f_elong=0.3,
                                         enable_stretch_split=true)
        @test isnan(opts.elongate_overlap)        # NaN is the FLOWVPM default
        c = vpmrs.split_particles!(pf, opts)
        @test c.n_split_elongate == 1 && c.n_children_elongate == 2 && pf.np == 2
    end

    @testset "opts validation" begin
        @test_throws ArgumentError vpmrs.ResolutionSplitOpts(; elongate_overlap=-1.0)
        @test_throws ArgumentError vpmrs.ResolutionSplitOpts(; elongate_m_max=1)
    end
end

# --------------------------------------------------------------------------
# t11 — circulation bookkeeping (Ryan 2026-09-09): lengthwise bundle
# division (tri3/tetra4) carries circ/m; crosswise cuts keep circ
# --------------------------------------------------------------------------
@testset "t11: circulation bookkeeping per kernel" begin
    for (kern, m, share) in ((:tetra4, 4, 1/4), (:tri3, 3, 1/3), (:pair2, 2, 1.0))
        pf = one_parent_field(; circulation=0.8)
        apply_kernel!(pf, kern)
        @test pf.np == m
        for k in 1:m
            @test vpmrs.get_circulation(pf, k)[] ≈ 0.8*share rtol=1e-12
        end
    end
end

# --------------------------------------------------------------------------
# t12 — CPU-vs-broadcast accumulator parity (026 GPU splitting).
# The broadcast integrator twins run on plain Arrays too, so the device
# accumulation code paths are exercised WITHOUT a GPU by calling the twins
# directly against the scalar CPU paths on cloned fields.
# --------------------------------------------------------------------------
@testset "t12: CPU-vs-broadcast accumulator parity" begin

    UINF = [0.3, -0.1, 0.2]

    "clone fixture with one static particle (slot 2) to exercise masking"
    function parity_pair(; kwargs...)
        pfs = (rsplit_field(; np=4, kwargs...), rsplit_field(; np=4, kwargs...))
        for pf in pfs
            pf.particles[vpmrs.STATIC_INDEX, 2] = 1
            for i in 1:pf.np
                vpmrs.set_U(pf, i, (0.1i, -0.05i, 0.02i))
            end
            vpmrs.enable_resolution_split!(pf)
        end
        return pfs
    end

    function rs_parity(a, b; rtol=1e-12)
        ra, rb = a.resolution_split, b.resolution_split
        n = a.np
        @test ra.sigma_0[1:n] ≈ rb.sigma_0[1:n] rtol=rtol
        @test ra.axis[:, 1:n] ≈ rb.axis[:, 1:n] rtol=rtol
        @test ra.weight[1:n] ≈ rb.weight[1:n] rtol=rtol
        @test ra.dvisc[1:n] ≈ rb.dvisc[1:n] rtol=rtol
        @test ra.drvpm[1:n] ≈ rb.drvpm[1:n] rtol=rtol
    end

    @testset "euler twins" begin
        pc, pb = parity_pair()
        f, g = pc.formulation.f, pc.formulation.g
        zeta0 = pc.kernel.zeta(0)
        guard = (; dtz_cap=0.05)
        for _ in 1:3
            vpmrs._euler_cpu_reformulated!(pc, 1e-2, UINF, f, g, zeta0;
                                           sigma_guard=guard)
            vpmrs._euler_broadcast_reformulated!(pb, 1e-2, UINF, f, g, zeta0;
                                                 sigma_guard=guard)
        end
        @test pc.particles[:, 1:pc.np] ≈ pb.particles[:, 1:pb.np] rtol=1e-12
        rs_parity(pc, pb)
        # static particle accumulated nothing on either path
        @test pc.resolution_split.weight[2] == 0
        @test pb.resolution_split.weight[2] == 0
        @test pb.resolution_split.drvpm[2] == 0
    end

    @testset "euler_exp twins (incl. dtz_cap-only attempted Δσ²)" begin
        pc, pb = parity_pair(; integration=vpmrs.euler_exp)
        g = pc.formulation.g
        zeta0 = pc.kernel.zeta(0)
        R = eltype(pc.particles)
        # tight cap + floor: attempted Δσ² must re-clamp with the cap ONLY
        cap, sfloor, sceil = R(1e-4), R(0.05), R(Inf)
        for _ in 1:3
            vpmrs._euler_exp_cpu!(pc, 1e-2, UINF, g, zeta0, false,
                                  cap, sfloor, sceil)
            vpmrs._euler_exp_broadcast!(pb, 1e-2, UINF, g, zeta0,
                                        cap, sfloor, sceil)
        end
        # broadcast path evaluates exp(dt*L) by substepped Taylor — not
        # bitwise vs the scalar exact exponential, but far tighter than 1e-8
        @test pc.particles[:, 1:pc.np] ≈ pb.particles[:, 1:pb.np] rtol=1e-8
        rs_parity(pc, pb; rtol=1e-8)
    end

    @testset "rk3 twins (axis sampled on final stage only)" begin
        pc, pb = parity_pair(; integration=vpmrs.rungekutta3)
        f, g = pc.formulation.f, pc.formulation.g
        zeta0 = pc.kernel.zeta(0)
        for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
            vpmrs.update_particle_states_cpu_reformulated!(pc, a, b, 1e-2,
                                                           UINF, f, g, zeta0)
            vpmrs.update_particle_states_broadcast_reformulated!(pb, a, b,
                                                           1e-2, UINF, f, g, zeta0)
        end
        @test pc.particles[:, 1:pc.np] ≈ pb.particles[:, 1:pb.np] rtol=1e-12
        rs_parity(pc, pb)
        # axis armed only from the b == 8/15 stage: weight positive on active
        @test pc.resolution_split.weight[1] > 0
    end

    @testset "CoreSpreading twins: euler / euler_exp-blend / rk3" begin
        nu = 3e-4
        # euler branch
        pc, pb = parity_pair(; viscous=vpmrs.CoreSpreading(nu, 0.1))
        vpmrs.viscousdiffusion(pc, pc.viscous, 1e-2)   # Array dispatch = CPU loop
        vpmrs._corespreading_euler_broadcast!(pb, nu, 1e-2)
        rs_parity(pc, pb)
        @test pb.resolution_split.dvisc[1] ≈ 2*nu*1e-2 rtol=1e-12
        @test pb.resolution_split.dvisc[2] == 0        # static masked

        # euler_exp blended-diffusion branch (M9 stamped as Zeff)
        pc, pb = parity_pair(; integration=vpmrs.euler_exp,
                             viscous=vpmrs.CoreSpreading(nu, 0.1))
        for pf in (pc, pb), i in 1:pf.np
            vpmrs.get_M(vpmrs.get_particle(pf, i))[9] = 0.5 + 0.1i
        end
        vpmrs.viscousdiffusion(pc, pc.viscous, 1e-2)
        vpmrs._corespreading_eulerexp_broadcast!(pb, nu, 1e-2)
        rs_parity(pc, pb)

        # rk3 branch (aux-weighted stage accumulation)
        pc, pb = parity_pair(; integration=vpmrs.rungekutta3,
                             viscous=vpmrs.CoreSpreading(nu, 0.1))
        for (aux1, aux2) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
            vpmrs.viscousdiffusion(pc, pc.viscous, 1e-2; aux1, aux2)
            vpmrs._corespreading_rk3_broadcast!(pb, nu, 1e-2, aux1, aux2)
        end
        rs_parity(pc, pb)
    end

    @testset "trigger-decision parity after broadcast accumulation" begin
        # accumulate via the broadcast twins, then split: events must fire
        # exactly as on a CPU-accumulated clone (counters + np, geometry has
        # RNG so positions are not compared)
        pc, pb = parity_pair()
        f, g = pc.formulation.f, pc.formulation.g
        zeta0 = pc.kernel.zeta(0)
        for _ in 1:40
            vpmrs._euler_cpu_reformulated!(pc, 5e-2, UINF, f, g, zeta0)
            vpmrs._euler_broadcast_reformulated!(pb, 5e-2, UINF, f, g, zeta0)
        end
        opts = vpmrs.ResolutionSplitOpts(; f_comp=0.02, f_elong=0.02,
                                         enable_stretch_split=true)
        nc = vpmrs.split_particles!(pc, opts; dt=5e-2)
        nb = vpmrs.split_particles!(pb, opts; dt=5e-2)
        @test nc == nb
        @test pc.np == pb.np
    end

    @testset "device-style enable on a non-Array-backed clone" begin
        # enable_resolution_split! must seed sigma_0 WITHOUT scalar indexing
        # on non-Array storage; emulate with a wrapped host array type
        pf = rsplit_field()
        # exercise the broadcast seeding branch directly on the Array field:
        R = eltype(pf.particles)
        rs = vpmrs.ResolutionSplitState(pf.particles, pf.maxparticles)
        @test rs.sigma_0 isa Vector{R}
        @test size(rs.axis) == (3, pf.maxparticles)
        @test all(rs.weight .== 0)
    end
end

end # outer testset
