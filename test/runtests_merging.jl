using Test
using LinearAlgebra
using Random
using Statistics
import FLOWVPM

function merged_particle(pfield)
    @test FLOWVPM.get_np(pfield) == 1
    return FLOWVPM.get_particle(pfield, 1)
end

@testset "Particle merging" begin
    @testset "Circulation and centroid conservation" begin
        pfield = FLOWVPM.ParticleField(4)
        length_scale = 0.25
        gamma1 = (1.0, 0.0, 0.0)
        gamma2 = (3.0, 0.0, 0.0)
        sigma1 = 1.0
        sigma2 = 2.0
        circulation1 = length_scale * norm(gamma1)
        circulation2 = length_scale * norm(gamma2)
        expected_circulation = (sigma1 * circulation1 + sigma2 * circulation2) / (sigma1 + sigma2)

        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), gamma1, sigma1; vol=2.0, circulation=circulation1, C=(1.0, 2.0, 3.0))
        FLOWVPM.add_particle(pfield, (4.0, 0.0, 0.0), gamma2, sigma2; vol=5.0, circulation=circulation2, C=(4.0, 5.0, 6.0))

        removed = FLOWVPM.merge_particles!(pfield; r_merge=4.1, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.GAMMA_INDEX] ≈ [4.0, 0.0, 0.0]
        @test p[FLOWVPM.X_INDEX] ≈ [3.0, 0.0, 0.0]
        @test p[FLOWVPM.VOL_INDEX][] ≈ 7.0
        @test p[FLOWVPM.CIRCULATION_INDEX][] ≈ expected_circulation
        @test p[FLOWVPM.C_INDEX] ≈ [3.25, 4.25, 5.25]
    end

    # 026 §22.1 (Ryan ruling 2026-09-16): the volume-conserving merged σ
    # cbrt(σᵢ³+σⱼ³) formerly asserted here was a σ-pump (+26% per equal pair
    # regardless of overlap); replaced by second-moment matching
    #     σ_new² = ⟨σ²⟩_w + (1/3)⟨|xᵢ−x̄|²⟩_w,  w = |Γ|.
    @testset "Second-moment sigma: unequal coincident pair" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][] ≈ sqrt((1.0^2 + 2.0^2) / 2)
    end

    @testset "Second-moment sigma: coincident equal pair is pump-free" begin
        sigma = 0.7
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (1.0, 2.0, 3.0), (0.0, 2.0, 0.0), sigma)
        FLOWVPM.add_particle(pfield, (1.0, 2.0, 3.0), (0.0, 2.0, 0.0), sigma)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][] ≈ sigma atol=4eps(sigma)
    end

    @testset "Second-moment sigma: equal pair at distance d" begin
        sigma = 0.5
        d = 0.3
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), sigma)
        FLOWVPM.add_particle(pfield, (d, 0.0, 0.0), (1.0, 1.0, 0.0), sigma)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=2d, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][]^2 ≈ sigma^2 + d^2 / 12
    end

    # 026 §22.2 (Ryan ruling 2026-09-17): coincident-limit ledger lineage —
    # on merge every resolution-split ledger line becomes the |α|-weighted
    # mean over members (σ₀² and each Δσ² accumulator), and the separation
    # term (1/3)⟨|Δx|²⟩_w is credited to drvpm (grow side).
    @testset "Merge lineage: maturity neutrality of coincident equals" begin
        g = 1.4                       # both particles at growth ratio g = σ/σ₀
        sigma = 0.42
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sigma)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sigma)
        rs = FLOWVPM.enable_resolution_split!(pfield)
        rs.sigma_0[1] = sigma / g
        rs.sigma_0[2] = sigma / g

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][] / rs.sigma_0[1] ≈ g
        @test rs.dvisc[1] == 0
        @test rs.drvpm[1] ≈ 0 atol=1e-15
    end

    @testset "Merge lineage: separation term credited to drvpm" begin
        sigma = 0.5
        d = 0.2
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sigma)
        FLOWVPM.add_particle(pfield, (d, 0.0, 0.0), (1.0, 0.0, 0.0), sigma)
        rs = FLOWVPM.enable_resolution_split!(pfield)
        rs.drvpm[1] = 3e-3
        rs.drvpm[2] = 5e-3
        rs.dvisc[1] = 1e-3
        rs.dvisc[2] = 2e-3

        removed = FLOWVPM.merge_particles!(pfield; r_merge=2d, sigma_relative=false)

        @test removed == 1
        # equal weights: ⟨|Δx|²⟩_w = (d/2)² and the credit is d²/12
        @test rs.drvpm[1] ≈ (3e-3 + 5e-3) / 2 + d^2 / 12
        @test rs.dvisc[1] ≈ (1e-3 + 2e-3) / 2
        @test rs.sigma_0[1] ≈ sigma
        # the σ² gained by the merge equals the drvpm credit exactly, so the
        # ledger identity σ² ≈ σ₀² + ΣΔσ² is preserved by the merge
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][]^2 - sigma^2 ≈ d^2 / 12
    end

    @testset "Merge lineage: unequal weights use |Gamma|-weighted means" begin
        w1, w2 = 1.0, 3.0
        s01, s02 = 0.30, 0.50
        v1, v2 = 1e-3, 4e-3
        r1, r2 = -2e-3, 6e-3
        sig1, sig2 = 0.35, 0.55
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (w1, 0.0, 0.0), sig1)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (w2, 0.0, 0.0), sig2)
        rs = FLOWVPM.enable_resolution_split!(pfield)
        rs.sigma_0[1] = s01; rs.sigma_0[2] = s02
        rs.dvisc[1] = v1;    rs.dvisc[2] = v2
        rs.drvpm[1] = r1;    rs.drvpm[2] = r2

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 1
        W = w1 + w2
        @test rs.sigma_0[1]^2 ≈ (w1 * s01^2 + w2 * s02^2) / W
        @test rs.dvisc[1] ≈ (w1 * v1 + w2 * v2) / W
        @test rs.drvpm[1] ≈ (w1 * r1 + w2 * r2) / W  # coincident: zero separation term
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][]^2 ≈ (w1 * sig1^2 + w2 * sig2^2) / W
    end

    @testset "Merge lineage: stretch axis sign-aligned weighted mean" begin
        # 026 axis lineage (Ryan 2026-09-18): the direction state inherits
        # the same |α|-weighted mean, with member axes sign-aligned to the
        # running sum (anti-parallel axes flip, matching _rsplit_accumulate!).
        w1, w2 = 1.0, 3.0
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (w1, 0.0, 0.0), 0.5)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (w2, 0.0, 0.0), 0.5)
        rs = FLOWVPM.enable_resolution_split!(pfield)
        rs.axis[1, 1] = 1.0; rs.weight[1] = 1.0
        rs.axis[1, 2] = -2.0; rs.weight[2] = 2.0   # anti-parallel → flips

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 1
        W = w1 + w2
        @test rs.axis[1, 1] ≈ (w1 * 1.0 + w2 * 2.0) / W
        @test rs.axis[2, 1] == 0 && rs.axis[3, 1] == 0
        @test rs.weight[1] ≈ (w1 * 1.0 + w2 * 2.0) / W
        # coincident identical members preserve coherence exactly (= 1 here)
        @test abs(rs.axis[1, 1]) / rs.weight[1] ≈ 1.0
    end

    @testset "Transitive chains form only one pair per call" begin
        pfield = FLOWVPM.ParticleField(4)
        for x in (0.0, 0.1, 0.2)
            FLOWVPM.add_particle(pfield, (x, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        end

        removed = FLOWVPM.merge_particles!(
            pfield; r_merge=0.11, r_hash=1.0, sigma_relative=false,
        )

        @test removed == 1
        @test FLOWVPM.get_np(pfield) == 2
        @test sort([FLOWVPM.get_X(pfield, i)[1] for i in 1:2]) ≈ [0.05, 0.2]
    end

    @testset "Seed chooses nearest available partner" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.08, 0.0, 0.0), (10.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.02, 0.0, 0.0), (100.0, 0.0, 0.0), 1.0)

        removed = FLOWVPM.merge_particles!(
            pfield; r_merge=0.1, r_hash=1.0, sigma_relative=false,
        )

        @test removed == 1
        @test FLOWVPM.get_np(pfield) == 2
        @test FLOWVPM.get_Gamma(pfield, 1) ≈ [101.0, 0.0, 0.0]
        @test FLOWVPM.get_X(pfield, 1) ≈ [2.0 / 101.0, 0.0, 0.0]
        @test FLOWVPM.get_Gamma(pfield, 2) ≈ [10.0, 0.0, 0.0]
    end

    @testset "Static particles are skipped" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0; static=true)
        FLOWVPM.add_particle(pfield, (0.1, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.5, sigma_relative=false)

        @test removed == 0
        @test FLOWVPM.get_np(pfield) == 2
        @test FLOWVPM.get_static(pfield, 1) == true
    end

    @testset "No merge when particles are distant" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.5, sigma_relative=false)

        @test removed == 0
        @test FLOWVPM.get_np(pfield) == 2
    end

    @testset "Hash radius controls absolute cell size" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.2, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.5, r_hash=0.25, sigma_relative=false)

        @test removed == 1
        @test FLOWVPM.get_np(pfield) == 1
    end

    @testset "Hash radius uses mean sigma when relative" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (1.1, 0.0, 0.0), (1.0, 0.0, 0.0), 3.0)

        removed = FLOWVPM.merge_particles!(
            pfield;
            r_merge=0.5,
            r_hash=0.5,
            sigma_relative=true,
            max_sigma_ratio=4.0,
        )

        @test removed == 0
        @test FLOWVPM.get_np(pfield) == 2
    end

    @testset "Sigma ratio guard" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.01, 0.0, 0.0), (1.0, 0.0, 0.0), 3.0)

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.5, sigma_relative=true, max_sigma_ratio=2.0)

        @test removed == 0
        @test FLOWVPM.get_np(pfield) == 2
    end

    @testset "Descending removals remain consistent" begin
        pfield = FLOWVPM.ParticleField(16)

        for i in 0:5
            x = 10.0 * i
            FLOWVPM.add_particle(pfield, (x, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0; vol=1.0)
            FLOWVPM.add_particle(pfield, (x + 0.05, 0.0, 0.0), (2.0, 0.0, 0.0), 1.0; vol=2.0)
        end

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 6
        @test FLOWVPM.get_np(pfield) == 6

        gamma_total = zeros(3)
        vol_total = 0.0
        for i in 1:FLOWVPM.get_np(pfield)
            gamma_total .+= FLOWVPM.get_Gamma(pfield, i)
            vol_total += FLOWVPM.get_vol(pfield, i)[]
        end

        @test gamma_total ≈ [18.0, 0.0, 0.0]
        @test vol_total ≈ 18.0
    end

    @testset "Callback runs once per pair before removals" begin
        pfield = FLOWVPM.ParticleField(4)
        for x in (0.0, 0.1, 10.0, 10.1)
            FLOWVPM.add_particle(pfield, (x, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        end

        seen = Tuple{Int, Int, Float64}[]
        callback = function (representative)
            push!(seen, (
                representative,
                FLOWVPM.get_np(pfield),
                FLOWVPM.get_Gamma(pfield, representative)[1],
            ))
            return nothing
        end

        removed = FLOWVPM.merge_particles!(
            pfield;
            r_merge=0.2,
            r_hash=20.0,
            sigma_relative=false,
            on_representative=callback,
        )

        @test removed == 2
        @test sort(first.(seen)) == [1, 3]
        @test length(seen) == 2
        @test all(entry[2] == 4 for entry in seen)
        @test all(entry[3] ≈ 2.0 for entry in seen)
        @test FLOWVPM.get_np(pfield) == 2
    end

    @testset "run_vpm! integration" begin
        pfield = FLOWVPM.ParticleField(4; UJ=FLOWVPM.UJ_direct)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.1, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)

        calls = Int[]
        runtime = function (pf, t, dt; vprintln=nothing)
            push!(calls, FLOWVPM.get_np(pf))
            return false
        end

        FLOWVPM.run_vpm!(pfield, 0.1, 1; merge_every=1, merge_kwargs=(; r_merge=0.5, sigma_relative=false), runtime_function=runtime, verbose=false)

        @test calls == [2, 1]
        @test FLOWVPM.get_np(pfield) == 1
    end

    @testset "Merged random cube preserves target velocity" begin
        nsource = 64
        ntarget = 16
        sigma = 0.25
        r_merge = 0.30
        rng = MersenneTwister(11)
        base_gamma = [0.0, 1.0, 0.0]

        source = FLOWVPM.ParticleField(nsource; UJ=FLOWVPM.UJ_direct)
        target_particles = Tuple{NTuple{3, Float64}, NTuple{3, Float64}, Float64}[]

        for _ in 1:nsource
            x = Tuple(rand(rng, 3))
            gamma = Tuple(base_gamma .+ 0.1 .* randn(rng, 3))
            FLOWVPM.add_particle(source, x, gamma, sigma; vol=1.0)
        end

        for _ in 1:ntarget
            x = rand(rng, 3)
            x[1] += 2.0
            gamma = Tuple(base_gamma .+ 0.1 .* randn(rng, 3))
            push!(target_particles, (Tuple(x), gamma, sigma))
        end

        make_target = function ()
            target = FLOWVPM.ParticleField(ntarget; UJ=FLOWVPM.UJ_direct)
            for (x, gamma, this_sigma) in target_particles
                FLOWVPM.add_particle(target, x, gamma, this_sigma; vol=1.0)
            end
            return target
        end

        target_before = make_target()
        FLOWVPM.UJ_direct(source, target_before)
        velocity_before = [copy(FLOWVPM.get_U(target_before, i)) for i in 1:ntarget]

        removed = FLOWVPM.merge_particles!(source; r_merge, sigma_relative=false, max_sigma_ratio=Inf)

        target_after = make_target()
        FLOWVPM.UJ_direct(source, target_after)

        relative_differences = [
            norm(FLOWVPM.get_U(target_after, i) .- velocity_before[i]) / max(norm(velocity_before[i]), eps())
            for i in 1:ntarget
        ]

        merged_positions = [copy(FLOWVPM.get_X(source, i)) for i in 1:FLOWVPM.get_np(source)]
        coordinate_span = [
            maximum(position[j] for position in merged_positions) - minimum(position[j] for position in merged_positions)
            for j in 1:3
        ]

        @info "Merged random cube velocity relative differences" minimum=minimum(relative_differences) maximum=maximum(relative_differences) mean=mean(relative_differences) std=std(relative_differences)

        @test removed == nsource - FLOWVPM.get_np(source)
        @test 0 < removed <= nsource ÷ 2
        @test FLOWVPM.get_np(source) >= cld(nsource, 2)
        @test all(coordinate_span .> 0.85)
        @test maximum(relative_differences) < 0.03
    end

    @testset "Widely spread field stays O(N) in memory" begin
        # Regression: the old dense (extent/cell_size)^3 grid OOM'd when a
        # single runaway particle stretched the bounding box (e.g. 10 m extent
        # at 2.4 mm cell size -> ~1.2 TB). The sparse cell list must handle
        # this with O(N) memory and still merge the close pair correctly.
        pfield = FLOWVPM.ParticleField(10)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (0.001, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FLOWVPM.add_particle(pfield, (1.0e7, 1.0e7, 1.0e7), (1.0, 0.0, 0.0), 1.0)

        # cell_size = 0.005 over a 1e7 extent: 2e9 cells per axis, 8e27 total
        # in the old dense scheme. Warm up compilation, then bound allocation.
        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.005, sigma_relative=false)
        @test removed == 1
        @test FLOWVPM.get_np(pfield) == 2

        # Re-run on the merged field (nothing left to merge): the workspace is
        # warm, so allocation must stay far below anything extent-scaled.
        allocated = @allocated FLOWVPM.merge_particles!(pfield; r_merge=0.005, sigma_relative=false)
        @test FLOWVPM.get_np(pfield) == 2
        @test allocated < 10^6
    end
end
