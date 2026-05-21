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

    @testset "Sigma volume conservation" begin
        pfield = FLOWVPM.ParticleField(4)
        for sigma in (1.0, 2.0, 3.0)
            FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sigma)
        end

        removed = FLOWVPM.merge_particles!(pfield; r_merge=0.1, sigma_relative=false)

        @test removed == 2
        p = merged_particle(pfield)
        @test p[FLOWVPM.SIGMA_INDEX][] ≈ cbrt(1.0^3 + 2.0^3 + 3.0^3)
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
        @test FLOWVPM.get_np(source) < nsource ÷ 2
        @test FLOWVPM.get_np(source) > 1
        @test all(coordinate_span .> 0.85)
        @test maximum(relative_differences) < 0.03
    end
end
