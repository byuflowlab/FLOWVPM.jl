using Test
import FLOWVPM

function merged_particle(pfield)
    @test FLOWVPM.get_np(pfield) == 1
    return FLOWVPM.get_particle(pfield, 1)
end

@testset "Particle merging" begin
    @testset "Circulation and centroid conservation" begin
        pfield = FLOWVPM.ParticleField(4)
        FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0; vol=2.0, C=(1.0, 2.0, 3.0))
        FLOWVPM.add_particle(pfield, (4.0, 0.0, 0.0), (3.0, 0.0, 0.0), 1.0; vol=5.0, C=(4.0, 5.0, 6.0))

        removed = FLOWVPM.merge_particles!(pfield; r_merge=4.1, sigma_relative=false)

        @test removed == 1
        p = merged_particle(pfield)
        @test p[FLOWVPM.GAMMA_INDEX] ≈ [4.0, 0.0, 0.0]
        @test p[FLOWVPM.X_INDEX] ≈ [3.0, 0.0, 0.0]
        @test p[FLOWVPM.VOL_INDEX][] ≈ 7.0
        @test p[FLOWVPM.CIRCULATION_INDEX][] ≈ 4.0
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
end
