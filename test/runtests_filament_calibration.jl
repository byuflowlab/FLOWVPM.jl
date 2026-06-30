using Test
import FLOWVPM
const FC = FLOWVPM

@testset "Filament edge calibration" begin

    function _line_pf(n::Int; cap::Int=n, σ=0.5, coherent=true)
        pf = FC.ParticleField(cap)
        for i in 1:n
            FC.add_particle(pf, (Float64(i), 0.0, 0.0), (1.0, 0.0, 0.0), σ)
        end
        g = pf.filament_edge_graph
        for i in 1:n-1
            FC.add_edge!(g, i, i + 1; coherent=coherent, score=1.0)
        end
        return pf
    end

    function _graph_snapshot(pf)
        g = pf.filament_edge_graph
        return (up=copy(g.up_neighbor),
                down=copy(g.down_neighbor),
                coherent=copy(g.down_coherent),
                score=copy(g.down_score),
                degree=copy(g.degree))
    end

    @testset "empty field returns zero counts" begin
        pf = FC.ParticleField(4)
        r = FC.calibrate_filament_edges(pf)
        @test r isa FC.FilamentCalibrationReport{Float64}
        @test r.np == 0
        @test r.active_edges == 0
        @test r.coherent_edges == 0
        @test r.capped_particles == 0
        @test r.candidate_visits == 0
        @test r.split_eligible == 0
        @test r.exact_coarsen_eligible == 0
        @test r.bundle_coarsen_eligible == 0
        @test r.cross_merge_observations == 0
        @test r.validation.ok
        @test length(r.degree_histogram) == 9
    end

    @testset "open chain and closed ring counts" begin
        pf = _line_pf(5)
        r = FC.calibrate_filament_edges(pf)
        @test r.np == 5
        @test r.active_edges == 4
        @test r.coherent_edges == 4
        @test r.degree_histogram[1 + 3*0 + 1] == 1
        @test r.degree_histogram[1 + 3*1 + 1] == 3
        @test r.degree_histogram[1 + 3*1 + 0] == 1

        N = 8
        ring = FC.ParticleField(N)
        for i in 1:N
            θ = 2π * (i - 1) / N
            FC.add_particle(ring, (cos(θ), sin(θ), 0.0), (-sin(θ), cos(θ), 0.0), 0.3)
        end
        g = ring.filament_edge_graph
        for i in 1:N
            FC.add_edge!(g, i, i == N ? 1 : i + 1; coherent=true, score=0.5)
        end
        rr = FC.calibrate_filament_edges(ring)
        @test rr.active_edges == N
        @test rr.coherent_edges == N
        @test rr.degree_histogram[1 + 3*1 + 1] == N
    end

    @testset "random cloud has low conservative topology-positive observations" begin
        N = 60
        pf = FC.ParticleField(N)
        state = UInt64(0x9e3779b97f4a7c15)
        rand_next() = (state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407);
                       Float64((state >> 33) & 0xffffffff) / Float64(0xffffffff))
        for _ in 1:N
            FC.add_particle(pf,
                            (rand_next(), rand_next(), rand_next()),
                            (rand_next() - 0.5, rand_next() - 0.5, rand_next() - 0.5),
                            0.015)
        end
        r = FC.calibrate_filament_edges(pf)
        @test r.active_edges == 0
        @test r.candidate_mutual <= 3
        @test r.split_eligible == 0
        @test FC.get_np(pf) == N
    end

    @testset "dense single cell reports capped bounded work" begin
        N = 30
        cap = 8
        pf = FC.ParticleField(N)
        for i in 1:N
            θ = 2π * (i / N)
            FC.add_particle(pf, (0.002*i, 0.001*i, 0.003*i), (cos(θ), sin(θ), 0.0), 0.1)
        end
        r = FC.calibrate_filament_edges(pf; candidate_cap=cap)
        @test r.capped_particles > 0
        @test r.candidate_visits <= N * (cap + 1)
        @test r.active_edges == 0
    end

    @testset "observation-only decisions leave topology unchanged" begin
        pf = FC.ParticleField(4)
        FC.add_particle(pf, (0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.5)
        FC.add_particle(pf, (1.0, 0.0, 0.0), (0.0, 0.0, 3.0), 0.5)
        FC.add_particle(pf, (2.0, 0.0, 0.0), (0.0, 0.0, 4.0), 0.5)
        g = pf.filament_edge_graph
        FC.add_edge!(g, 1, 2; coherent=true, score=0.7)
        FC.add_edge!(g, 2, 3; coherent=true, score=0.7)
        snap = _graph_snapshot(pf)
        np0 = FC.get_np(pf)
        r = FC.calibrate_filament_edges(pf; L_min=10.0, L_max=1.0)
        @test r.split_eligible == 2
        @test r.exact_coarsen_eligible == 1
        @test FC.get_np(pf) == np0
        @test _graph_snapshot(pf) == snap

        bundle = FC.ParticleField(3)
        FC.add_particle(bundle, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5)
        FC.add_particle(bundle, (0.1, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5)
        FC.add_particle(bundle, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5)
        gb = bundle.filament_edge_graph
        FC.add_edge!(gb, 1, 3; coherent=true, score=0.5)
        FC.add_edge!(gb, 2, 3; coherent=true, score=0.5)
        snapb = _graph_snapshot(bundle)
        rb = FC.calibrate_filament_edges(bundle)
        @test rb.bundle_coarsen_eligible == 1
        @test _graph_snapshot(bundle) == snapb

        cross = FC.ParticleField(2)
        FC.add_particle(cross, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        FC.add_particle(cross, (0.2, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)
        rc = FC.calibrate_filament_edges(cross; cross_r_merge=0.5)
        @test rc.cross_merge_observations == 1
        @test FC.get_np(cross) == 2
    end

    @testset "return type and sweep" begin
        pf = _line_pf(4)
        r = @inferred FC.calibrate_filament_edges(pf)
        @test r isa FC.FilamentCalibrationReport{Float64}
        reports = FC.filament_calibration_sweep(n -> _line_pf(n), 3:4)
        @test length(reports) == 2
        @test reports[1].active_edges == 2
        @test reports[2].active_edges == 3
    end
end
