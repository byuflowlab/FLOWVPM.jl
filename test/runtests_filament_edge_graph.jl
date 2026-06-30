using Test
import FLOWVPM
const F = FLOWVPM

@testset "FilamentEdgeGraph storage" begin

    @testset "construction: empty graph, zero adjacency" begin
        pf = F.ParticleField(8)
        g = pf.filament_edge_graph
        @test size(g.up_neighbor) == (2, 8)
        @test size(g.down_neighbor) == (2, 8)
        @test size(g.down_coherent) == (2, 8)
        @test size(g.down_score) == (2, 8)
        @test length(g.degree) == 8
        @test length(g.filament_id) == 8
        @test all(g.up_neighbor .== 0)
        @test all(g.down_neighbor .== 0)
        @test !any(g.down_coherent)
        @test all(g.down_score .== 0)
        @test all(g.degree .== 0)
        @test all(g.filament_id .== 0)
    end

    function _make_pf(n::Int; cap::Int=n)
        pf = F.ParticleField(cap)
        for i in 1:n
            F.add_particle(pf, (Float64(i), 0.0, 0.0),
                           (0.0, 0.0, 1.0), 0.1)
        end
        return pf
    end

    @testset "add_edge!: basics + per-edge state stored" begin
        pf = _make_pf(3)
        g = pf.filament_edge_graph
        @test F.add_edge!(g, 1, 2; coherent=true, score=0.75)
        @test g.down_neighbor[1, 1] == 2
        @test g.up_neighbor[1, 2] == 1
        @test g.down_coherent[1, 1] == true
        @test g.down_score[1, 1] == 0.75
        @test F.down_count(g, 1) == 1
        @test F.up_count(g, 2) == 1
        @test F.add_edge!(g, 1, 3; coherent=false, score=0.1)
        @test g.down_neighbor[2, 1] == 3
        @test F.down_count(g, 1) == 2
    end

    @testset "mirror slot lookup" begin
        pf = _make_pf(4)
        g = pf.filament_edge_graph
        F.add_edge!(g, 1, 3)
        F.add_edge!(g, 2, 3)
        @test F.find_down_slot(g, 1, 3) == 1
        @test F.find_down_slot(g, 2, 3) == 1
        @test F.find_up_slot(g, 3, 1) in (1, 2)
        @test F.find_up_slot(g, 3, 2) in (1, 2)
        @test F.find_down_slot(g, 1, 4) == 0
        @test F.find_up_slot(g, 4, 1) == 0
    end

    @testset "degree cap rejects third edge on a side" begin
        pf = _make_pf(5)
        g = pf.filament_edge_graph
        @test F.add_edge!(g, 1, 4)
        @test F.add_edge!(g, 2, 4)
        @test !F.add_edge!(g, 3, 4)
        @test F.up_count(g, 4) == 2
        @test g.up_neighbor[1, 4] == 1 && g.up_neighbor[2, 4] == 2
        @test g.down_neighbor[1, 3] == 0
    end

    @testset "self-loop and duplicate rejection" begin
        pf = _make_pf(3)
        g = pf.filament_edge_graph
        @test !F.add_edge!(g, 2, 2)
        @test F.down_count(g, 2) == 0 && F.up_count(g, 2) == 0
        @test F.add_edge!(g, 1, 2)
        @test !F.add_edge!(g, 1, 2)
        @test F.down_count(g, 1) == 1
        @test F.up_count(g, 2) == 1
    end

    @testset "remove_edge! compacts both endpoints" begin
        pf = _make_pf(4)
        g = pf.filament_edge_graph
        F.add_edge!(g, 1, 2; coherent=true, score=0.5)
        F.add_edge!(g, 1, 3; coherent=false, score=0.25)
        @test F.remove_edge!(g, 1, 2)
        @test g.down_neighbor[1, 1] == 3
        @test g.down_coherent[1, 1] == false
        @test g.down_score[1, 1] == 0.25
        @test F.down_count(g, 1) == 1
        @test g.down_neighbor[2, 1] == 0
        @test g.down_coherent[2, 1] == false
        @test g.down_score[2, 1] == 0.0
        @test g.up_neighbor[1, 2] == 0
        @test F.up_count(g, 2) == 0
        @test F.validate_filament_edges(pf).ok
        @test !F.remove_edge!(g, 1, 4)
    end

    @testset "remove_edge! compacts upstream mirrors" begin
        pf = _make_pf(4)
        g = pf.filament_edge_graph
        F.add_edge!(g, 2, 1)
        F.add_edge!(g, 3, 1)
        @test F.remove_edge!(g, 2, 1)
        @test F.up_count(g, 1) == 1
        @test g.up_neighbor[1, 1] == 3
        @test g.up_neighbor[2, 1] == 0
        @test F.down_count(g, 2) == 0
        @test F.validate_filament_edges(pf).ok
    end

    @testset "remove_edge! removing last slot leaves first slot in place" begin
        pf = _make_pf(4)
        g = pf.filament_edge_graph
        F.add_edge!(g, 1, 2; coherent=true, score=0.5)
        F.add_edge!(g, 1, 3; coherent=false, score=0.25)
        @test F.remove_edge!(g, 1, 3)
        @test F.down_count(g, 1) == 1
        @test g.down_neighbor[1, 1] == 2
        @test g.down_coherent[1, 1] == true
        @test g.down_score[1, 1] == 0.5
        @test g.down_neighbor[2, 1] == 0
        @test g.up_neighbor[1, 3] == 0
        @test F.up_count(g, 3) == 0
        @test F.validate_filament_edges(pf).ok
    end

    @testset "add_particle: new slot starts fully zero" begin
        pf = F.ParticleField(5)
        F.add_particle(pf, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        F.add_particle(pf, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        g = pf.filament_edge_graph
        F.add_edge!(g, 1, 2; coherent=true, score=0.9)
        F.add_particle(pf, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1)
        @test g.up_neighbor[1, 3] == 0 && g.up_neighbor[2, 3] == 0
        @test g.down_neighbor[1, 3] == 0 && g.down_neighbor[2, 3] == 0
        @test g.degree[3] == 0
        @test g.filament_id[3] == 0
    end

    @testset "remove_particle: mid-chain" begin
        pf = _make_pf(5)
        g = pf.filament_edge_graph
        for i in 1:4
            @test F.add_edge!(g, i, i + 1)
        end
        F.remove_particle(pf, 3)
        @test F.get_np(pf) == 4
        @test F.down_count(g, 2) == 0
        @test g.down_neighbor[1, 2] == 0
        @test F.up_count(g, 4) == 0
        @test F.down_count(g, 4) == 1
        @test g.down_neighbor[1, 1] == 2
        @test g.up_neighbor[1, 2] == 1
    end

    @testset "remove_particle: tail slot (no swap)" begin
        pf = _make_pf(4)
        g = pf.filament_edge_graph
        F.add_edge!(g, 1, 2)
        F.add_edge!(g, 3, 4)
        F.remove_particle(pf, 4)
        @test F.get_np(pf) == 3
        @test F.down_count(g, 3) == 0
        @test g.down_neighbor[1, 3] == 0
        @test g.down_neighbor[1, 1] == 2
        @test F.up_count(g, 2) == 1
    end

    @testset "remove_particle: moved-slot back-references rewritten" begin
        pf = _make_pf(5)
        g = pf.filament_edge_graph
        F.add_edge!(g, 4, 5)
        F.add_edge!(g, 5, 1)
        F.add_edge!(g, 2, 3)
        F.remove_particle(pf, 2)
        @test F.get_np(pf) == 4
        @test g.down_neighbor[1, 4] == 2
        @test F.find_up_slot(g, 2, 4) != 0
        @test g.up_neighbor[1, 1] == 2 || g.up_neighbor[2, 1] == 2
        @test F.find_down_slot(g, 2, 1) != 0
        @test F.up_count(g, 3) == 0
    end

    @testset "2-in / 2-out topology" begin
        pf = _make_pf(5)
        g = pf.filament_edge_graph
        @test F.add_edge!(g, 2, 1)
        @test F.add_edge!(g, 3, 1)
        @test F.up_count(g, 1) == 2
        @test !F.add_edge!(g, 4, 1)
        @test F.remove_edge!(g, 2, 1)
        @test F.up_count(g, 1) == 1
        @test F.add_edge!(g, 4, 1)
        @test F.up_count(g, 1) == 2
        @test F.add_edge!(g, 5, 2)
        @test F.add_edge!(g, 5, 3)
        @test F.down_count(g, 5) == 2
        @test !F.add_edge!(g, 5, 4)
    end

    @testset "clear_edges! wipes adjacency, keeps filament_id" begin
        pf = _make_pf(3)
        g = pf.filament_edge_graph
        g.filament_id[1] = 7
        F.add_edge!(g, 1, 2; coherent=true, score=0.5)
        F.add_edge!(g, 2, 3)
        F.clear_edges!(g)
        @test all(g.up_neighbor .== 0)
        @test all(g.down_neighbor .== 0)
        @test !any(g.down_coherent)
        @test all(g.down_score .== 0)
        @test all(g.degree .== 0)
        @test g.filament_id[1] == 7
    end

    @testset "FilamentEdgeGraph inference" begin

        # --- closed ring (12 particles, strength-aligned to tangent) ---
        @testset "closed ring → 12 edges, 1-in/1-out everywhere" begin
            N = 12
            R = 1.0
            σ = 0.3
            pf = F.ParticleField(N)
            for i in 1:N
                θ = 2π * (i - 1) / N
                x = (R * cos(θ), R * sin(θ), 0.0)
                # Tangent direction (right-handed, increasing θ)
                Γ = (-sin(θ), cos(θ), 0.0)
                F.add_particle(pf, x, Γ, σ)
            end
            n_added = F.infer_filament_edges!(pf)
            g = pf.filament_edge_graph
            @test n_added == N
            for i in 1:N
                @test F.up_count(g, i) == 1
                @test F.down_count(g, i) == 1
            end
            # Walk the ring forward: start at 1, follow down_neighbor, should
            # return to 1 after N steps.
            cur = 1
            for _ in 1:N
                k = g.down_neighbor[1, cur] != 0 ? 1 : 2
                cur = g.down_neighbor[k, cur]
                @test cur != 0
            end
            @test cur == 1
        end

        # --- open line: 5 particles forming a chain ---
        @testset "open line → chain edges, endpoints have one side empty" begin
            N = 5
            σ = 0.5
            pf = F.ParticleField(N)
            for i in 1:N
                F.add_particle(pf, (Float64(i), 0.0, 0.0),
                               (1.0, 0.0, 0.0), σ)
            end
            n_added = F.infer_filament_edges!(pf)
            g = pf.filament_edge_graph
            @test n_added == N - 1
            for i in 1:N
                if i == 1
                    @test F.up_count(g, i) == 0
                    @test F.down_count(g, i) == 1
                elseif i == N
                    @test F.up_count(g, i) == 1
                    @test F.down_count(g, i) == 0
                else
                    @test F.up_count(g, i) == 1
                    @test F.down_count(g, i) == 1
                end
            end
        end

        # --- random cloud: σ small enough that η > 2 for nearly all pairs ---
        @testset "random cloud → near-zero false positives" begin
            N = 60
            σ = 0.015
            seed = 0x9e3779b97f4a7c15
            state = seed
            rand_next() = (state = state * 6364136223846793005 + 1442695040888963407;
                           Float64((state >> 33) & 0xffffffff) / Float64(0xffffffff))
            pf = F.ParticleField(N)
            for _ in 1:N
                x = (rand_next(), rand_next(), rand_next())
                Γ = (rand_next() - 0.5, rand_next() - 0.5, rand_next() - 0.5)
                F.add_particle(pf, x, Γ, σ)
            end
            n_added = F.infer_filament_edges!(pf)
            # Expected pairs within η ≤ 2 are <1 at this density; even with
            # axis/projection coincidences a handful is the worst case.
            @test n_added <= 3
        end

        # --- dense single cell: visit cap fires, edge count bounded ---
        @testset "dense cell → visit cap fires, capped flag set" begin
            N = 30
            σ = 0.1
            pf = F.ParticleField(N)
            # All particles inside a single cell_size = max_eta*σ_max = 0.2 box.
            for i in 1:N
                x = (0.01 * i, 0.005 * i, 0.003 * i)
                # Random-ish axes — most pairs reject on cos θ, but every
                # candidate counts as a visit.
                θ = 2π * (i / N)
                Γ = (cos(θ), sin(θ), 0.0)
                F.add_particle(pf, x, Γ, σ)
            end
            cap = 16
            n_added = F.infer_filament_edges!(pf; candidate_cap=cap)
            ws = pf.filament_edge_workspace
            g  = pf.filament_edge_graph
            # At least the low-index particles cap (they see all the higher
            # indices in the same cell).
            @test any(ws.capped)
            # No degree-cap violation despite messy inputs.
            for i in 1:N
                @test F.up_count(g, i) <= 2
                @test F.down_count(g, i) <= 2
            end
            # Inserted edge count is bounded by 2·N (degree cap).
            @test n_added <= 2 * N
        end

        # --- re-call idempotence (non-destructive posture) ---
        @testset "re-call adds zero edges (non-destructive)" begin
            N = 12
            R = 1.0
            σ = 0.3
            pf = F.ParticleField(N)
            for i in 1:N
                θ = 2π * (i - 1) / N
                F.add_particle(pf, (R*cos(θ), R*sin(θ), 0.0),
                               (-sin(θ), cos(θ), 0.0), σ)
            end
            n1 = F.infer_filament_edges!(pf)
            n2 = F.infer_filament_edges!(pf)
            @test n1 == N
            @test n2 == 0
        end

        # --- static particles are skipped ---
        @testset "static particles excluded from inference" begin
            N = 5
            σ = 0.5
            pf = F.ParticleField(N)
            for i in 1:N
                F.add_particle(pf, (Float64(i), 0.0, 0.0),
                               (1.0, 0.0, 0.0), σ;
                               static = (i == 3))
            end
            n_added = F.infer_filament_edges!(pf)
            g = pf.filament_edge_graph
            # Static particle has no edges.
            @test F.up_count(g, 3) == 0
            @test F.down_count(g, 3) == 0
            # Adjacent pairs (1,2) and (4,5) still wire up.
            @test F.down_count(g, 1) == 1
            @test F.up_count(g, 2)   == 1
            @test F.down_count(g, 4) == 1
            @test F.up_count(g, 5)   == 1
        end

    end

    @testset "FilamentEdgeGraph validation + repair" begin

        # Build a small open chain 1→2→3→4→5 (all coherent, no bundles).
        function _chain_pf(n::Int)
            pf = _make_pf(n)
            g = pf.filament_edge_graph
            for i in 1:n-1
                F.add_edge!(g, i, i+1; coherent=true, score=1.0)
            end
            return pf
        end

        @testset "clean inputs are ok" begin
            # Empty
            pf = F.ParticleField(4)
            @test F.validate_filament_edges(pf).ok

            # Open chain
            pf = _chain_pf(5)
            r = F.validate_filament_edges(pf)
            @test r.ok
            @test r.mirror_mismatches == 0
            @test r.count_mismatches == 0
            @test r.self_loops == 0
            @test r.duplicate_locals == 0
            @test r.invalid_endpoints == 0
            @test r.stale_inactive_slots == 0

            # Closed ring (5 particles, 5 edges)
            pf = _make_pf(5)
            g = pf.filament_edge_graph
            for i in 1:5
                @test F.add_edge!(g, i, mod1(i+1, 5); coherent=true, score=1.0)
            end
            @test F.validate_filament_edges(pf).ok

            # 2→1 bundle: 1→3, 2→3.
            pf = _make_pf(3)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 3; coherent=true, score=1.0)
            F.add_edge!(g, 2, 3; coherent=true, score=1.0)
            @test F.validate_filament_edges(pf).ok
        end

        @testset "mirror_mismatch detection + repair" begin
            pf = _chain_pf(5)
            g = pf.filament_edge_graph
            # Tear the mirror: clear up_neighbor[1, 3] (was 2). Down side still
            # claims 2→3, but 3's up slot no longer points back to 2.
            @test g.up_neighbor[1, 3] == 2
            g.up_neighbor[1, 3] = 0
            F.set_up_count!(g, 3, 0)

            r = F.validate_filament_edges(pf)
            @test !r.ok
            @test r.mirror_mismatches == 1

            # Repair clears down_coherent on slot owning the torn edge.
            @test g.down_coherent[1, 2] == true
            r2 = F.repair_filament_edges!(pf)
            @test r2.mirror_mismatches == 1
            @test g.down_coherent[1, 2] == false
            # Structural tear remains (repair is conservative); coherence-side
            # is the only thing self-healed.
            r3 = F.repair_filament_edges!(pf)
            @test r3.mirror_mismatches == 1
        end

        @testset "self_loop detection + repair" begin
            pf = _chain_pf(3)
            g = pf.filament_edge_graph
            # Force a self-loop in slot 2 of particle 2.
            g.down_neighbor[2, 2] = 2
            F.set_down_count!(g, 2, 2)

            r = F.validate_filament_edges(pf)
            @test r.self_loops == 1

            @test g.down_coherent[2, 2] == false  # was never set true
            g.down_coherent[2, 2] = true
            F.repair_filament_edges!(pf)
            @test g.down_coherent[2, 2] == false
        end

        @testset "duplicate_local detection + repair" begin
            pf = _make_pf(3)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=true, score=1.0)
            # Force slot 2 to also point at 2.
            g.down_neighbor[2, 1] = 2
            g.down_coherent[2, 1] = true
            F.set_down_count!(g, 1, 2)

            r = F.validate_filament_edges(pf)
            @test r.duplicate_locals == 1

            F.repair_filament_edges!(pf)
            @test g.down_coherent[2, 1] == false
        end

        @testset "invalid_endpoint detection + repair" begin
            pf = _chain_pf(3)
            g = pf.filament_edge_graph
            # Point slot 1 of particle 1 at an out-of-range index.
            g.down_neighbor[1, 1] = 99
            g.down_coherent[1, 1] = true

            r = F.validate_filament_edges(pf)
            @test r.invalid_endpoints == 1

            F.repair_filament_edges!(pf)
            @test g.down_coherent[1, 1] == false
        end

        @testset "count_mismatch detection" begin
            pf = _chain_pf(3)
            g = pf.filament_edge_graph
            # Inflate down_count of particle 1 from 1 to 2; slot 2 is 0.
            F.set_down_count!(g, 1, 2)
            r = F.validate_filament_edges(pf)
            @test r.count_mismatches >= 1
        end

        @testset "stale_inactive_slot detection + repair" begin
            pf = _chain_pf(3)
            g = pf.filament_edge_graph
            # Particle 1 has down_count == 1 (slot 1 → 2). Stuff slot 2 with
            # leftover data without bumping the count.
            g.down_neighbor[2, 1] = 3
            g.down_coherent[2, 1] = true
            g.down_score[2, 1]    = 0.5

            r = F.validate_filament_edges(pf)
            @test r.stale_inactive_slots == 1

            F.repair_filament_edges!(pf)
            @test g.down_coherent[2, 1] == false
        end

        @testset "repair idempotence on clean inputs" begin
            pf = _chain_pf(6)
            r1 = F.repair_filament_edges!(pf)
            @test r1.ok
            r2 = F.repair_filament_edges!(pf)
            @test r2.ok
        end

        @testset "validate is allocation-free on clean inputs" begin
            pf = _chain_pf(200)
            F.validate_filament_edges(pf)  # warmup
            allocs = @allocated F.validate_filament_edges(pf)
            @test allocs == 0
        end
    end

    @testset "FilamentEdgeGraph refine (edge-driven split)" begin

        # Helper: build a particle field with given positions, Γ vectors,
        # and σ; then link consecutive particles into an open chain.
        function _line_pf(xs::Vector{NTuple{3,Float64}},
                         gs::Vector{NTuple{3,Float64}},
                         σs::Vector{Float64};
                         cap::Int = length(xs))
            pf = F.ParticleField(cap)
            for k in eachindex(xs)
                F.add_particle(pf, xs[k], gs[k], σs[k])
            end
            g = pf.filament_edge_graph
            for k in 1:length(xs)-1
                F.add_edge!(g, k, k+1; coherent=true, score=1.0)
            end
            return pf
        end

        function impulse(pf, idxs)
            Ix = 0.0; Iy = 0.0; Iz = 0.0
            for i in idxs
                xx = pf.particles[F.X_INDEX.start,     i]
                xy = pf.particles[F.X_INDEX.start + 1, i]
                xz = pf.particles[F.X_INDEX.start + 2, i]
                gx = pf.particles[F.GAMMA_INDEX.start,     i]
                gy = pf.particles[F.GAMMA_INDEX.start + 1, i]
                gz = pf.particles[F.GAMMA_INDEX.start + 2, i]
                Ix += xy*gz - xz*gy
                Iy += xz*gx - xx*gz
                Iz += xx*gy - xy*gx
            end
            return (0.5*Ix, 0.5*Iy, 0.5*Iz)
        end

        function angular_impulse(pf, idxs)
            Ax = 0.0; Ay = 0.0; Az = 0.0
            for i in idxs
                xx = pf.particles[F.X_INDEX.start,     i]
                xy = pf.particles[F.X_INDEX.start + 1, i]
                xz = pf.particles[F.X_INDEX.start + 2, i]
                gx = pf.particles[F.GAMMA_INDEX.start,     i]
                gy = pf.particles[F.GAMMA_INDEX.start + 1, i]
                gz = pf.particles[F.GAMMA_INDEX.start + 2, i]
                cx = xy*gz - xz*gy
                cy = xz*gx - xx*gz
                cz = xx*gy - xy*gx
                Ax += xy*cz - xz*cy
                Ay += xz*cx - xx*cz
                Az += xx*cy - xy*cx
            end
            return (-Ax/3, -Ay/3, -Az/3)
        end

        @testset "trivial: empty graph returns 0" begin
            pf = F.ParticleField(4)
            n = F.refine_filament_edges!(pf)
            @test n == 0
            @test pf.np == 0
        end

        @testset "trivial: short edges → no splits" begin
            # Edge length 0.1, σ̄ = 0.5, L_max = 1.5 → thresh 0.75. No split.
            pf = _line_pf([(0.0,0.0,0.0), (0.1,0.0,0.0)],
                          [(0.0,0.0,1.0), (0.0,0.0,1.0)],
                          [0.5, 0.5])
            n = F.refine_filament_edges!(pf)
            @test n == 0
            @test pf.np == 2
        end

        @testset "single-edge split: geometry + topology" begin
            # Edge length 2.0, σ̄ = 0.5 → thresh 0.75, triggers.
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.0,0.0,3.0), (0.0,0.0,6.0)],
                          [0.5, 0.5]; cap=3)
            g = pf.filament_edge_graph
            n = F.refine_filament_edges!(pf)
            @test n == 1
            @test pf.np == 3
            m = 3
            # Geometry
            @test pf.particles[F.X_INDEX.start,     m] ≈ 1.0
            @test pf.particles[F.X_INDEX.start + 1, m] ≈ 0.0
            @test pf.particles[F.X_INDEX.start + 2, m] ≈ 0.0
            @test pf.particles[F.SIGMA_INDEX, m] ≈ 0.5
            # Topology: 1 → m → 2
            @test g.down_neighbor[1, 1] == m
            @test g.up_neighbor[1, m] == 1
            @test g.down_neighbor[1, m] == 2
            @test g.up_neighbor[1, 2] == m
            # Degrees still 1-in/1-out where applicable
            @test F.down_count(g, 1) == 1
            @test F.up_count(g, 1) == 0
            @test F.down_count(g, m) == 1
            @test F.up_count(g, m) == 1
            @test F.down_count(g, 2) == 0
            @test F.up_count(g, 2) == 1
        end

        @testset "Γ conservation under split" begin
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.1, 0.2, 3.0), (-0.05, 0.4, 6.0)],
                          [0.5, 0.5]; cap=3)
            # Pre-split totals
            gp = (pf.particles[F.GAMMA_INDEX.start,     1],
                  pf.particles[F.GAMMA_INDEX.start + 1, 1],
                  pf.particles[F.GAMMA_INDEX.start + 2, 1])
            gq = (pf.particles[F.GAMMA_INDEX.start,     2],
                  pf.particles[F.GAMMA_INDEX.start + 1, 2],
                  pf.particles[F.GAMMA_INDEX.start + 2, 2])
            tot_pre = gp .+ gq

            F.refine_filament_edges!(pf)

            gp_new = (pf.particles[F.GAMMA_INDEX.start,     1],
                      pf.particles[F.GAMMA_INDEX.start + 1, 1],
                      pf.particles[F.GAMMA_INDEX.start + 2, 1])
            gq_new = (pf.particles[F.GAMMA_INDEX.start,     2],
                      pf.particles[F.GAMMA_INDEX.start + 1, 2],
                      pf.particles[F.GAMMA_INDEX.start + 2, 2])
            gm = (pf.particles[F.GAMMA_INDEX.start,     3],
                  pf.particles[F.GAMMA_INDEX.start + 1, 3],
                  pf.particles[F.GAMMA_INDEX.start + 2, 3])
            tot_post = gp_new .+ gq_new .+ gm

            for c in 1:3
                @test tot_post[c] ≈ tot_pre[c] atol=1e-12
                # Specific scheme: Γ_p' = (2/3) Γ_p, Γ_m = (Γ_p + Γ_q)/3
                @test gp_new[c] ≈ (2/3) * gp[c] atol=1e-12
                @test gq_new[c] ≈ (2/3) * gq[c] atol=1e-12
                @test gm[c]     ≈ (gp[c] + gq[c]) / 3 atol=1e-12
            end
        end

        @testset "impulse exactly conserved when Γ ∥ tangent" begin
            # Γ vectors aligned with the edge tangent ((x_q − x_p) along x).
            pf = _line_pf([(0.0, 0.5, -0.3), (2.0, 0.5, -0.3)],
                          [(1.5, 0.0, 0.0), (2.0, 0.0, 0.0)],
                          [0.5, 0.5]; cap=3)
            I_pre = impulse(pf, 1:pf.np)
            F.refine_filament_edges!(pf)
            I_post = impulse(pf, 1:pf.np)
            @test I_post[1] ≈ I_pre[1] atol=1e-12
            @test I_post[2] ≈ I_pre[2] atol=1e-12
            @test I_post[3] ≈ I_pre[3] atol=1e-12
        end

        @testset "angular impulse is not exact for offset unequal tangent-aligned Γ" begin
            pf = _line_pf([(1.0, 1.0, 0.0), (3.0, 1.0, 0.0)],
                          [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                          [0.5, 0.5]; cap=3)
            I_pre = impulse(pf, 1:pf.np)
            A_pre = angular_impulse(pf, 1:pf.np)

            n = F.refine_filament_edges!(pf)
            @test n == 1

            I_post = impulse(pf, 1:pf.np)
            A_post = angular_impulse(pf, 1:pf.np)
            @test I_post[1] ≈ I_pre[1] atol=1e-12
            @test I_post[2] ≈ I_pre[2] atol=1e-12
            @test I_post[3] ≈ I_pre[3] atol=1e-12

            dA = sqrt((A_post[1] - A_pre[1])^2 +
                      (A_post[2] - A_pre[2])^2 +
                      (A_post[3] - A_pre[3])^2)
            @test dA > 1e-6
        end

        @testset "vol and circulation of inserted particle" begin
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.0,0.0,1.0), (0.0,0.0,1.0)],
                          [0.4, 0.6]; cap=3)
            F.refine_filament_edges!(pf)
            σm = pf.particles[F.SIGMA_INDEX, 3]
            @test σm ≈ 0.5
            @test pf.particles[F.VOL_INDEX, 3] ≈ (4/3)*π*0.5^3 atol=1e-12
        end

        @testset "coherent flag and score preserved across split" begin
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.0,0.0,1.0), (0.0,0.0,1.0)],
                          [0.5, 0.5]; cap=3)
            g = pf.filament_edge_graph
            # Set a distinctive score on the only edge.
            g.down_score[1, 1] = 0.42
            F.refine_filament_edges!(pf)
            # p → m and m → q both inherit coherent=true and score=0.42.
            kpm = F.find_down_slot(g, 1, 3)
            kmq = F.find_down_slot(g, 3, 2)
            @test kpm != 0 && kmq != 0
            @test g.down_coherent[kpm, 1] == true
            @test g.down_coherent[kmq, 3] == true
            @test g.down_score[kpm, 1] ≈ 0.42
            @test g.down_score[kmq, 3] ≈ 0.42
        end

        @testset "only_coherent=true skips non-coherent edges" begin
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.0,0.0,1.0), (0.0,0.0,1.0)],
                          [0.5, 0.5]; cap=3)
            g = pf.filament_edge_graph
            g.down_coherent[1, 1] = false
            n_def = F.refine_filament_edges!(pf)
            @test n_def == 0
            @test pf.np == 2
            # With only_coherent=false the same edge splits.
            n_all = F.refine_filament_edges!(pf; only_coherent=false)
            @test n_all == 1
            @test pf.np == 3
        end

        @testset "closed ring N=12: every edge splits, ring closes" begin
            R0 = 1.0
            N = 12
            xs = [(R0*cos(2π*k/N), R0*sin(2π*k/N), 0.0) for k in 0:N-1]
            gs = [(0.0, 0.0, 1.0) for _ in 1:N]
            σs = fill(0.05, N)  # σ̄ = 0.05; chord ≈ 0.518 >> 1.5·0.05
            pf = F.ParticleField(2N)
            for k in 1:N
                F.add_particle(pf, xs[k], gs[k], σs[k])
            end
            g = pf.filament_edge_graph
            for k in 1:N
                kn = (k == N) ? 1 : k+1
                F.add_edge!(g, k, kn; coherent=true, score=1.0)
            end
            n = F.refine_filament_edges!(pf)
            @test n == N
            @test pf.np == 2N
            # Validate: closed ring, all 1-in/1-out, walk closes in 2N hops.
            r = F.validate_filament_edges(pf)
            @test r.ok
            visited = 0
            cur = 1
            for _ in 1:2N
                cur = g.down_neighbor[1, cur]
                visited += 1
                cur == 1 && break
            end
            @test cur == 1
            @test visited == 2N
        end

        @testset "max_splits cap honored" begin
            R0 = 1.0; N = 12
            xs = [(R0*cos(2π*k/N), R0*sin(2π*k/N), 0.0) for k in 0:N-1]
            gs = [(0.0, 0.0, 1.0) for _ in 1:N]
            σs = fill(0.05, N)
            pf = F.ParticleField(2N)
            for k in 1:N
                F.add_particle(pf, xs[k], gs[k], σs[k])
            end
            g = pf.filament_edge_graph
            for k in 1:N
                kn = (k == N) ? 1 : k+1
                F.add_edge!(g, k, kn; coherent=true, score=1.0)
            end
            n = F.refine_filament_edges!(pf; max_splits=3)
            @test n == 3
            @test pf.np == N + 3
        end

        @testset "capacity guard: no append possible → no splits, intact graph" begin
            # cap = np, so add_particle cannot append.
            pf = _line_pf([(0.0,0.0,0.0), (2.0,0.0,0.0)],
                          [(0.0,0.0,1.0), (0.0,0.0,1.0)],
                          [0.5, 0.5]; cap=2)
            g = pf.filament_edge_graph
            n = F.refine_filament_edges!(pf)
            @test n == 0
            @test pf.np == 2
            @test g.down_neighbor[1, 1] == 2
            @test g.up_neighbor[1, 2] == 1
            # Validator: graph is still consistent.
            r = F.validate_filament_edges(pf)
            @test r.ok
        end

        @testset "no-op call is allocation-free" begin
            # Chain with all short edges → no splits performed.
            pf = _line_pf([(0.0,0.0,0.0), (0.1,0.0,0.0), (0.2,0.0,0.0),
                          (0.3,0.0,0.0), (0.4,0.0,0.0)],
                         [(0.0,0.0,1.0) for _ in 1:5],
                         fill(0.5, 5); cap=20)
            F.refine_filament_edges!(pf)  # warmup
            allocs = @allocated F.refine_filament_edges!(pf)
            @test allocs == 0
        end
    end

    @testset "FilamentEdgeGraph coarsen (exact inverse split)" begin

        function _inverse_pf(; xs=[(0.0,0.0,0.0), (1.0,0.0,0.0), (2.0,0.0,0.0)],
                             gs=[(0.0,0.0,2.0), (0.0,0.0,3.0), (0.0,0.0,4.0)],
                             σs=[0.5, 0.5, 0.5],
                             coherent=true,
                             score=0.7,
                             cap=max(length(xs), 8))
            pf = F.ParticleField(cap)
            for k in eachindex(xs)
                F.add_particle(pf, xs[k], gs[k], σs[k])
            end
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=coherent, score=score)
            F.add_edge!(g, 2, 3; coherent=coherent, score=score)
            return pf
        end

        function _gamma_tuple(pf, i)
            return (pf.particles[F.GAMMA_INDEX.start,     i],
                    pf.particles[F.GAMMA_INDEX.start + 1, i],
                    pf.particles[F.GAMMA_INDEX.start + 2, i])
        end

        function _local_gamma_sum(pf, idxs)
            sx = 0.0; sy = 0.0; sz = 0.0
            for i in idxs
                g = _gamma_tuple(pf, i)
                sx += g[1]; sy += g[2]; sz += g[3]
            end
            return (sx, sy, sz)
        end

        @testset "trivial: empty graph returns 0" begin
            pf = F.ParticleField(4)
            @test F.coarsen_filament_edges!(pf) == 0
            @test pf.np == 0
        end

        @testset "split then coarsen restores local state" begin
            pf = F.ParticleField(3)
            F.add_particle(pf, (0.0,0.0,0.0), (0.1,0.2,3.0), 0.5; vol=9.0, circulation=2.0)
            F.add_particle(pf, (2.0,0.0,0.0), (-0.1,0.4,6.0), 0.5; vol=8.0, circulation=4.0)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=true, score=0.42)
            pre = copy(pf.particles[:, 1:2])

            @test F.refine_filament_edges!(pf) == 1
            @test pf.np == 3
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 1

            @test pf.np == 2
            @test pf.particles[:, 1:2] ≈ pre atol=1e-12
            @test F.find_down_slot(g, 1, 2) != 0
            k = F.find_down_slot(g, 1, 2)
            @test g.down_coherent[k, 1]
            @test g.down_score[k, 1] ≈ 0.42
            @test F.validate_filament_edges(pf).ok
        end

        @testset "coarsen conserves local Γ" begin
            pf = _inverse_pf()
            total_pre = _local_gamma_sum(pf, 1:3)
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 1
            total_post = _local_gamma_sum(pf, 1:2)
            for c in 1:3
                @test total_post[c] ≈ total_pre[c] atol=1e-12
            end
            @test _gamma_tuple(pf, 1)[3] ≈ 3.0 atol=1e-12
            @test _gamma_tuple(pf, 2)[3] ≈ 6.0 atol=1e-12
        end

        @testset "rejects non-inverse fingerprints" begin
            pf = _inverse_pf()
            pf.particles[F.X_INDEX.start + 1, 2] = 0.1
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0

            pf = _inverse_pf()
            pf.particles[F.SIGMA_INDEX, 2] = 0.6
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0

            pf = _inverse_pf()
            pf.particles[F.GAMMA_INDEX.start + 2, 2] = 3.1
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0

            pf = _inverse_pf()
            g = pf.filament_edge_graph
            g.down_score[F.find_down_slot(g, 2, 3), 2] = 0.8
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0
        end

        @testset "rejects ineligible topology and edge metadata" begin
            pf = _inverse_pf(cap=4)
            F.add_particle(pf, (-1.0,0.0,0.0), (0.0,0.0,1.0), 0.5)
            F.add_edge!(pf.filament_edge_graph, 4, 2; coherent=true, score=0.7)
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0

            pf = _inverse_pf(coherent=false)
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0
            @test F.coarsen_filament_edges!(pf; L_min=10.0, only_coherent=false) == 1

            pf = _inverse_pf()
            F.add_edge!(pf.filament_edge_graph, 1, 3; coherent=true, score=0.7)
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 0

            pf = _inverse_pf()
            @test F.coarsen_filament_edges!(pf; L_min=1.0) == 0
        end

        @testset "max_coarsens cap honored" begin
            pf = F.ParticleField(8)
            for x in (0.0, 1.0, 2.0, 4.0, 5.0, 6.0)
                F.add_particle(pf, (x,0.0,0.0), (0.0,0.0,x + 2.0), 0.5)
            end
            pf.particles[F.GAMMA_INDEX.start + 2, 2] = 3.0
            pf.particles[F.GAMMA_INDEX.start + 2, 5] = 6.0
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=true, score=0.5)
            F.add_edge!(g, 2, 3; coherent=true, score=0.5)
            F.add_edge!(g, 4, 5; coherent=true, score=0.6)
            F.add_edge!(g, 5, 6; coherent=true, score=0.6)
            @test F.coarsen_filament_edges!(pf; L_min=10.0, max_coarsens=1) == 1
            @test pf.np == 5
            @test F.validate_filament_edges(pf).ok
        end

        @testset "closed ring round trip" begin
            R0 = 1.0
            N = 8
            pf = F.ParticleField(2N)
            for k in 0:N-1
                F.add_particle(pf, (R0*cos(2π*k/N), R0*sin(2π*k/N), 0.0),
                               (0.0,0.0,1.0), 0.05)
            end
            g = pf.filament_edge_graph
            for k in 1:N
                F.add_edge!(g, k, k == N ? 1 : k+1; coherent=true, score=1.0)
            end
            @test F.refine_filament_edges!(pf) == N
            @test F.coarsen_filament_edges!(pf; L_min=100.0) == N
            @test pf.np == N
            @test F.validate_filament_edges(pf).ok
        end

        @testset "graph validates after swap-with-last midpoint removal" begin
            pf = _inverse_pf(cap=4)
            F.add_particle(pf, (3.0,0.0,0.0), (0.0,1.0,0.0), 0.5)
            F.add_edge!(pf.filament_edge_graph, 4, 1; coherent=true, score=0.2)
            @test F.coarsen_filament_edges!(pf; L_min=10.0) == 1
            @test pf.np == 3
            @test F.validate_filament_edges(pf).ok
            @test F.find_down_slot(pf.filament_edge_graph, 2, 1) != 0
        end

        @testset "no-op call is allocation-free" begin
            pf = F.ParticleField(4)
            F.coarsen_filament_edges!(pf)  # warmup
            allocs = @allocated F.coarsen_filament_edges!(pf)
            @test allocs == 0
        end
    end

    @testset "FilamentEdgeGraph refine orchestration" begin

        function _orchestration_line_pf(; cap=3, coherent=true)
            pf = F.ParticleField(cap)
            F.add_particle(pf, (0.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            F.add_particle(pf, (1.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            F.add_edge!(pf.filament_edge_graph, 1, 2; coherent=coherent, score=1.0)
            return pf
        end

        function _orchestration_inverse_pf()
            pf = F.ParticleField(4)
            F.add_particle(pf, (0.0,0.0,0.0), (0.0,0.0,2.0), 0.5)
            F.add_particle(pf, (1.0,0.0,0.0), (0.0,0.0,3.0), 0.5)
            F.add_particle(pf, (2.0,0.0,0.0), (0.0,0.0,4.0), 0.5)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=true, score=0.7)
            F.add_edge!(g, 2, 3; coherent=true, score=0.7)
            return pf
        end

        @testset "type-stable no-op returns" begin
            pf = F.ParticleField(4)
            n = @inferred F.refine_filaments!(pf)
            @test n == 0
            @test n isa Int

            pf = F.ParticleField(4)
            obs = @inferred F.refine_filaments_observables!(pf)
            @test obs.reports.initial isa F.FilamentEdgeReport
            @test obs.reports.repair isa F.FilamentEdgeReport
            @test obs.reports.final isa F.FilamentEdgeReport
            @test obs.counts == (
                inferred = 0,
                split = 0,
                exact_coarsened = 0,
                bundle_coarsened = 0,
                merged = 0,
                total = 0,
            )
        end

        @testset "repair clears coherence before split" begin
            pf = _orchestration_line_pf()
            g = pf.filament_edge_graph
            g.up_neighbor[1, 2] = 0
            F.set_up_count!(g, 2, 0)

            n = F.refine_filaments!(pf; do_infer=false, do_coarsen=false)
            @test n == 0
            @test pf.np == 2
            @test g.down_neighbor[1, 1] == 2
            @test g.down_coherent[1, 1] == false
        end

        @testset "inference and split end-to-end" begin
            pf = F.ParticleField(3)
            F.add_particle(pf, (0.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            F.add_particle(pf, (1.0,0.0,0.0), (1.0,0.0,0.0), 0.5)

            n = F.refine_filaments!(pf; do_coarsen=false)
            @test n == 2
            @test pf.np == 3
            @test F.validate_filament_edges(pf).ok
        end

        @testset "exact coarsen count and reserved counts" begin
            pf = _orchestration_inverse_pf()
            obs = F.refine_filaments_observables!(pf;
                                        do_infer=false,
                                        do_split=false,
                                        L_min=10.0)
            @test obs.counts.inferred == 0
            @test obs.counts.split == 0
            @test obs.counts.exact_coarsened == 1
            @test obs.counts.bundle_coarsened == 0
            @test obs.counts.merged == 0
            @test obs.counts.total == 1
            @test obs.reports.final.ok
        end

        @testset "toggles isolate stages" begin
            pf = F.ParticleField(3)
            F.add_particle(pf, (0.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            F.add_particle(pf, (1.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            obs = F.refine_filaments_observables!(pf; do_split=false, do_coarsen=false)
            @test obs.counts.inferred == 1
            @test obs.counts.split == 0
            @test obs.counts.total == 1
            @test pf.np == 2

            pf = _orchestration_line_pf()
            obs = F.refine_filaments_observables!(pf; do_infer=false, do_coarsen=false)
            @test obs.counts.inferred == 0
            @test obs.counts.split == 1
            @test obs.counts.total == 1
            @test pf.np == 3

            pf = _orchestration_inverse_pf()
            obs = F.refine_filaments_observables!(pf;
                                        do_infer=false,
                                        do_split=false,
                                        do_coarsen=false,
                                        L_min=10.0)
            @test obs.counts.total == 0
            @test pf.np == 3
        end

        @testset "observables report invalid topology without repair" begin
            pf = _orchestration_line_pf()
            g = pf.filament_edge_graph
            g.up_neighbor[1, 2] = 0
            F.set_up_count!(g, 2, 0)

            obs = F.refine_filaments_observables!(pf;
                                        do_repair=false,
                                        do_infer=false,
                                        do_split=false,
                                        do_coarsen=false)
            @test !obs.reports.initial.ok
            @test !obs.reports.repair.ok
            @test !obs.reports.final.ok
            @test g.down_coherent[1, 1] == true
        end
    end

    @testset "FilamentEdgeGraph bundle coarsen" begin

        # 2→1 trio: a, b → c with a, b overlapping and tangent-aligned.
        function _converge_trio_pf(; xa=(0.0, 0.0, 0.0),
                                    xb=(0.05, 0.0, 0.0),
                                    xc=(1.0, 0.0, 0.0),
                                    Γa=(1.0, 0.0, 0.0),
                                    Γb=(1.0, 0.0, 0.0),
                                    σa=0.5, σb=0.5, σc=0.5,
                                    coh_ac=true, coh_bc=true,
                                    cap=4)
            pf = F.ParticleField(cap)
            F.add_particle(pf, xa, Γa, σa)
            F.add_particle(pf, xb, Γb, σb)
            F.add_particle(pf, xc, (1.0, 0.0, 0.0), σc)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 3; coherent=coh_ac, score=0.7)
            F.add_edge!(g, 2, 3; coherent=coh_bc, score=0.7)
            return pf
        end

        # 1→2 trio: a → b, c with b, c overlapping and tangent-aligned.
        function _diverge_trio_pf(; xa=(0.0, 0.0, 0.0),
                                   xb=(1.0, 0.0, 0.0),
                                   xc=(1.05, 0.0, 0.0),
                                   Γb=(1.0, 0.0, 0.0),
                                   Γc=(1.0, 0.0, 0.0),
                                   σa=0.5, σb=0.5, σc=0.5,
                                   coh_ab=true, coh_ac=true,
                                   cap=4)
            pf = F.ParticleField(cap)
            F.add_particle(pf, xa, (1.0, 0.0, 0.0), σa)
            F.add_particle(pf, xb, Γb, σb)
            F.add_particle(pf, xc, Γc, σc)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 2; coherent=coh_ab, score=0.7)
            F.add_edge!(g, 1, 3; coherent=coh_ac, score=0.7)
            return pf
        end

        @testset "2→1 happy path: merge, Γ sum, position weighted" begin
            pf = _converge_trio_pf(Γa=(1.0, 0.0, 0.0), Γb=(3.0, 0.0, 0.0),
                                   xa=(0.0, 0.0, 0.0), xb=(0.1, 0.0, 0.0))
            g = pf.filament_edge_graph
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 1
            @test pf.np == 2  # a, c survive; b was removed via swap-with-last
            # Survivor at slot 1 (a's slot, lower index).
            xs = view(pf.particles, 1:3, 1)
            Γs = view(pf.particles, 4:6, 1)
            @test Γs[1] ≈ 4.0  # 1 + 3
            @test Γs[2] ≈ 0.0
            @test Γs[3] ≈ 0.0
            # x_uv = (1·0 + 3·0.1) / (1 + 3) = 0.075
            @test xs[1] ≈ 0.075
            @test xs[2] ≈ 0.0
            @test xs[3] ≈ 0.0
            # Edge graph: only edge survivor (slot 1) → c (slot 2 after swap).
            @test F.down_count(g, 1) == 1
            @test g.down_neighbor[1, 1] == 2
            @test g.down_coherent[1, 1] == true
            @test F.up_count(g, 2) == 1
        end

        @testset "1→2 happy path: merge, Γ sum, position weighted" begin
            pf = _diverge_trio_pf(Γb=(1.0, 0.0, 0.0), Γc=(3.0, 0.0, 0.0),
                                  xb=(1.0, 0.0, 0.0), xc=(1.1, 0.0, 0.0))
            g = pf.filament_edge_graph
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 1
            @test pf.np == 2  # a, survivor
            # Survivor of (b, c) lives at b's slot (slot 2, lower index).
            xs = view(pf.particles, 1:3, 2)
            Γs = view(pf.particles, 4:6, 2)
            @test Γs[1] ≈ 4.0
            # x = (1·1.0 + 3·1.1) / 4 = 1.075
            @test xs[1] ≈ 1.075
            @test F.down_count(g, 1) == 1
            @test g.down_neighbor[1, 1] == 2
        end

        @testset "linear impulse exact for tangent-aligned pair" begin
            pf = _converge_trio_pf(Γa=(2.0, 0.0, 0.0), Γb=(5.0, 0.0, 0.0),
                                   xa=(0.3, 0.7, -0.2),
                                   xb=(0.31, 0.71, -0.18))
            # I = ½ Σ x × Γ pre
            xa = pf.particles[1:3, 1]; Γa = pf.particles[4:6, 1]
            xb = pf.particles[1:3, 2]; Γb = pf.particles[4:6, 2]
            xc = pf.particles[1:3, 3]; Γc = pf.particles[4:6, 3]
            cross(a, b) = (a[2]*b[3] - a[3]*b[2],
                           a[3]*b[1] - a[1]*b[3],
                           a[1]*b[2] - a[2]*b[1])
            I_pre = cross(xa, Γa) .+ cross(xb, Γb) .+ cross(xc, Γc)
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 1
            xu = pf.particles[1:3, 1]; Γu = pf.particles[4:6, 1]
            xcp = pf.particles[1:3, 2]; Γcp = pf.particles[4:6, 2]
            I_post = cross(xu, Γu) .+ cross(xcp, Γcp)
            for k in 1:3
                @test I_pre[k] ≈ I_post[k] atol=1e-12
            end
        end

        @testset "total Γ conserved across trio" begin
            pf = _converge_trio_pf(Γa=(0.7, 0.3, -0.1),
                                   Γb=(0.6, 0.2, -0.05),
                                   xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            # Skip angle check by ensuring tangent gate passes (Γa, Γb are
            # nearly parallel here — verify).
            ga = pf.particles[4:6, 1]; gb = pf.particles[4:6, 2]
            gc = pf.particles[4:6, 3]
            total_pre = ga .+ gb .+ gc
            n = F.bundle_coarsen_filament_edges!(pf; bundle_angle_tol=π/2)
            @test n == 1
            total_post = pf.particles[4:6, 1] .+ pf.particles[4:6, 2]
            for k in 1:3
                @test total_pre[k] ≈ total_post[k]
            end
        end

        @testset "rejection: overlap gate (particles too far apart)" begin
            pf = _converge_trio_pf(xa=(0.0,0.0,0.0), xb=(5.0,0.0,0.0),
                                   xc=(2.5,1.0,0.0))
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 0
            @test pf.np == 3
        end

        @testset "rejection: tangent gate (anti-parallel Γ)" begin
            pf = _converge_trio_pf(Γa=(1.0,0.0,0.0), Γb=(-1.0,0.0,0.0),
                                   xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 0
            @test pf.np == 3
        end

        @testset "rejection: σ tol" begin
            pf = _converge_trio_pf(σa=0.5, σb=1.5,
                                   xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 0
            @test pf.np == 3
        end

        @testset "rejection: only_coherent with one incoherent edge" begin
            pf = _converge_trio_pf(coh_ac=true, coh_bc=false,
                                   xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 0
            # With only_coherent=false the coherence flags still must match,
            # so this still rejects.
            n2 = F.bundle_coarsen_filament_edges!(pf; only_coherent=false)
            @test n2 == 0
        end

        @testset "rejection: extra down edge on a or b would drop topology" begin
            pf = _converge_trio_pf(cap=5,
                                   xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            g = pf.filament_edge_graph
            # Add a 4th particle so a has an additional downstream edge.
            F.add_particle(pf, (-1.0,0.0,0.0), (1.0,0.0,0.0), 0.5)
            F.add_edge!(g, 1, 4; coherent=true, score=0.7)
            @test F.down_count(g, 1) == 2
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 0
        end

        @testset "transfer of upstream edge across merge" begin
            # x → a, x → b not possible (degree cap on x.down would prevent
            # both reaching the trio). Use distinct upstream donors.
            pf = F.ParticleField(6)
            F.add_particle(pf, (-1.0, 0.5, 0.0), (1.0,0.0,0.0), 0.5)  # 1: x
            F.add_particle(pf, (-1.0,-0.5, 0.0), (1.0,0.0,0.0), 0.5)  # 2: y
            F.add_particle(pf, ( 0.0, 0.0, 0.0), (1.0,0.0,0.0), 0.5)  # 3: a
            F.add_particle(pf, ( 0.05,0.0, 0.0), (1.0,0.0,0.0), 0.5)  # 4: b
            F.add_particle(pf, ( 1.0, 0.0, 0.0), (1.0,0.0,0.0), 0.5)  # 5: c
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 3; coherent=true, score=0.7)  # x → a
            F.add_edge!(g, 2, 4; coherent=true, score=0.7)  # y → b
            F.add_edge!(g, 3, 5; coherent=true, score=0.7)  # a → c
            F.add_edge!(g, 4, 5; coherent=true, score=0.7)  # b → c
            n = F.bundle_coarsen_filament_edges!(pf)
            @test n == 1
            @test pf.np == 4  # b removed; particles 1,2,3,5 (last shifted)
            # After remove_particle(4), the old particle 5 (c) moves to slot 4.
            # Survivor sits at slot 3 (lower of a=3, b=4).
            # Verify survivor has two upstream edges (x and y).
            @test F.up_count(g, 3) == 2
            # Verify survivor → c (now at slot 4).
            @test F.down_count(g, 3) == 1
            @test g.down_neighbor[1, 3] == 4
            @test F.validate_filament_edges(pf).ok
        end

        @testset "max_coarsens cap honored" begin
            pf = F.ParticleField(8)
            # Two independent 2→1 trios in distinct regions.
            F.add_particle(pf, (0.0,0.0,0.0), (1.0,0.0,0.0), 0.5)  # 1: a1
            F.add_particle(pf, (0.05,0.0,0.0),(1.0,0.0,0.0), 0.5) # 2: b1
            F.add_particle(pf, (1.0,0.0,0.0),  (1.0,0.0,0.0), 0.5) # 3: c1
            F.add_particle(pf, (10.0,0.0,0.0), (1.0,0.0,0.0), 0.5) # 4: a2
            F.add_particle(pf, (10.05,0.0,0.0),(1.0,0.0,0.0), 0.5) # 5: b2
            F.add_particle(pf, (11.0,0.0,0.0), (1.0,0.0,0.0), 0.5) # 6: c2
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 3; coherent=true, score=0.7)
            F.add_edge!(g, 2, 3; coherent=true, score=0.7)
            F.add_edge!(g, 4, 6; coherent=true, score=0.7)
            F.add_edge!(g, 5, 6; coherent=true, score=0.7)
            n = F.bundle_coarsen_filament_edges!(pf; max_coarsens=1)
            @test n == 1
            @test pf.np == 5
        end

        @testset "refine_filaments_observables! reports bundle count" begin
            pf = _converge_trio_pf(xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            obs = F.refine_filaments_observables!(pf;
                                                  do_infer=false,
                                                  do_split=false,
                                                  do_coarsen=false)
            @test obs.counts.bundle_coarsened == 1
            @test obs.counts.merged == 0
            @test obs.counts.total == 1
            @test obs.reports.final.ok
        end

        @testset "do_bundle_coarsen=false skips the pass" begin
            pf = _converge_trio_pf(xa=(0.0,0.0,0.0), xb=(0.05,0.0,0.0))
            obs = F.refine_filaments_observables!(pf;
                                                  do_infer=false,
                                                  do_split=false,
                                                  do_coarsen=false,
                                                  do_bundle_coarsen=false)
            @test obs.counts.bundle_coarsened == 0
            @test pf.np == 3
        end

        @testset "allocation-free no-op pass" begin
            pf = F.ParticleField(4)
            F.bundle_coarsen_filament_edges!(pf)  # warmup
            allocs = @allocated F.bundle_coarsen_filament_edges!(pf)
            @test allocs == 0
        end
    end

    @testset "FilamentEdgeGraph cross-filament merge" begin

        # Two filaments at y = ±dy, both running along +x with strength along
        # +x or ±x (parallel vs anti-parallel). σ uniform; spacing chosen so
        # cross-filament pairs sit inside r_merge·σ.
        function _two_filament_pf(; dy=0.05, σ=0.5, n=3, sign_b=+1.0,
                                   coh_a=true, coh_b=true, cap=2n+2)
            pf = F.ParticleField(cap)
            for k in 0:(n-1)
                F.add_particle(pf, (Float64(k), +dy, 0.0), (1.0, 0.0, 0.0), σ)
            end
            for k in 0:(n-1)
                F.add_particle(pf, (Float64(k), -dy, 0.0), (sign_b, 0.0, 0.0), σ)
            end
            g = pf.filament_edge_graph
            for k in 1:(n-1)
                F.add_edge!(g, k,     k+1;   coherent=coh_a, score=0.9)
                F.add_edge!(g, n+k,   n+k+1; coherent=coh_b, score=0.9)
            end
            return pf
        end

        @testset "merge_particles! default behavior unchanged" begin
            # Two close particles, no Γ-alignment gate: should merge as before.
            pf = F.ParticleField(4)
            F.add_particle(pf, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5)
            F.add_particle(pf, (0.05, 0.0, 0.0), (-1.0, 0.0, 0.0), 0.5)
            removed = F.merge_particles!(pf; r_merge=0.5)
            @test removed == 1
            @test pf.np == 1
        end

        @testset "anti-parallel rejection" begin
            pf = _two_filament_pf(sign_b=-1.0)
            np_before = pf.np
            removed = F.merge_filament_bundles!(pf;
                                                r_merge=0.5,
                                                cross_angle_tol=π/4)
            @test removed == 0
            @test pf.np == np_before
        end

        @testset "parallel merge: count, Γ sum, σ cube-root rule" begin
            pf = _two_filament_pf(sign_b=+1.0, n=3, dy=0.05, σ=0.5)
            np_before = pf.np
            # Total Γ_x before
            gamma_x_before = 0.0
            for i in 1:np_before
                gamma_x_before += pf.particles[F.GAMMA_INDEX.start, i]
            end
            removed = F.merge_filament_bundles!(pf;
                                                r_merge=0.5,
                                                cross_angle_tol=π/4)
            @test removed == 3      # 3 cross-pairs collapse
            @test pf.np == 3
            gamma_x_after = 0.0
            for i in 1:pf.np
                gamma_x_after += pf.particles[F.GAMMA_INDEX.start, i]
            end
            @test gamma_x_after ≈ gamma_x_before
            # Each survivor gets σ_uv = cbrt(σ_p^3 + σ_q^3) = cbrt(2) * 0.5
            σ_expected = cbrt(2.0) * 0.5
            for i in 1:pf.np
                @test pf.particles[F.SIGMA_INDEX, i] ≈ σ_expected atol=1e-12
            end
        end

        @testset "survivor incident coherence cleared" begin
            # Build two parallel particles with overlap; survivor (index 1)
            # has pre-existing chain edges 1↔3 and 4↔1 set coherent. After
            # the cross-merge collapses {1, 2}, the incident edges to 1
            # should have down_coherent cleared. An unrelated edge 5↔6 keeps
            # its coherence.
            pf = F.ParticleField(8)
            F.add_particle(pf, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5)  # 1 (survivor)
            F.add_particle(pf, (0.05, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5) # 2 (will merge into 1)
            F.add_particle(pf, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5)  # 3 (1 → 3)
            F.add_particle(pf, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5) # 4 (4 → 1)
            F.add_particle(pf, (5.0, 5.0, 0.0), (1.0, 0.0, 0.0), 0.5)  # 5 (unrelated)
            F.add_particle(pf, (6.0, 5.0, 0.0), (1.0, 0.0, 0.0), 0.5)  # 6 (unrelated)
            g = pf.filament_edge_graph
            F.add_edge!(g, 1, 3; coherent=true, score=0.9)
            F.add_edge!(g, 4, 1; coherent=true, score=0.9)
            F.add_edge!(g, 5, 6; coherent=true, score=0.9)

            removed = F.merge_filament_bundles!(pf; r_merge=0.5)
            @test removed == 1
            # After remove_particle(2), the moved particle (was index 6) lands
            # in slot 2. So 5→6 became 5→2 (mirror updated by remove_particle).
            # Slot identity of the survivor (index 1) is stable.
            # Incident edges to survivor: down_coherent on 1→3 cleared; on 4→1
            # cleared via the upstream walk.
            slot_1_3 = F.find_down_slot(g, 1, 3)
            @test slot_1_3 != 0
            @test g.down_coherent[slot_1_3, 1] == false
            slot_4_1 = F.find_down_slot(g, 4, 1)
            @test slot_4_1 != 0
            @test g.down_coherent[slot_4_1, 4] == false
            # Unrelated edge (originally 5→6, now 5→2 since particle 6 was
            # swapped into slot 2) keeps its coherence flag.
            slot_unrelated = F.find_down_slot(g, 5, 2)
            @test slot_unrelated != 0
            @test g.down_coherent[slot_unrelated, 5] == true
        end

        @testset "orchestrator toggle: do_cross_merge populates counts.merged" begin
            pf = _two_filament_pf(sign_b=+1.0)
            obs_off = F.refine_filaments_observables!(pf;
                                                       do_repair=false,
                                                       do_infer=false,
                                                       do_split=false,
                                                       do_coarsen=false,
                                                       do_bundle_coarsen=false)
            @test obs_off.counts.merged == 0

            pf2 = _two_filament_pf(sign_b=+1.0)
            obs_on = F.refine_filaments_observables!(pf2;
                                                      do_repair=false,
                                                      do_infer=false,
                                                      do_split=false,
                                                      do_coarsen=false,
                                                      do_bundle_coarsen=false,
                                                      do_cross_merge=true,
                                                      cross_r_merge=0.5,
                                                      cross_angle_tol=π/4)
            @test obs_on.counts.merged > 0
            @test obs_on.counts.total == obs_on.counts.merged
        end

        @testset "refine_filaments! return type stable with do_cross_merge" begin
            pf = F.ParticleField(4)
            n = @inferred F.refine_filaments!(pf; do_cross_merge=true)
            @test n isa Int
        end

        @testset "anti-parallel rejection via orchestrator" begin
            pf = _two_filament_pf(sign_b=-1.0)
            obs = F.refine_filaments_observables!(pf;
                                                   do_repair=false,
                                                   do_infer=false,
                                                   do_split=false,
                                                   do_coarsen=false,
                                                   do_bundle_coarsen=false,
                                                   do_cross_merge=true,
                                                   cross_angle_tol=π/4)
            @test obs.counts.merged == 0
        end

        @testset "static particles skipped" begin
            pf = F.ParticleField(4)
            F.add_particle(pf, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5; static=true)
            F.add_particle(pf, (0.05, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5)
            removed = F.merge_filament_bundles!(pf; r_merge=0.5)
            @test removed == 0
            @test pf.np == 2
        end
    end

end
