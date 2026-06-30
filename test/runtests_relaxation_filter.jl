using Test
import FLOWVPM

# Build a field whose vorticity (from J) is perpendicular to each particle's
# circulation, so an applied relaxation produces a measurable change in Γ.
function relaxation_filter_field()
    pfield = FLOWVPM.ParticleField(4; integration=FLOWVPM.euler)

    # Γ = +x for every particle; positions spread along z.
    FLOWVPM.add_particle(pfield, ( 0.0, 0.0,  1.0), (1.0, 0.0, 0.0), 0.1)
    FLOWVPM.add_particle(pfield, ( 0.0, 0.0,  0.5), (1.0, 0.0, 0.0), 0.1)
    FLOWVPM.add_particle(pfield, ( 0.0, 0.0, -0.5), (1.0, 0.0, 0.0), 0.1)
    FLOWVPM.add_particle(pfield, ( 0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 0.1)

    # Velocity gradient producing vorticity ω = (J6-J8, J7-J3, J2-J4) = (0, 2, 0),
    # i.e. +y, perpendicular to Γ=+x, so corrected-Pedrizzetti realigns Γ.
    J = zeros(9); J[7] = 1.0; J[3] = -1.0
    for i in 1:FLOWVPM.get_np(pfield)
        FLOWVPM.set_J(pfield, i, J)
    end
    return pfield
end

gammas(pf) = [copy(FLOWVPM.get_Gamma(pf, i)) for i in 1:FLOWVPM.get_np(pf)]

@testset "Relaxation filter" begin
    @testset "3-arg constructor stays backward compatible" begin
        r = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3)
        @test r.filter === FLOWVPM.relax_filter_all
        @test FLOWVPM.relax_filter_all(relaxation_filter_field(), 1)
        @test r != FLOWVPM.relaxation_none          # comparison guard still distinguishes
        @test r == FLOWVPM.relaxation_correctedpedrizzetti
    end

    @testset "all-pass filter realigns every particle (legacy behavior)" begin
        pf = relaxation_filter_field()
        G0 = gammas(pf)
        r = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3)
        for i in 1:FLOWVPM.get_np(pf); r(pf, i); end
        G1 = gammas(pf)
        @test all(G1[i] != G0[i] for i in 1:FLOWVPM.get_np(pf))
    end

    @testset "false filter leaves Γ untouched" begin
        pf = relaxation_filter_field()
        G0 = gammas(pf)
        r = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3, p->false)
        @test r != FLOWVPM.relaxation_none
        for i in 1:FLOWVPM.get_np(pf); r(pf, i); end
        @test gammas(pf) == G0
    end

    @testset "spatial filter gates by particle position" begin
        pf = relaxation_filter_field()
        G0 = gammas(pf)
        # relax only particles with z < 0 (the two downstream particles)
        downstream(p) = FLOWVPM.get_X(p)[3] < 0
        r = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3, downstream)
        for i in 1:FLOWVPM.get_np(pf); r(pf, i); end
        G1 = gammas(pf)
        @test G1[1] == G0[1]   # z = +1.0, skipped
        @test G1[2] == G0[2]   # z = +0.5, skipped
        @test G1[3] != G0[3]   # z = -0.5, relaxed
        @test G1[4] != G0[4]   # z = -1.0, relaxed
    end
end
