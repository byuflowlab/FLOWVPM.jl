using Test
import FLOWVPM

function vorticity_storage_fmm()
    return FLOWVPM.FMM(;
        p=4,
        ncrit=8,
        theta=0.0,
        shrink_recenter=true,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-10,
        autotune_p=false,
        autotune_ncrit=false,
        autotune_reg_error=false,
        min_ncrit=1,
    )
end

function vorticity_storage_field(; integration=FLOWVPM.rungekutta3)
    pfield = FLOWVPM.ParticleField(4; fmm=vorticity_storage_fmm(), integration)

    FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.2, -0.1), 0.50)
    FLOWVPM.add_particle(pfield, (0.2, 0.1, 0.0), (-0.2, 1.1, 0.3), 0.45)
    FLOWVPM.add_particle(pfield, (1.5, 0.2, 0.1), (0.5, -0.4, 0.8), 0.55)
    FLOWVPM.add_particle(pfield, (-0.3, 1.0, 0.4), (-0.7, 0.3, 1.2), 0.60)

    return pfield
end

function particle_snapshots(pfield)
    U = [copy(FLOWVPM.get_U(pfield, i)) for i in 1:FLOWVPM.get_np(pfield)]
    V = [copy(FLOWVPM.get_vorticity(pfield, i)) for i in 1:FLOWVPM.get_np(pfield)]
    J = [copy(FLOWVPM.get_J(pfield, i)) for i in 1:FLOWVPM.get_np(pfield)]
    return U, V, J
end

function set_all_vorticity!(pfield, value)
    for i in 1:FLOWVPM.get_np(pfield)
        FLOWVPM.set_vorticity(pfield, i, value)
    end
end

function set_all_J!(pfield, value)
    for i in 1:FLOWVPM.get_np(pfield)
        FLOWVPM.set_J(pfield, i, value)
    end
end

@testset "Vorticity storage" begin
    @testset "UJ_fmm without vorticity leaves VORTICITY_INDEX untouched unless reset" begin
        reference = vorticity_storage_field()
        FLOWVPM.UJ_direct(reference; reset=true)
        U_ref, _, J_ref = particle_snapshots(reference)

        noreset = vorticity_storage_field()
        sentinel = (9.0, -8.0, 7.0)
        set_all_vorticity!(noreset, sentinel)
        FLOWVPM.UJ_fmm(noreset; vorticity=false, reset=false, autotune=false)
        U_noreset, V_noreset, J_noreset = particle_snapshots(noreset)

        for i in 1:FLOWVPM.get_np(noreset)
            @test U_noreset[i] ≈ U_ref[i] rtol=1e-12 atol=1e-14
            @test J_noreset[i] ≈ J_ref[i] rtol=1e-12 atol=1e-14
            @test V_noreset[i] == collect(sentinel)
        end

        reset = vorticity_storage_field()
        set_all_vorticity!(reset, sentinel)
        FLOWVPM.UJ_fmm(reset; vorticity=false, reset=true, autotune=false)
        U_reset, V_reset, J_reset = particle_snapshots(reset)

        for i in 1:FLOWVPM.get_np(reset)
            @test U_reset[i] ≈ U_ref[i] rtol=1e-12 atol=1e-14
            @test J_reset[i] ≈ J_ref[i] rtol=1e-12 atol=1e-14
            @test V_reset[i] ≈ zeros(3) atol=0
        end
    end

    @testset "UJ_fmm with vorticity preserves U/J and stores induced curl in VORTICITY_INDEX" begin
        without_vorticity = vorticity_storage_field()
        FLOWVPM.UJ_fmm(without_vorticity; vorticity=false, reset=true, autotune=false)
        U0, V0, J0 = particle_snapshots(without_vorticity)

        with_vorticity = vorticity_storage_field()
        FLOWVPM.UJ_fmm(with_vorticity; vorticity=true, reset=true, autotune=false)
        U1, V1, J1 = particle_snapshots(with_vorticity)

        reference = vorticity_storage_field()
        FLOWVPM.UJ_direct(reference; reset=true)

        nonzero_vorticity = false
        for i in 1:FLOWVPM.get_np(with_vorticity)
            @test U1[i] ≈ U0[i] rtol=1e-12 atol=1e-14
            @test J1[i] ≈ J0[i] rtol=1e-12 atol=1e-14
            @test V0[i] ≈ zeros(3) atol=0
            @test V1[i] ≈ collect(FLOWVPM.get_W(with_vorticity, i)) rtol=1e-12 atol=1e-14
            @test V1[i] ≈ collect(FLOWVPM.get_W(reference, i)) rtol=1e-12 atol=1e-14
            nonzero_vorticity |= any(!iszero, V1[i])
        end
        @test nonzero_vorticity
    end

    @testset "Basis-function vorticity uses VORTICITY_INDEX and consumers read it" begin
        direct = vorticity_storage_field(; integration=FLOWVPM.euler)
        J_sentinel = collect(1.0:9.0)
        set_all_J!(direct, J_sentinel)
        FLOWVPM.zeta_direct(direct)
        _, V_direct, J_direct = particle_snapshots(direct)

        fmm = vorticity_storage_field()
        set_all_J!(fmm, J_sentinel)
        FLOWVPM.zeta_fmm(fmm)
        _, V_fmm, J_fmm = particle_snapshots(fmm)

        for i in 1:FLOWVPM.get_np(direct)
            @test J_direct[i] == J_sentinel
            @test J_fmm[i] == J_sentinel
            @test V_fmm[i] ≈ V_direct[i] rtol=1e-12 atol=1e-14
        end

        captured_targets = Ref{Vector{Vector{Float64}}}()
        function capture_rbf!(pfield, scheme)
            captured_targets[] = [copy(FLOWVPM.get_M(pfield, i)[7:9]) for i in 1:FLOWVPM.get_np(pfield)]
            return nothing
        end

        scheme = FLOWVPM.CoreSpreading(1.0, 1.0, FLOWVPM.zeta_direct; beta=1.0, rbf=capture_rbf!)
        viscous = vorticity_storage_field(; integration=FLOWVPM.euler)
        set_all_J!(viscous, (-100.0, -200.0, -300.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        FLOWVPM.viscousdiffusion(viscous, scheme, 1.0)

        @test captured_targets[] !== nothing
        for i in 1:FLOWVPM.get_np(viscous)
            @test captured_targets[][i] ≈ collect(FLOWVPM.get_vorticity(viscous, i)) rtol=1e-12 atol=1e-14
            @test captured_targets[][i] != [-100.0, -200.0, -300.0]
        end
    end
end
