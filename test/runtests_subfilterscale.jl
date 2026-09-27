using Test
using LinearAlgebra
import FLOWVPM

function sfs_consistency_fmm()
    return FLOWVPM.FMM(;
        p=4,
        ncrit=2,
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

function sfs_consistency_field(; static_first=false, transposed=true)
    pfield = FLOWVPM.ParticleField(4; transposed, fmm=sfs_consistency_fmm())

    FLOWVPM.add_particle(pfield, (0.0, 0.0, 0.0), (1.0, 0.2, -0.1), 0.50; static=static_first)
    FLOWVPM.add_particle(pfield, (0.2, 0.1, 0.0), (-0.2, 1.1, 0.3), 0.45)
    FLOWVPM.add_particle(pfield, (1.5, 0.2, 0.1), (0.5, -0.4, 0.8), 0.55)
    FLOWVPM.add_particle(pfield, (-0.3, 1.0, 0.4), (-0.7, 0.3, 1.2), 0.60)

    return pfield
end

function sfs_thread_race_field()
    n = 48
    pfield = FLOWVPM.ParticleField(
        n;
        fmm=FLOWVPM.FMM(;
            p=4,
            ncrit=1,
            theta=0.0,
            shrink_recenter=true,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-10,
            autotune_p=false,
            autotune_ncrit=false,
            autotune_reg_error=false,
            min_ncrit=1,
        ),
    )

    for i in 1:n
        x = (sin(0.37 * i), cos(0.29 * i), sin(0.19 * i) + 0.03 * i)
        gamma = (cos(0.41 * i), sin(0.31 * i), cos(0.23 * i))
        sigma = 0.35 + 0.01 * (i % 5)
        FLOWVPM.add_particle(pfield, x, gamma, sigma)
    end

    return pfield
end

function run_Estr_direct!(pfield)
    FLOWVPM.UJ_direct(pfield; sfs=false, reset=true, reset_sfs=true)
    FLOWVPM.Estr_direct!(pfield)
    return pfield
end

function run_Estr_fmm_direct_list!(pfield)
    fmm_options = pfield.fmm
    _, _, target_tree, source_tree, _, direct_list, _ = FLOWVPM.fmm.fmm!(
        pfield;
        expansion_order=fmm_options.p - 1,
        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit),
        multipole_acceptance=fmm_options.theta,
        error_tolerance=FLOWVPM.fmm.PowerRelativeGradient{
            fmm_options.relative_tolerance,
            fmm_options.absolute_tolerance,
            true,
        }(),
        tune=true,
        shrink=fmm_options.shrink_recenter,
        recenter=fmm_options.shrink_recenter,
        nearfield_device=false,
        scalar_potential=false,
        hessian=true,
        silence_warnings=true,
    )

    FLOWVPM.Estr_fmm!(pfield, pfield, target_tree, source_tree, direct_list)
    return pfield
end

function test_Estr_consistency(; static_first=false, transposed=true)
    direct = run_Estr_direct!(sfs_consistency_field(; static_first, transposed))
    fmm = run_Estr_fmm_direct_list!(sfs_consistency_field(; static_first, transposed))

    @test FLOWVPM.get_np(direct) == FLOWVPM.get_np(fmm)
    for i in 1:FLOWVPM.get_np(direct)
        @test FLOWVPM.get_SFS(fmm, i) ≈ FLOWVPM.get_SFS(direct, i) rtol=1e-12 atol=1e-14
    end
end

function test_threaded_Estr_fmm_consistency()
    direct = run_Estr_direct!(sfs_thread_race_field())
    reference_sfs = [copy(FLOWVPM.get_SFS(direct, i)) for i in 1:FLOWVPM.get_np(direct)]

    for _ in 1:10
        fmm = run_Estr_fmm_direct_list!(sfs_thread_race_field())
        @test FLOWVPM.get_np(fmm) == length(reference_sfs)
        for i in 1:FLOWVPM.get_np(fmm)
            @test FLOWVPM.get_SFS(fmm, i) ≈ reference_sfs[i] rtol=1e-12 atol=1e-14
        end
    end
end

@testset "Subfilter-scale models" begin
    @testset "Estr_fmm direct list matches Estr_direct" begin
        test_Estr_consistency(; transposed=true)
        test_Estr_consistency(; transposed=false)
    end

    @testset "Estr_fmm direct list matches Estr_direct for static targets" begin
        test_Estr_consistency(; static_first=true)
    end

    @testset "Estr_fmm threaded direct list is race-free" begin
        if Threads.nthreads() == 1
            @test_skip "run with julia -t >1 to exercise threaded Estr_fmm!"
        else
            test_threaded_Estr_fmm_consistency()
        end
    end
end
