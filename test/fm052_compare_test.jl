using Test

include(joinpath(@__DIR__, "..", "scripts", "fm052_compare.jl"))

function write_fixture(root, name; ct_delta=0.0, gamma_factor=1.0,
                       particle_delta=0)
    dir = joinpath(root, name)
    monitors = joinpath(dir, "monitors")
    mkpath(monitors)
    open(joinpath(dir, "$(name)_CT_vs_rev.csv"), "w") do io
        println(io, "step,revolution,CT_bernoulli")
        println(io, "1,0.0,NaN")
        println(io, "721,20.0,$(0.07 + ct_delta)")
        println(io, "722,20.0278,$(0.08 + ct_delta)")
    end
    open(joinpath(monitors, "$(name)_bound_circulation.csv"), "w") do io
        println(io, "step,time,blade,section,r_over_R,circulation_te,circulation_slice")
        for step in 720:721, blade in 1:2, (section, radius, gamma) in
                ((1, 0.2, 1.0), (2, 0.5, 2.0), (3, 0.9, 4.0), (4, 1.0, 5.0))
            println(io, "$step,0,$blade,$section,$radius,$(gamma_factor * gamma),0")
        end
    end
    open(joinpath(monitors, "$(name)_wake_health.csv"), "w") do io
        println(io, "step,time,n_particles,wall_s")
        println(io, "720,0,$(100 + particle_delta),1.0")
        println(io, "721,0,$(110 + particle_delta),1.1")
    end
    return dir
end

@testset "fm052 campaign M2 and strict continuation gates" begin
    mktempdir() do root
        reference = write_fixture(root, "reference")
        candidate = write_fixture(root, "candidate"; ct_delta=0.001,
                                  gamma_factor=1.01)
        out = joinpath(root, "out")
        rows, steps = mode_compare(reference, candidate, out)
        metrics = Dict(row.metric => row.value for row in rows)
        @test steps == [720, 721]
        @test metrics["ct_cycle_mean_abs_difference"] ≈ 0.001
        @test metrics["gamma_m2_max_normalized_difference"] ≈ 0.01
        @test metrics["gamma_m2_rms_normalized_difference"] ≈
              sqrt(mean(abs2, [0.02, 0.04])) / 4.0
        @test isfile(joinpath(out, "fm052_comparison.csv"))
        @test isfile(joinpath(out, "fm052_comparison.md"))
        @test isfile(joinpath(out, "fm052_comparison_inputs.csv"))

        open(joinpath(candidate, "candidate_case_metadata.toml"), "w") do io
            println(io, "wall_time_s = 2.0")
            println(io, "n_steps = 2")
            println(io, "solver_S_gpu_enabled = true")
            println(io, "solver_S_gpu_upload_count = 1")
            println(io, "solver_S_gpu_gemv_count = 2")
            println(io, "solver_S_gpu_free_after_bytes = $(40 * 1024^3)")
            println(io, "solver_S_gpu_min_free_bytes = $(19 * 1024^3)")
        end
        write(joinpath(candidate, "candidate_process_wall_s.txt"), "5.0\n")
        logpath = joinpath(root, "candidate.log")
        open(logpath, "w") do io
            println(io, "[ Info: step_timer solve step=720 0.2 s")
            println(io, "[ Info: step_timer_nested wake_sfs step=720 0.1 s")
            println(io, "[ Info: gpu_timer wake_self 0.05 s")
            println(io, "[ Info: source_influence_s_gemv 0.01 s")
            println(io, "[ Info: source_s_gpu_memory gemv_1 total_bytes=$(80 * 1024^3) free_bytes=$(20 * 1024^3) pool_reserved_bytes=$(12 * 1024^3) pool_used_bytes=$(11 * 1024^3)")
            println(io, "[ Info: source_s_gpu_memory gemv_2 total_bytes=$(80 * 1024^3) free_bytes=$(19 * 1024^3) pool_reserved_bytes=$(12 * 1024^3) pool_used_bytes=$(11 * 1024^3)")
        end
        mode_report(candidate, logpath, out)
        @test isfile(joinpath(out, "fm052_timer_summary.csv"))
        @test occursin("source,source_influence_s_gemv", read(
            joinpath(out, "fm052_timer_summary.csv"), String))
        @test isfile(joinpath(out, "fm052_run_summary.csv"))
        @test isfile(joinpath(out, "fm052_step_trajectory.csv"))
        mode_memory_gate(candidate, out, 16.0)
        @test isfile(joinpath(out, "fm052_memory_gate.md"))
        mode_lock(reference, candidate, 720, 721, out)
        tolerance_path = joinpath(out, "fm052_locked_tolerances.toml")
        @test isfile(tolerance_path)
        mode_gate(reference, candidate, tolerance_path, out)
        @test isfile(joinpath(out, "fm052_gate.md"))

        divergent = write_fixture(root, "divergent"; particle_delta=1)
        @test_throws ErrorException mode_compare(reference, divergent,
                                                  joinpath(root, "bad"))

        # Platform-independent package gate permits isolated path roots but not
        # version/source changes (needed by the ARM-native GH200 environment).
        manifest_a = joinpath(root, "Manifest-a.toml")
        manifest_b = joinpath(root, "Manifest-b.toml")
        manifest_bad = joinpath(root, "Manifest-bad.toml")
        write(manifest_a, """manifest_format = \"2.0\"\n[deps]\n[[deps.Example]]\nuuid = \"00000000-0000-0000-0000-000000000001\"\nversion = \"1.0.0\"\npath = \"/x86/source\"\n""")
        write(manifest_b, """manifest_format = \"2.0\"\n[deps]\n[[deps.Example]]\nuuid = \"00000000-0000-0000-0000-000000000001\"\nversion = \"1.0.0\"\npath = \"/arm/source\"\n""")
        write(manifest_bad, """manifest_format = \"2.0\"\n[deps]\n[[deps.Example]]\nuuid = \"00000000-0000-0000-0000-000000000001\"\nversion = \"2.0.0\"\npath = \"/arm/source\"\n""")
        manifest_gate = joinpath(root, "manifest-gate.md")
        mode_manifest_gate(manifest_a, manifest_b, manifest_gate)
        @test isfile(manifest_gate)
        @test_throws ErrorException mode_manifest_gate(manifest_a, manifest_bad,
                                                        joinpath(root, "bad-manifest.md"))

        # Architecture-qualified mature reports retain raw values, normalize
        # performance to H200, and keep low-memory rejection as a result row.
        gh200 = write_fixture(root, "fm052_gh200_mature_gpu_s";
                              ct_delta=0.001, gamma_factor=1.01)
        open(joinpath(gh200, "fm052_gh200_mature_gpu_s_case_metadata.toml"), "w") do io
            println(io, "wall_time_s = 3.0")
            println(io, "n_steps = 2")
            println(io, "solver_S_gpu_enabled = true")
            println(io, "solver_S_gpu_upload_count = 1")
            println(io, "solver_S_gpu_gemv_count = 2")
            println(io, "solver_S_gpu_matrix_bytes = 100")
            println(io, "solver_S_gpu_allocation_bytes = 120")
            println(io, "solver_S_gpu_pool_reserved_after_bytes = 30")
            println(io, "solver_S_gpu_pool_used_after_bytes = 20")
            println(io, "solver_S_gpu_upload_s = 0.5")
            println(io, "solver_S_gpu_free_after_bytes = $(40 * 1024^3)")
            println(io, "solver_S_gpu_min_free_bytes = $(18 * 1024^3)")
        end
        write(joinpath(gh200, "fm052_gh200_mature_gpu_s_process_wall_s.txt"), "10.0\n")
        ghlog = joinpath(root, "fm052_gh200_mature_gpu_s.log")
        open(ghlog, "w") do io
            println(io, "[ Info: step_timer solve step=720 0.4 s")
            println(io, "[ Info: source_influence_s_gpu_gemv 0.02 s")
            println(io, "[ Info: source_influence_s_gpu_gemv 0.03 s")
            println(io, "[ Info: source_s_gpu_memory gemv_1 total_bytes=$(96 * 1024^3) free_bytes=$(19 * 1024^3) pool_reserved_bytes=30 pool_used_bytes=20")
            println(io, "[ Info: source_s_gpu_memory gemv_2 total_bytes=$(96 * 1024^3) free_bytes=$(18 * 1024^3) pool_reserved_bytes=30 pool_used_bytes=20")
        end
        ghreport = joinpath(root, "reports", "gh200", "mature")
        mode_compare(reference, gh200, ghreport)
        mode_report(gh200, ghlog, ghreport)

        function write_result(path, arch, status, run_dir, report_dir; reason="ok")
            mkpath(dirname(path))
            open(path, "w") do io
                println(io, "architecture = ", repr(arch))
                println(io, "stage = \"mature\"")
                println(io, "status = ", repr(status))
                println(io, "reason = ", repr(reason))
                println(io, "run_dir = ", repr(run_dir))
                println(io, "report_dir = ", repr(report_dir))
                println(io, "observed_gpu_model = ", repr(uppercase(arch)))
                println(io, "observed_gpu_vram_mib = 100000")
                println(io, "observed_compute_capability = \"9.0\"")
                println(io, "node = \"test-node\"")
                println(io, "partition = \"test\"")
                println(io, "cpu_architecture = \"x86_64\"")
                println(io, "cpu_model = \"test CPU\"")
                println(io, "job_id = \"1\"")
            end
            path
        end
        hmanifest = write_result(joinpath(root, "manifests", "h200", "fm052_h200_mature_result.toml"),
                                 "h200", "pass", candidate, out)
        ghmanifest = write_result(joinpath(root, "manifests", "gh200", "fm052_gh200_mature_result.toml"),
                                  "gh200", "pass", gh200, ghreport)
        lmanifest = write_result(joinpath(root, "manifests", "l40s", "fm052_l40s_probe_result.toml"),
                                 "l40s", "ineligible", "", "";
                                 reason="startup_vram_below_official_requirement")
        # Ineligible rows originate at probe, not mature.
        ldata = TOML.parsefile(lmanifest)
        ldata["stage"] = "probe"
        open(lmanifest, "w") do io
            TOML.print(io, ldata; sorted=true)
        end
        cross = joinpath(root, "cross-architecture")
        mode_cross_arch(cross, [hmanifest, ghmanifest, lmanifest])
        cross_csv = read(joinpath(cross, "fm052_cross_architecture_summary.csv"), String)
        @test occursin("gh200,pass,ok,process_wall,10.0,s,5.0,2.0", cross_csv)
        @test occursin("l40s,ineligible,startup_vram_below_official_requirement", cross_csv)
        @test isfile(joinpath(cross, "fm052_cross_architecture_summary.md"))
    end
end
