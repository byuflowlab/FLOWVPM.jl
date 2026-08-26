#!/usr/bin/env julia
# Task 052 matched-continuation comparison, timer report, and artifact gates.

using Printf
using SHA
using Statistics
using TOML

function read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = strip.(split(lines[1], ','))
    rows = Vector{Vector{String}}()
    for (irow, line) in enumerate(lines[2:end])
        iline = irow + 1
        isempty(strip(line)) && continue
        row = strip.(split(line, ','; keepempty=true))
        length(row) == length(header) || error(
            "malformed CSV $path:$iline: $(length(row)) fields, expected $(length(header))")
        push!(rows, row)
    end
    return header, rows
end

function col(header, name)
    index = findfirst(==(name), header)
    index === nothing && error("column $name not found in $(header)")
    return index
end

function finite64(s, context)
    x = tryparse(Float64, strip(s))
    x === nothing && error("nonnumeric sample at $context: $(repr(s))")
    isfinite(x) || error("nonfinite sample at $context: $x")
    return x
end

intvalue(s, context) = round(Int, finite64(s, context))

function unique_file(dir, pred, description)
    hits = sort(filter(pred, readdir(dir; join=true)))
    length(hits) == 1 || error("expected one $description in $dir, found $(length(hits)): $hits")
    return only(hits)
end

function monitor_csv(run_dir, stem)
    dir = joinpath(run_dir, "monitors")
    isdir(dir) || error("missing monitor directory: $dir")
    return unique_file(dir, p -> occursin(stem, basename(p)) && endswith(p, ".csv"),
                       "$stem monitor CSV")
end

function insert_unique!(dict, key, value, context)
    haskey(dict, key) && error("duplicate key $key in $context")
    dict[key] = value
end

"Simulation step => CT. Driver CSV indices are one greater than simulate!'s step."
function ct_history(run_dir)
    path = unique_file(run_dir, p -> endswith(p, "_CT_vs_rev.csv"), "CT history")
    h, rows = read_csv(path)
    cs, cc = col(h, "step"), col(h, "CT_bernoulli")
    out = Dict{Int,Float64}()
    for (irow, row) in enumerate(rows)
        i = irow + 1
        step = intvalue(row[cs], "$path:$i step") - 1
        value = tryparse(Float64, row[cc])
        value === nothing && error("nonnumeric CT at $path:$i")
        isfinite(value) || continue
        insert_unique!(out, step, value, path)
    end
    isempty(out) && error("no finite continuation CT samples in $path")
    return out
end

"Simulation step => particle count and wall."
function wake_health(run_dir)
    path = monitor_csv(run_dir, "wake_health")
    h, rows = read_csv(path)
    cs, cw, cn = col(h, "step"), col(h, "wall_s"), col(h, "n_particles")
    out = Dict{Int,NamedTuple{(:wall_s,:n_particles),Tuple{Float64,Int}}}()
    for (irow, row) in enumerate(rows)
        i = irow + 1
        step = intvalue(row[cs], "$path:$i step")
        value = (; wall_s=finite64(row[cw], "$path:$i wall_s"),
                  n_particles=intvalue(row[cn], "$path:$i n_particles"))
        insert_unique!(out, step, value, path)
    end
    return out
end

"(simulation step, blade, section) => (r/R, Gamma)."
function bound_circulation(run_dir)
    path = monitor_csv(run_dir, "bound_circulation")
    h, rows = read_csv(path)
    cs, cb, cn = col(h, "step"), col(h, "blade"), col(h, "section")
    cr, cg = col(h, "r_over_R"), col(h, "circulation_te")
    out = Dict{NTuple{3,Int},Tuple{Float64,Float64}}()
    for (irow, row) in enumerate(rows)
        i = irow + 1
        key = (intvalue(row[cs], "$path:$i step"),
               intvalue(row[cb], "$path:$i blade"),
               intvalue(row[cn], "$path:$i section"))
        value = (finite64(row[cr], "$path:$i r_over_R"),
                 finite64(row[cg], "$path:$i circulation_te"))
        insert_unique!(out, key, value, path)
    end
    return out
end

step_set(d::Dict{Int}) = Set(keys(d))
step_set(d::Dict{NTuple{3,Int}}) = Set(k[1] for k in keys(d))

function arm_data(run_dir; require_wake::Bool=true)
    ct, gamma = ct_history(run_dir), bound_circulation(run_dir)
    sets = (ct=step_set(ct), gamma=step_set(gamma))
    # Restarted driver histories retain finite pre-restart CT placeholders.
    # The circulation monitor identifies the actual continuation, and CT must
    # contain every one of those samples; extra earlier CT rows are ignored.
    sets.gamma <= sets.ct || error(
        "CT is missing continuation samples within $run_dir: " *
        "Gamma=$(sort!(collect(sets.gamma))) CT=$(sort!(collect(sets.ct)))")
    wake = require_wake ? wake_health(run_dir) : nothing
    if require_wake
        wake_steps = step_set(wake)
        sets.gamma == wake_steps || error(
            "unequal continuation coverage within $run_dir: Gamma=$(sort!(collect(sets.gamma))) " *
            "wake=$(sort!(collect(wake_steps)))")
    end
    return (; ct, gamma, wake, steps=sets.gamma)
end

function averaged_gamma(data, steps)
    sections = sort!(unique(k[3] for k in keys(data.gamma) if k[1] in steps))
    blades = unique(k[2] for k in keys(data.gamma) if k[1] in steps)
    isempty(sections) && error("no circulation sections in continuation window")
    profile = Dict{Int,Tuple{Float64,Float64}}()
    for section in sections
        samples = [(v[1], v[2]) for (k, v) in data.gamma
                   if k[1] in steps && k[3] == section]
        expected = length(steps) * length(blades)
        length(samples) == expected || error(
            "incomplete blade/step coverage at section $section: $(length(samples)) != $expected")
        # The two rotor blades use opposite signed radial coordinates; campaign
        # M2 compares the common physical radius and blade-averages Gamma.
        radii = abs.(first.(samples))
        maximum(radii) - minimum(radii) <= 1e-12 || error(
            "radial grid moves within arm at section $section: $(extrema(radii))")
        profile[section] = (mean(radii), mean(last.(samples)))
    end
    return profile
end

function write_comparison_outputs(outdir, reference_dir, candidate_dir, steps, rows;
                                  include_wake::Bool=true, gamma_steps=steps)
    mkpath(outdir)
    csvpath = joinpath(outdir, "fm052_comparison.csv")
    open(csvpath, "w") do io
        println(io, "metric,value,units,window_start,window_stop,gamma_window_start,gamma_window_stop,reference,candidate")
        for row in rows
            println(io, join((row.metric, row.value, row.units, first(steps), last(steps),
                              first(gamma_steps), last(gamma_steps),
                              abspath(reference_dir), abspath(candidate_dir)), ','))
        end
    end
    inputpath = joinpath(outdir, "fm052_comparison_inputs.csv")
    input_files = String[]
    for dir in (reference_dir, candidate_dir)
        push!(input_files, unique_file(dir, p -> endswith(p, "_CT_vs_rev.csv"), "CT history"))
        push!(input_files, monitor_csv(dir, "bound_circulation"))
        include_wake && push!(input_files, monitor_csv(dir, "wake_health"))
    end
    open(inputpath, "w") do io
        println(io, "path,sha256")
        for path in input_files
            println(io, "$(abspath(path)),$(bytes2hex(sha256(read(path))))")
        end
    end
    mdpath = joinpath(outdir, "fm052_comparison.md")
    open(mdpath, "w") do io
        println(io, "# fm052 matched-continuation comparison\n")
        println(io, "Reference: `$(abspath(reference_dir))`  ")
        println(io, "Candidate: `$(abspath(candidate_dir))`  ")
        println(io, "Common continuation window: **$(first(steps))–$(last(steps))** ($(length(steps)) steps)\n")
        gamma_steps == steps || println(io,
            "Campaign M2 circulation window: **$(first(gamma_steps))–$(last(gamma_steps))** ($(length(gamma_steps)) steps)\n")
        println(io, "| Metric | Value | Units |\n|---|---:|---|")
        for row in rows
            value_string = @sprintf("%.12g", row.value)
            println(io, "| $(row.metric) | $(value_string) | $(row.units) |")
        end
    end
    println("wrote $csvpath")
    println("wrote $mdpath")
    println("wrote $inputpath")
end

function mode_compare(reference_dir, candidate_dir, outdir=candidate_dir;
                      window=nothing, require_particle_match::Bool=true,
                      require_wake::Bool=true, gamma_window=nothing)
    require_particle_match && !require_wake && error("particle matching requires wake data")
    ref = arm_data(reference_dir; require_wake)
    cand = arm_data(candidate_dir; require_wake)
    if window === nothing
        ref.steps == cand.steps || error(
            "unequal continuation coverage: reference=$(sort!(collect(ref.steps))) " *
            "candidate=$(sort!(collect(cand.steps)))")
        selected_steps = ref.steps
    else
        selected_steps = Set(window)
        selected_steps <= ref.steps || error("reference is missing requested window steps")
        selected_steps <= cand.steps || error("candidate is missing requested window steps")
    end
    steps = sort!(collect(selected_steps))
    isempty(steps) && error("empty continuation window")
    steps == collect(first(steps):last(steps)) || error("noncontiguous continuation window: $steps")
    selected_gamma_steps = gamma_window === nothing ? selected_steps : Set(gamma_window)
    selected_gamma_steps <= ref.steps || error("reference is missing requested Gamma window steps")
    selected_gamma_steps <= cand.steps || error("candidate is missing requested Gamma window steps")
    gamma_steps = sort!(collect(selected_gamma_steps))
    isempty(gamma_steps) && error("empty Gamma window")
    gamma_steps == collect(first(gamma_steps):last(gamma_steps)) ||
        error("noncontiguous Gamma window: $gamma_steps")
    if require_particle_match
        for step in steps
            ref.wake[step].n_particles == cand.wake[step].n_particles || error(
                "particle-count divergence at step $step: $(ref.wake[step].n_particles) != " *
                "$(cand.wake[step].n_particles)")
        end
    end

    ct_ref = [ref.ct[s] for s in steps]
    ct_cand = [cand.ct[s] for s in steps]
    ct_diff = ct_cand .- ct_ref
    ct_ref_mean, ct_cand_mean = mean(ct_ref), mean(ct_cand)

    gp_ref = averaged_gamma(ref, selected_gamma_steps)
    gp_cand = averaged_gamma(cand, selected_gamma_steps)
    Set(keys(gp_ref)) == Set(keys(gp_cand)) || error("mismatched radial section grids")
    selected = Int[]
    for section in sort!(collect(keys(gp_ref)))
        rr, rc = gp_ref[section][1], gp_cand[section][1]
        isapprox(rr, rc; rtol=0, atol=1e-12) || error(
            "mismatched radial grid at section $section: $rr != $rc")
        0.3 <= rr <= 0.95 && push!(selected, section)
    end
    isempty(selected) && error("no Gamma samples in 0.3 <= r/R <= 0.95")
    gamma_ref = [gp_ref[s][2] for s in selected]
    gamma_cand = [gp_cand[s][2] for s in selected]
    gamma_diff = gamma_cand .- gamma_ref
    gamma_scale = maximum(abs, gamma_ref)
    gamma_scale > 0 || error("reference averaged Gamma profile has zero global magnitude")

    rows = [
        (metric="ct_reference_cycle_mean", value=ct_ref_mean, units="1"),
        (metric="ct_candidate_cycle_mean", value=ct_cand_mean, units="1"),
        (metric="ct_cycle_mean_abs_difference", value=abs(ct_cand_mean-ct_ref_mean), units="1"),
        (metric="ct_cycle_mean_relative_difference", value=abs(ct_cand_mean-ct_ref_mean)/max(abs(ct_ref_mean), eps()), units="1"),
        (metric="ct_per_step_max_abs_difference", value=maximum(abs, ct_diff), units="1"),
        (metric="ct_per_step_rms_difference", value=sqrt(mean(abs2, ct_diff)), units="1"),
        (metric="gamma_m2_max_normalized_difference", value=maximum(abs, gamma_diff)/gamma_scale, units="1"),
        (metric="gamma_m2_rms_normalized_difference", value=sqrt(mean(abs2, gamma_diff))/gamma_scale, units="1"),
        (metric="gamma_reference_global_max_magnitude", value=gamma_scale, units="m2_per_s"),
    ]
    require_wake && push!(rows,
        (metric="particle_count_final", value=Float64(ref.wake[last(steps)].n_particles), units="particles"))
    println("matched continuation $(first(steps)):$(last(steps)) ($(length(steps)) steps)")
    for row in rows
        @printf("  %-46s %.12g %s\n", row.metric, row.value, row.units)
    end
    write_comparison_outputs(outdir, reference_dir, candidate_dir, steps, rows;
        include_wake=require_wake, gamma_steps)
    return rows, steps
end

function mode_lock(reference_dir, candidate_dir, first_step, last_step, outdir)
    # p018_analyze.py selected CT rows by revolutions 22--31, then used those
    # unshifted driver row indices as monitor steps.  CT history indices are one
    # greater than simulate! steps, so reproducing the campaign's published M2
    # definition uses Gamma first_step+1:last_step (the final requested monitor
    # step does not exist).  Keep this convention confined to tolerance locking;
    # new matched continuations use their exact common raw-step window.
    gamma_window = (first_step + 1):last_step
    rows, steps = mode_compare(reference_dir, candidate_dir, outdir;
        window=first_step:last_step, require_particle_match=false,
        require_wake=false, gamma_window)
    values = Dict(row.metric => row.value for row in rows)
    path = joinpath(outdir, "fm052_locked_tolerances.toml")
    open(path, "w") do io
        println(io, "reference = $(repr(abspath(reference_dir)))")
        println(io, "candidate = $(repr(abspath(candidate_dir)))")
        println(io, "window_start = $(first(steps))")
        println(io, "window_stop = $(last(steps))")
        println(io, "gamma_window_start = $(first(gamma_window))")
        println(io, "gamma_window_stop = $(last(gamma_window))")
        println(io, "ct_ceiling = ", values["ct_cycle_mean_relative_difference"])
        println(io, "gamma_max_ceiling = ", values["gamma_m2_max_normalized_difference"])
        println(io, "gamma_rms_ceiling = ", values["gamma_m2_rms_normalized_difference"])
    end
    println("wrote locked tolerance: $path")
end

function mode_gate(reference_dir, candidate_dir, tolerance_path, outdir)
    rows, steps = mode_compare(reference_dir, candidate_dir, outdir)
    values = Dict(row.metric => row.value for row in rows)
    tolerances = TOML.parsefile(tolerance_path)
    checks = [
        ("CT cycle-mean", values["ct_cycle_mean_relative_difference"], tolerances["ct_ceiling"]),
        ("Gamma M2 max", values["gamma_m2_max_normalized_difference"], tolerances["gamma_max_ceiling"]),
        ("Gamma M2 RMS", values["gamma_m2_rms_normalized_difference"], tolerances["gamma_rms_ceiling"]),
    ]
    gatepath = joinpath(outdir, "fm052_gate.md")
    open(gatepath, "w") do io
        println(io, "# fm052 correctness gate\n")
        println(io, "Window: **$(first(steps))–$(last(steps))**  ")
        println(io, "Locked tolerance: `$(abspath(tolerance_path))`\n")
        println(io, "| Gate | Measured | Ceiling | Result |\n|---|---:|---:|---|")
        for (name, measured, ceiling) in checks
            println(io, "| $name | $measured | $ceiling | $(measured <= ceiling ? "PASS" : "FAIL") |")
        end
    end
    failures = filter(check -> check[2] > check[3], checks)
    isempty(failures) || error("correctness gate failed: $failures")
    println("all locked correctness gates passed; wrote $gatepath")
end

const EXCLUSIVE_TIMERS = Set([
    "controls_setup", "wake_influence", "solve", "body_influence",
    "remaining_aerodynamics", "monitors", "io", "wake_propagation_maintenance",
    "rigid_kinematics", "shedding", "total_step", "unclassified_residual",
])

function parse_timers(logpath)
    timers = Dict{Tuple{String,String},Vector{Float64}}()
    source_cpu_s, source_gpu_s, source_backend = 0, 0, 0
    memory = NamedTuple[]
    rx = r"(step_timer_nested|step_timer|gpu_timer)\s+(.+?)\s+([0-9.eE+\-]+)\s+s"
    source_rx = r"(source_influence_s_gpu_gemv|source_influence_s_gemv|source_influence_backend)\s+([0-9.eE+\-]+)\s+s"
    memory_rx = r"source_s_gpu_memory\s+(\S+)\s+total_bytes=(\d+)\s+free_bytes=(\d+)\s+pool_reserved_bytes=(-?\d+)\s+pool_used_bytes=(-?\d+)"
    for line in eachline(logpath)
        source_cpu_s += occursin("source_influence_s_gemv", line)
        source_gpu_s += occursin("source_influence_s_gpu_gemv", line)
        source_backend += occursin("source_influence_backend", line)
        sm = match(source_rx, line)
        if sm !== nothing
            label, rawseconds = sm.captures
            seconds = parse(Float64, rawseconds)
            isfinite(seconds) || error("nonfinite source timer in $logpath: $line")
            push!(get!(timers, ("source", label), Float64[]), seconds)
        end
        mm = match(memory_rx, line)
        if mm !== nothing
            label, total, free, reserved, used = mm.captures
            push!(memory, (; label, total_bytes=parse(Int, total),
                free_bytes=parse(Int, free), pool_reserved_bytes=parse(Int, reserved),
                pool_used_bytes=parse(Int, used)))
        end
        m = match(rx, line)
        m === nothing && continue
        family, rawlabel, rawseconds = m.captures
        label = strip(replace(rawlabel, r"\s*step=\d+" => ""))
        seconds = parse(Float64, rawseconds)
        isfinite(seconds) || error("nonfinite timer in $logpath: $line")
        kind = family == "step_timer_nested" ? "nested" :
               family == "step_timer" && label in EXCLUSIVE_TIMERS ? "exclusive" : "backend"
        push!(get!(timers, (kind, label), Float64[]), seconds)
    end
    return timers, source_cpu_s, source_gpu_s, source_backend, memory
end

function mode_report(run_dir, logpath, outdir=run_dir)
    wh = wake_health(run_dir)
    steps = sort!(collect(keys(wh)))
    isempty(steps) && error("no wake-health samples")
    timers, source_cpu_s, source_gpu_s, source_backend, memory = parse_timers(logpath)
    rows = NamedTuple[]
    for ((kind, label), values) in sort!(collect(timers); by=first)
        push!(rows, (; kind, label, n=length(values), mean=mean(values),
                      median=median(values), total=sum(values)))
    end
    mkpath(outdir)
    trajectory_path = joinpath(outdir, "fm052_step_trajectory.csv")
    open(trajectory_path, "w") do io
        println(io, "step,n_particles,wall_s")
        for step in steps
            println(io, "$(step),$(wh[step].n_particles),$(wh[step].wall_s)")
        end
    end

    metadata_path = unique_file(run_dir, p -> endswith(p, "_case_metadata.toml"),
                                "case metadata")
    metadata = TOML.parsefile(metadata_path)
    s_matrix_bytes = Int(get(metadata, "solver_S_gpu_matrix_bytes", 0))
    s_allocation_bytes = Int(get(metadata, "solver_S_gpu_allocation_bytes", 0))
    pool_reserved_after = Int(get(metadata, "solver_S_gpu_pool_reserved_after_bytes", -1))
    pool_used_after = Int(get(metadata, "solver_S_gpu_pool_used_after_bytes", -1))
    process_files = filter(p -> endswith(p, "_process_wall_s.txt"), readdir(run_dir; join=true))
    process_wall = isempty(process_files) ? NaN :
        sum(parse(Float64, strip(read(path, String))) for path in process_files)
    marching_wall = Float64(metadata["wall_time_s"])
    n_steps = Int(metadata["n_steps"])
    setup_wall = isfinite(process_wall) ? process_wall - marching_wall : NaN
    effective = isfinite(process_wall) ? process_wall / n_steps : NaN
    tail = steps[max(1, end - 9):end]
    tail_mean = mean(wh[s].wall_s for s in tail)
    before_samples = filter(m -> m.label == "before_upload", memory)
    after_samples = filter(m -> m.label == "after_upload", memory)
    gemv_samples = filter(m -> startswith(m.label, "gemv_"), memory)
    startup_usable = isempty(before_samples) ? -1 : first(before_samples).free_bytes
    free_after_upload = isempty(after_samples) ? -1 : last(after_samples).free_bytes
    post_fmm_min_free = isempty(gemv_samples) ? -1 : minimum(m.free_bytes for m in gemv_samples)
    mature_samples = gemv_samples[max(1, end - 9):end]
    mature_tail_min_free = isempty(mature_samples) ? -1 : minimum(m.free_bytes for m in mature_samples)
    summary_path = joinpath(outdir, "fm052_run_summary.csv")
    open(summary_path, "w") do io
        println(io, "metric,value,units")
        println(io, "process_wall,$process_wall,s")
        println(io, "marching_wall,$marching_wall,s")
        println(io, "setup_jit_assembly_wall,$setup_wall,s")
        println(io, "effective_total_per_step,$effective,s_per_step")
        println(io, "mature_tail_mean,$tail_mean,s_per_step")
        println(io, "n_steps,$n_steps,steps")
        println(io, "final_particles,$(wh[last(steps)].n_particles),particles")
        println(io, "gpu_startup_usable,$startup_usable,bytes")
        println(io, "gpu_s_matrix,$s_matrix_bytes,bytes")
        println(io, "gpu_s_allocation,$s_allocation_bytes,bytes")
        println(io, "gpu_free_after_s_upload,$free_after_upload,bytes")
        println(io, "gpu_pool_reserved_after_s_upload,$pool_reserved_after,bytes")
        println(io, "gpu_pool_used_after_s_upload,$pool_used_after,bytes")
        println(io, "gpu_post_fmm_min_free,$post_fmm_min_free,bytes")
        println(io, "gpu_mature_tail_min_free,$mature_tail_min_free,bytes")
        println(io, "under_one_hour,$(isfinite(process_wall) && process_wall < 3600),bool")
    end
    memory_path = joinpath(outdir, "fm052_gpu_memory.csv")
    open(memory_path, "w") do io
        println(io, "label,total_bytes,free_bytes,pool_reserved_bytes,pool_used_bytes")
        for m in memory
            println(io, join((m.label, m.total_bytes, m.free_bytes,
                m.pool_reserved_bytes, m.pool_used_bytes), ','))
        end
    end
    csvpath = joinpath(outdir, "fm052_timer_summary.csv")
    open(csvpath, "w") do io
        println(io, "kind,label,n,mean_s,median_s,total_s")
        for row in rows
            println(io, join((row.kind, row.label, row.n, row.mean, row.median, row.total), ','))
        end
    end
    mdpath = joinpath(outdir, "fm052_timer_summary.md")
    open(mdpath, "w") do io
        println(io, "# fm052 timing summary\n")
        println(io, "Continuation window: **$(first(steps))–$(last(steps))** ($(length(steps)) steps)  ")
        println(io, "Process wall: **$(process_wall) s**  ")
        println(io, "Marching wall: **$(marching_wall) s**  ")
        println(io, "Setup/JIT/assembly wall: **$(setup_wall) s**  ")
        println(io, "Effective total cost: **$(effective) s/step**  ")
        println(io, "Mature tail mean (last $(length(tail)) steps): **$(tail_mean) s/step**  ")
        println(io, "One-hour verdict: **$(isfinite(process_wall) && process_wall < 3600 ? "PASS" : "FAIL/NOT AVAILABLE")**  ")
        println(io, "CPU-S observations: **$source_cpu_s**; GPU-S observations: **$source_gpu_s**; backend source-path observations: **$source_backend**  ")
        println(io, "GPU startup usable memory: **$startup_usable bytes**  ")
        println(io, "Resident S matrix/allocation: **$s_matrix_bytes / $s_allocation_bytes bytes**  ")
        println(io, "GPU free memory after S upload: **$free_after_upload bytes**  ")
        println(io, "CUDA pool used/reserved after S upload: **$pool_used_after / $pool_reserved_after bytes**  ")
        println(io, "GPU minimum free after FMM construction: **$post_fmm_min_free bytes**  ")
        println(io, "GPU mature-tail minimum free: **$mature_tail_min_free bytes**\n")
        println(io, "| Family | Timer | n | Mean (s) | Median (s) | Total (s) |\n|---|---|---:|---:|---:|---:|")
        for row in rows
            mean_string = @sprintf("%.6f", row.mean)
            median_string = @sprintf("%.6f", row.median)
            total_string = @sprintf("%.6f", row.total)
            println(io, "| $(row.kind) | $(row.label) | $(row.n) | $(mean_string) | $(median_string) | $(total_string) |")
        end
        println(io, "\nNested, backend, and source timers are diagnostic subsets and are not added to the exclusive pass total.")
    end
    println("source paths: CPU-S=$source_cpu_s GPU-S=$source_gpu_s backend=$source_backend")
    println("wrote $csvpath")
    println("wrote $mdpath")
    println("wrote $summary_path")
    println("wrote $trajectory_path")
    println("wrote $memory_path")
end

function mode_memory_gate(run_dir, outdir=run_dir, minimum_gib=16.0)
    metadata_path = unique_file(run_dir, p -> endswith(p, "_case_metadata.toml"),
                                "case metadata")
    metadata = TOML.parsefile(metadata_path)
    required = round(Int, minimum_gib * 1024^3)
    enabled = get(metadata, "solver_S_gpu_enabled", false)
    uploads = Int(get(metadata, "solver_S_gpu_upload_count", 0))
    gemvs = Int(get(metadata, "solver_S_gpu_gemv_count", 0))
    after_upload = Int(get(metadata, "solver_S_gpu_free_after_bytes", -1))
    minimum_free = Int(get(metadata, "solver_S_gpu_min_free_bytes", -1))
    # A restarted run only executes steps restart_step+1 .. n_steps-1, so the
    # gemv/sample counts cover the executed window, not the absolute end step.
    restart_step = Int(get(metadata, "restart_step", -1))
    expected_steps = Int(metadata["n_steps"]) - restart_step - 1
    logpath = run_dir * ".log"
    isfile(logpath) || error("mature memory gate: missing log $logpath")
    _, _, _, _, memory = parse_timers(logpath)
    gemv_samples = filter(m -> startswith(m.label, "gemv_"), memory)
    length(gemv_samples) == expected_steps || error(
        "mature memory gate: expected one memory sample per mature step " *
        "($expected_steps), found $(length(gemv_samples))")
    tail_samples = gemv_samples[max(1, end - 9):end]
    post_fmm_min_free = minimum(m.free_bytes for m in gemv_samples)
    mature_tail_min_free = minimum(m.free_bytes for m in tail_samples)
    enabled || error("mature memory gate: GPU-S was not enabled")
    uploads == 1 || error("mature memory gate: expected one S upload, found $uploads")
    gemvs == expected_steps || error(
        "mature memory gate: expected $expected_steps GPU-S gemvs, found $gemvs")
    minimum_free >= required || error(
        "mature memory gate failed: minimum free $minimum_free bytes < $required bytes")
    mature_tail_min_free >= required || error(
        "mature memory gate failed: tail minimum free $mature_tail_min_free bytes < $required bytes")
    mkpath(outdir)
    path = joinpath(outdir, "fm052_memory_gate.md")
    open(path, "w") do io
        println(io, "# fm052 GPU memory gate\n")
        println(io, "- GPU-S enabled: **$enabled**")
        println(io, "- One-time uploads: **$uploads**")
        println(io, "- GPU-S gemvs: **$gemvs / $expected_steps**")
        println(io, "- Free after S upload: **$after_upload bytes**")
        println(io, "- Minimum free after FMM cache construction: **$post_fmm_min_free bytes**")
        println(io, "- Mature-tail minimum free (last $(length(tail_samples)) steps): **$mature_tail_min_free bytes**")
        println(io, "- Global minimum observed free: **$minimum_free bytes**")
        println(io, "- Required mature-tail headroom: **$required bytes ($(minimum_gib) GiB)**")
        println(io, "- Verdict: **PASS**")
    end
    println("GPU-S mature memory gate passed; wrote $path")
end

function pvd_files(path)
    text = read(path, String)
    return [m.captures[1] for m in eachmatch(r"file=\"([^\"]+)\"", text)]
end

function file_index(path)
    m = match(r"\.(\d+)\.vt[ump]$", path)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function mode_verify(run_dir, first_index::Int, last_index::Int)
    readvtk = Base.require(Base.PkgId(
        Base.UUID("dc215faf-f008-4882-a9f7-a79a826fadc3"), "ReadVTK"))
    pvd_paths = [
        unique_file(run_dir, p -> endswith(p, "_body1.pvd"), "body PVD"),
        unique_file(run_dir, p -> endswith(p, "_wake1.pvd"), "wake PVD"),
        unique_file(run_dir, p -> endswith(p, "_wake1_particles.pvd"), "particle PVD"),
    ]
    function load_reference(path)
        if endswith(path, ".vtm")
            leaves = pvd_files(path)
            if isempty(leaves)
                lightxml = getglobal(readvtk, :LightXML)
                document = Base.invokelatest(lightxml.parse_file, path)
                Base.invokelatest(lightxml.free, document)
                println("loaded empty VTM manifest $path")
                return
            end
            for leaf in leaves
                leafpath = normpath(joinpath(dirname(path), leaf))
                isfile(leafpath) || error("VTM reference missing: $leafpath")
                vtk = Base.invokelatest(readvtk.VTKFile, leafpath)
                Base.invokelatest(readvtk.get_points, vtk)
                println("loaded $leafpath")
            end
        else
            vtk = Base.invokelatest(readvtk.VTKFile, path)
            Base.invokelatest(readvtk.get_points, vtk)
            println("loaded $path")
        end
    end
    for pvd in pvd_paths
        refs = pvd_files(pvd)
        for idx in (first_index, last_index)
            hits = filter(ref -> file_index(ref) == idx, refs)
            length(hits) == 1 || error("$pvd does not reference exactly one step-$idx file")
            vtkpath = normpath(joinpath(dirname(pvd), only(hits)))
            isfile(vtkpath) || error("PVD reference missing: $vtkpath")
            load_reference(vtkpath)
        end
    end
    expected = Set(first_index:last_index)
    monitor_dir = joinpath(run_dir, "monitors")
    for path in filter(p -> endswith(p, ".csv"), readdir(monitor_dir; join=true))
        h, rows = read_csv(path)
        "step" in h || continue
        cs = col(h, "step")
        actual = Set(intvalue(row[cs], "$path step") for row in rows)
        actual == expected || error("monitor step coverage mismatch in $path")
    end
    println("artifact and monitor gates passed for indices $first_index:$last_index")
end

function manifest_package_lines(path)
    manifest = TOML.parsefile(path)
    packages = get(manifest, "deps", Dict{String,Any}())
    lines = String[]
    kept_fields = ("uuid", "version", "git-tree-sha1", "repo-url", "repo-rev", "pinned")
    for name in sort!(collect(keys(packages)))
        rawentries = packages[name]
        entries = rawentries isa Vector ? rawentries : [rawentries]
        for entry in entries
            fields = ["$field=$(get(entry, field, ""))" for field in kept_fields]
            deps = sort!(string.(get(entry, "deps", String[])))
            push!(fields, "deps=$(join(deps, ';'))")
            push!(lines, join(vcat(name, fields), '|'))
        end
    end
    return sort!(lines)
end

manifest_package_fingerprint(path) = bytes2hex(sha256(join(manifest_package_lines(path), '\n')))

function mode_manifest_gate(reference_manifest, candidate_manifest, outpath)
    reference_lines = manifest_package_lines(reference_manifest)
    candidate_lines = manifest_package_lines(candidate_manifest)
    reference_lines == candidate_lines || begin
        only_reference = setdiff(reference_lines, candidate_lines)
        only_candidate = setdiff(candidate_lines, reference_lines)
        error("package manifest mismatch: only reference=$(only_reference), only candidate=$(only_candidate)")
    end
    fingerprint = manifest_package_fingerprint(reference_manifest)
    mkpath(dirname(outpath))
    open(outpath, "w") do io
        println(io, "# fm052 package manifest gate\n")
        println(io, "- Reference: `$(abspath(reference_manifest))`")
        println(io, "- Candidate: `$(abspath(candidate_manifest))`")
        println(io, "- Platform-independent package fingerprint: `$fingerprint`")
        println(io, "- Package entries: **$(length(reference_lines))**")
        println(io, "- Verdict: **PASS**")
    end
    println("package manifest gate passed; wrote $outpath")
end

function metric_csv(path)
    h, rows = read_csv(path)
    cm, cv, cu = col(h, "metric"), col(h, "value"), col(h, "units")
    out = Dict{String,Tuple{Float64,String}}()
    for (i, row) in enumerate(rows)
        metric = row[cm]
        haskey(out, metric) && error("duplicate metric $metric in $path")
        value = if row[cu] == "bool"
            parsed = tryparse(Bool, row[cv])
            parsed === nothing && error("invalid boolean sample at $path:$(i+1): $(row[cv])")
            parsed ? 1.0 : 0.0
        else
            finite64(row[cv], "$path:$(i+1)")
        end
        out[metric] = (value, row[cu])
    end
    return out
end

function cross_metrics(manifest)
    report_dir = String(manifest["report_dir"])
    run_dir = String(manifest["run_dir"])
    isdir(report_dir) || error("cross-architecture report directory missing: $report_dir")
    isdir(run_dir) || error("cross-architecture run directory missing: $run_dir")
    metrics = metric_csv(joinpath(report_dir, "fm052_run_summary.csv"))
    comparison = metric_csv(joinpath(report_dir, "fm052_comparison.csv"))
    for (name, value) in comparison
        metrics["correctness_" * name] = value
    end
    timer_path = joinpath(report_dir, "fm052_timer_summary.csv")
    h, rows = read_csv(timer_path)
    ck, cl = col(h, "kind"), col(h, "label")
    cmean, cmedian, ctotal = col(h, "mean_s"), col(h, "median_s"), col(h, "total_s")
    for (i, row) in enumerate(rows)
        prefix = "timer_$(row[ck])_$(row[cl])"
        metrics[prefix * "_mean"] = (finite64(row[cmean], "$timer_path:$(i+1)"), "s")
        metrics[prefix * "_median"] = (finite64(row[cmedian], "$timer_path:$(i+1)"), "s")
        metrics[prefix * "_total"] = (finite64(row[ctotal], "$timer_path:$(i+1)"), "s")
    end
    metadata_path = unique_file(run_dir, p -> endswith(p, "_case_metadata.toml"), "case metadata")
    metadata = TOML.parsefile(metadata_path)
    construction = (
        "solver_total_setup" => "solver_construction_s",
        "solver_G_assembly" => "solver_G_assembly_s",
        "solver_LU" => "solver_LU_s",
        "solver_S_assembly" => "solver_S_assembly_s",
        "solver_S_gpu_upload" => "solver_S_gpu_upload_s",
    )
    for (metric, key) in construction
        haskey(metadata, key) && (metrics[metric] = (Float64(metadata[key]), "s"))
    end
    return metrics
end

csv_field(value) = begin
    s = string(value)
    occursin(r"[,\"\n]", s) ? "\"" * replace(s, '"' => "\"\"") * "\"" : s
end

is_performance_metric(metric, units) =
    units in ("s", "s_per_step") || startswith(metric, "timer_")

function mode_cross_arch(outdir, manifest_paths)
    isempty(manifest_paths) && error("cross-arch needs at least an H200 result manifest")
    results = Dict{String,Dict{String,Any}}()
    for path in manifest_paths
        result = TOML.parsefile(path)
        arch = String(result["architecture"])
        arch in ("h200", "h100", "gh200", "b200", "l40s") || error("invalid architecture in $path: $arch")
        haskey(results, arch) && error("duplicate architecture manifest: $arch")
        result["_manifest_path"] = abspath(path)
        results[arch] = result
    end
    haskey(results, "h200") || error("cross-architecture normalization requires an H200 result")
    String(results["h200"]["status"]) == "pass" || error("H200 reference did not pass")
    metrics = Dict{String,Dict{String,Tuple{Float64,String}}}()
    for (arch, result) in results
        status = String(result["status"])
        if status == "pass"
            String(result["stage"]) == "mature" || error("passing cross-architecture input must be a mature result: $arch")
            metrics[arch] = cross_metrics(result)
        elseif status in ("fail", "ineligible")
            metrics[arch] = Dict{String,Tuple{Float64,String}}()
        else
            error("invalid result status for $arch: $status")
        end
    end
    h200 = metrics["h200"]
    mkpath(outdir)
    csvpath = joinpath(outdir, "fm052_cross_architecture_summary.csv")
    open(csvpath, "w") do io
        println(io, "architecture,status,reason,metric,raw_value,units,h200_value,normalized_to_h200")
        for arch in sort!(collect(keys(results)))
            result = results[arch]
            status, reason = String(result["status"]), String(get(result, "reason", ""))
            if isempty(metrics[arch])
                println(io, join(csv_field.((arch, status, reason, "", "", "", "", "")), ','))
                continue
            end
            for metric in sort!(collect(keys(metrics[arch])))
                raw, units = metrics[arch][metric]
                hraw = haskey(h200, metric) ? first(h200[metric]) : NaN
                normalized = is_performance_metric(metric, units) && isfinite(hraw) && hraw != 0 ? raw / hraw : NaN
                fields = (arch, status, reason, metric, raw, units,
                          isfinite(hraw) ? hraw : "", isfinite(normalized) ? normalized : "")
                println(io, join(csv_field.(fields), ','))
            end
        end
    end
    mdpath = joinpath(outdir, "fm052_cross_architecture_summary.md")
    open(mdpath, "w") do io
        println(io, "# fm052 cross-architecture summary\n")
        println(io, "Performance ratios are raw architecture measurements divided by the H200 measurement; raw values remain in the CSV.\n")
        println(io, "| Architecture | Status | GPU | VRAM (MiB) | CC | Node | Partition | CPU architecture/model | Julia/CUDA | Job | Reason |")
        println(io, "|---|---|---|---:|---|---|---|---|---|---|---|")
        for arch in sort!(collect(keys(results)))
            r = results[arch]
            cpu = "$(get(r, "cpu_architecture", "")) / $(get(r, "cpu_model", ""))"
            runtime = "$(get(r, "julia_version", "see provenance")) / $(get(r, "cuda_runtime", "see provenance"))"
            println(io, "| $arch | $(r["status"]) | $(get(r, "observed_gpu_model", "")) | $(get(r, "observed_gpu_vram_mib", -1)) | $(get(r, "observed_compute_capability", "")) | $(get(r, "node", "")) | $(get(r, "partition", "")) | $cpu | $runtime | $(get(r, "job_id", "")) | $(get(r, "reason", "")) |")
        end
        println(io, "\n## Selected measurements\n")
        println(io, "| Architecture | Process wall (s) | Setup/JIT/assembly (s) | Mature tail (s/step) | GPU-S total (s) | Particles | Tail free (GiB) | Process/H200 | Tail/H200 |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        selected = ("process_wall", "setup_jit_assembly_wall", "mature_tail_mean",
                    "timer_source_source_influence_s_gpu_gemv_total", "final_particles",
                    "gpu_mature_tail_min_free")
        for arch in sort!(collect(keys(results)))
            m = metrics[arch]
            value(key) = haskey(m, key) ? first(m[key]) : NaN
            ratio(key) = haskey(m, key) && haskey(h200, key) && first(h200[key]) != 0 ? first(m[key]) / first(h200[key]) : NaN
            gib = value(selected[6]) / 1024^3
            println(io, "| $arch | $(value(selected[1])) | $(value(selected[2])) | $(value(selected[3])) | $(value(selected[4])) | $(value(selected[5])) | $gib | $(ratio(selected[1])) | $(ratio(selected[3])) |")
        end
        println(io, "\nThe CSV also contains every exclusive/nested/backend/source timer, construction and upload timing, memory metric, particle count, and CT/Γ comparison metric.")
    end
    println("wrote $csvpath")
    println("wrote $mdpath")
end

function main(args)
    isempty(args) && error("usage: compare REF CAND [OUTDIR] | lock REF CAND FIRST LAST OUTDIR | gate REF CAND TOLERANCE OUTDIR | report RUN LOG [OUTDIR] | memory-gate RUN [OUTDIR] [MIN_GIB] | verify RUN FIRST LAST | manifest-gate REF_MANIFEST CAND_MANIFEST OUT | cross-arch OUTDIR RESULT_MANIFEST...")
    if args[1] == "compare"
        length(args) in (3, 4) || error("compare needs REF CAND [OUTDIR]")
        mode_compare(args[2], args[3], length(args) == 4 ? args[4] : args[3])
    elseif args[1] == "lock"
        length(args) == 6 || error("lock needs REF CAND FIRST LAST OUTDIR")
        mode_lock(args[2], args[3], parse(Int, args[4]), parse(Int, args[5]), args[6])
    elseif args[1] == "gate"
        length(args) == 5 || error("gate needs REF CAND TOLERANCE OUTDIR")
        mode_gate(args[2], args[3], args[4], args[5])
    elseif args[1] == "report"
        length(args) in (3, 4) || error("report needs RUN LOG [OUTDIR]")
        mode_report(args[2], args[3], length(args) == 4 ? args[4] : args[2])
    elseif args[1] == "memory-gate"
        length(args) in (2, 3, 4) || error("memory-gate needs RUN [OUTDIR] [MIN_GIB]")
        mode_memory_gate(args[2], length(args) >= 3 ? args[3] : args[2],
            length(args) == 4 ? parse(Float64, args[4]) : 16.0)
    elseif args[1] == "verify"
        length(args) == 4 || error("verify needs RUN FIRST LAST")
        mode_verify(args[2], parse(Int, args[3]), parse(Int, args[4]))
    elseif args[1] == "manifest-gate"
        length(args) == 4 || error("manifest-gate needs REF_MANIFEST CAND_MANIFEST OUT")
        mode_manifest_gate(args[2], args[3], args[4])
    elseif args[1] == "cross-arch"
        length(args) >= 3 || error("cross-arch needs OUTDIR RESULT_MANIFEST...")
        mode_cross_arch(args[2], args[3:end])
    else
        error("unknown mode $(args[1])")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
