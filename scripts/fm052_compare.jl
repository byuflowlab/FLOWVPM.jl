# fm052_compare.jl — task 052 stage-B comparison and stage-C report.
# Stdlib-only (no CSV dep). Two modes:
#
#   julia fm052_compare.jl compare <cpu_run_dir> <gpu_run_dir>
#       Compares the 018 driver's outputs between the CPU and GPU arms over
#       the overlapping steps: CT (…_CT_vs_rev.csv), bound circulation
#       Gamma(r/R) (…bound_circulation… monitor CSV), and per-step wall time
#       + particle counts (…wake_health… monitor CSV).
#
#   julia fm052_compare.jl report <gpu_run_dir> [driver_log]
#       Stage-C report: per-step wall-time trajectory vs n_particles, mean of
#       the last 10 steps, extrapolated 30-rev (1080-step) wall time (naive
#       and linearly rescaled to the production 181k-particle count), and an
#       aggregate of the env-gated "gpu_timer" pass/solve lines in the log.

function read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty csv: $path")
    header = strip.(split(lines[1], ','))
    rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]
    return header, rows
end

function monitor_csv(run_dir::AbstractString, stem::AbstractString)
    mdir = joinpath(run_dir, "monitors")
    isdir(mdir) || error("no monitors dir in $run_dir")
    hits = sort(filter(f -> occursin(stem, f), readdir(mdir)))
    isempty(hits) && error("no '$stem' monitor csv in $mdir")
    length(hits) > 1 && println("  (note: $(length(hits)) '$stem' files, using $(hits[1]))")
    return joinpath(mdir, hits[1])
end

col(header, name) = something(findfirst(==(name), header),
    error("column $name not in $(header)"))

f64(s) = parse(Float64, strip(s))

"step => (wall_s, n_particles) from the wake_health monitor csv"
function wake_health(run_dir)
    h, rows = read_csv(monitor_csv(run_dir, "wake_health"))
    cs, cw, cn = col(h, "step"), col(h, "wall_s"), col(h, "n_particles")
    return Dict(round(Int, f64(r[cs])) => (f64(r[cw]), round(Int, f64(r[cn])))
                for r in rows)
end

"step => CT_bernoulli from the driver's CT_vs_rev csv (written at run end)"
function ct_history(run_dir)
    name = basename(abspath(run_dir))
    path = joinpath(run_dir, "$(name)_CT_vs_rev.csv")
    isfile(path) || error("missing $path (did the run finish?)")
    h, rows = read_csv(path)
    cs, cc = col(h, "step"), col(h, "CT_bernoulli")
    return Dict(round(Int, f64(r[cs])) => f64(r[cc]) for r in rows)
end

"(step, blade, section) => (r_over_R, circulation_te)"
function bound_circulation(run_dir)
    h, rows = read_csv(monitor_csv(run_dir, "bound_circulation"))
    cs, cb, cn = col(h, "step"), col(h, "blade"), col(h, "section")
    cr, cg = col(h, "r_over_R"), col(h, "circulation_te")
    return Dict((round(Int, f64(r[cs])), round(Int, f64(r[cb])),
                 round(Int, f64(r[cn]))) => (f64(r[cr]), f64(r[cg]))
                for r in rows)
end

reldiff(a, b) = abs(a - b) / max(abs(b), 1e-30)

function mode_compare(cpu_dir, gpu_dir)
    println("== fm052 stage-B comparison: CPU=$(cpu_dir)  GPU=$(gpu_dir) ==")

    # ---- CT ----
    ct_c, ct_g = ct_history(cpu_dir), ct_history(gpu_dir)
    steps = sort(collect(intersect(keys(ct_c), keys(ct_g))))
    isempty(steps) && error("no overlapping CT steps")
    println("\n-- CT_bernoulli over $(length(steps)) overlapping steps --")
    println("step | CT_cpu | CT_gpu | absdiff | reldiff")
    maxrel, maxrel_step = 0.0, steps[1]
    for s in steps
        rd = reldiff(ct_g[s], ct_c[s])
        rd > maxrel && ((maxrel, maxrel_step) = (rd, s))
        (s % 6 == 0 || s == steps[end]) && println("  $s | $(ct_c[s]) | $(ct_g[s]) | " *
            "$(abs(ct_g[s] - ct_c[s])) | $(round(rd, sigdigits=3))")
    end
    println("  max rel diff = $(round(maxrel, sigdigits=4)) at step $maxrel_step")

    # ---- Gamma(r/R) ----
    bc_c, bc_g = bound_circulation(cpu_dir), bound_circulation(gpu_dir)
    keys_both = sort(collect(intersect(keys(bc_c), keys(bc_g))))
    if !isempty(keys_both)
        maxrel_g, key_g = 0.0, keys_both[1]
        for k in keys_both
            rd = reldiff(bc_g[k][2], bc_c[k][2])
            rd > maxrel_g && ((maxrel_g, key_g) = (rd, k))
        end
        last_step = maximum(k[1] for k in keys_both)
        println("\n-- bound circulation Gamma_te: $(length(keys_both)) " *
            "(step,blade,section) samples --")
        println("  max rel diff = $(round(maxrel_g, sigdigits=4)) at " *
            "(step,blade,section)=$(key_g)")
        println("  final step $(last_step) Gamma(r/R), blade 1:")
        println("  r/R | Gamma_cpu | Gamma_gpu | reldiff")
        for k in filter(k -> k[1] == last_step && k[2] == 1, keys_both)
            println("  $(round(bc_c[k][1], digits=4)) | $(bc_c[k][2]) | " *
                "$(bc_g[k][2]) | $(round(reldiff(bc_g[k][2], bc_c[k][2]), sigdigits=3))")
        end
    end

    # ---- wall time ----
    wh_c, wh_g = wake_health(cpu_dir), wake_health(gpu_dir)
    wsteps = sort(collect(intersect(keys(wh_c), keys(wh_g))))
    if !isempty(wsteps)
        println("\n-- per-step wall time (wake_health wall_s) --")
        println("step | np_cpu | np_gpu | s/step cpu | s/step gpu | speedup")
        for s in wsteps
            (s % 6 == 0 || s == wsteps[end]) && println("  $s | $(wh_c[s][2]) | " *
                "$(wh_g[s][2]) | $(round(wh_c[s][1], digits=2)) | " *
                "$(round(wh_g[s][1], digits=2)) | " *
                "$(round(wh_c[s][1] / max(wh_g[s][1], 1e-9), digits=2))x")
        end
        tc = sum(wh_c[s][1] for s in wsteps)
        tg = sum(wh_g[s][1] for s in wsteps)
        println("  totals over $(length(wsteps)) steps: cpu $(round(tc, digits=1)) s, " *
            "gpu $(round(tg, digits=1)) s, speedup $(round(tc / max(tg, 1e-9), digits=2))x")
    end
    return nothing
end

function mode_report(gpu_dir, logpath)
    println("== fm052 stage-C report: $(gpu_dir) ==")
    wh = wake_health(gpu_dir)
    steps = sort(collect(keys(wh)))
    isempty(steps) && error("no wake_health rows")
    println("\n-- per-step trajectory --")
    println("step | n_particles | s/step")
    for s in steps
        println("  $s | $(wh[s][2]) | $(round(wh[s][1], digits=3))")
    end
    tail = steps[max(1, end - 9):end]
    tail_mean = sum(wh[s][1] for s in tail) / length(tail)
    println("\nmean s/step over final $(length(tail)) steps: $(round(tail_mean, digits=3))")

    n30 = 30 * 36
    println("extrapolated 30-rev ($(n30)-step) wall, naive (tail mean): " *
        "$(round(tail_mean * n30 / 3600, digits=2)) h")

    # linear rescale of the tail to the production-maturity particle count
    half = steps[max(1, end - div(length(steps), 2)):end]
    xs = [Float64(wh[s][2]) for s in half]
    ys = [wh[s][1] for s in half]
    if length(half) >= 3 && maximum(xs) > minimum(xs)
        mx, my = sum(xs) / length(xs), sum(ys) / length(ys)
        b = sum((xs .- mx) .* (ys .- my)) / sum(abs2, xs .- mx)
        a = my - b * mx
        np_prod = 181_000.0
        s_prod = a + b * np_prod
        println("linear fit s/step = $(round(a, sigdigits=4)) + " *
            "$(round(b, sigdigits=4)) * np  (over final $(length(half)) steps)")
        println("predicted s/step at production np=181k: $(round(s_prod, digits=3)) " *
            "-> 30-rev wall $(round(s_prod * n30 / 3600, digits=2)) h " *
            "(budget target <= 3.3 s/step)")
    end

    # gpu_timer aggregation from the driver log
    if logpath !== nothing && isfile(logpath)
        agg = Dict{String,Vector{Float64}}()
        for line in eachline(logpath)
            i = findfirst("gpu_timer", line)
            i === nothing && continue
            toks = split(strip(line[first(i):end]))
            # "gpu_timer <label...> <seconds> s"
            length(toks) >= 3 || continue
            secs = tryparse(Float64, toks[end-1])
            secs === nothing && continue
            label = join(toks[2:end-2], " ")
            # collapse per-step solve labels
            label = replace(label, r" step=\d+" => "")
            push!(get!(agg, label, Float64[]), secs)
        end
        println("\n-- gpu_timer aggregate ($(logpath)) --")
        for (label, v) in sort(collect(agg); by=first)
            sv = sort(v)
            println("  $(label): n=$(length(v)) median=$(round(sv[cld(end,2)], digits=3)) " *
                "mean=$(round(sum(v)/length(v), digits=3)) total=$(round(sum(v), digits=1)) s")
        end
        isempty(agg) && println("  (no gpu_timer lines found — was FLOWPANEL_GPU_TIMERS=1 set?)")
    end
    return nothing
end

function main(args)
    length(args) >= 2 || error("usage: fm052_compare.jl compare <cpu_dir> <gpu_dir> | report <gpu_dir> [log]")
    if args[1] == "compare"
        length(args) == 3 || error("compare needs <cpu_dir> <gpu_dir>")
        mode_compare(args[2], args[3])
    elseif args[1] == "report"
        mode_report(args[2], length(args) >= 3 ? args[3] : nothing)
    else
        error("unknown mode $(args[1])")
    end
end

main(ARGS)
