# Corrected task-048 synchronized same-state A/B benchmark. Device-only;
# cuda_048_run.sh captures stdout separately and hashes it after completion.
using CUDA, Random, Statistics, SHA
import FLOWVPM
const vpm = FLOWVPM

CUDA.functional() || error("CUDA is not functional")
const N = parse(Int, get(ENV, "FM048_AB_N", "20000"))
const REPS = parse(Int, get(ENV, "FM048_AB_REPS", "9"))
const SEED = parse(Int, get(ENV, "FM048_AB_SEED", "48048"))
const OUT = get(ENV, "FM048_AB_CSV", "fm048_ab_results.csv")
const RHO_CANDIDATES = (4.211, 4.789)
const P018_BIN = get(ENV, "FM048_P018_BIN", "")

function load_snapshot(path)
    isfile(path) || error("FM048_P018_BIN snapshot not found: $path")
    open(path, "r") do io
        nrows = read(io, Int64); n = read(io, Int64)
        nrows == 46 || error("unexpected snapshot row count $nrows")
        A = Matrix{Float64}(undef, nrows, n); read!(io, A)
        return A
    end
end

function build_field(::Type{R}, P, rho_t; snapshot=nothing) where R
    n = snapshot === nothing ? N : size(snapshot, 2)
    rng = MersenneTwister(SEED)
    sigma = R(2 * n^(-1 / 3))
    pf = vpm.ParticleField(n, R; formulation=vpm.rVPM,
        kernel=vpm.gaussianerf, viscous=vpm.Inviscid(), SFS=vpm.noSFS,
        UJ=vpm.UJ_fmm, arraytype=CuArray,
        fmm=vpm.FMM(; p=P, ncrit=50, theta=0.4, autotune_p=false,
            autotune_ncrit=false, autotune_reg_error=false))
    if snapshot === nothing
        for _ in 1:n
            vpm.add_particle(pf, rand(rng, R, 3),
                (R(2) .* rand(rng, R, 3) .- one(R)) ./ R(n), sigma)
        end
    else
        pf.np = n
        pf.particles .= CUDA.CuArray(R.(snapshot))
    end
    vpm.radix_fmm_settings!(pf; expansion_order=P, rho_t)
    return pf
end

gpu_seconds(f) = (CUDA.synchronize(); t0=time_ns(); f(); CUDA.synchronize();
                  (time_ns()-t0) / 1e9)

rows = String[]
push!(rows, "case,p,precision,rho_t,n,seed,reps,uj_median_s,ujsfs_median_s,marginal_s,host_alloc_uj,host_alloc_ujsfs,device_alloc_uj,device_alloc_ujsfs,body_uploads,expansion_host_copies")
specs = Tuple{String,DataType,Int,Float64,Union{Nothing,Matrix{Float64}}}[
        ("cube", R, P, rho, nothing)
         for R in (Float64, Float32) for P in (4, 8)
         for rho in RHO_CANDIDATES]
if !isempty(P018_BIN)
    snap = load_snapshot(P018_BIN)
    # Production p018 is Float64/P4. Test both conservative rho candidates;
    # the synthetic cube above carries the full P/precision/rho matrix.
    append!(specs, [("p018", Float64, 4, rho, snap)
                    for rho in RHO_CANDIDATES])
end
for (case, R, P, rho_t, snapshot) in specs
    pf = build_field(R, P, rho_t; snapshot)
    n = pf.np
    # Compile, establish occupancy, capture, and replay both call shapes.
    for _ in 1:3
        vpm.UJ_fmm(pf; sfs=false)
        vpm.UJ_fmm(pf; sfs=true)
    end
    coupling = vpm._radix_fmm_couplings[pf]
    @info "fm048 resolved configuration" case P R rho_t n SEED REPS settings=coupling.settings cache=(ell=coupling.cache.ell, ell_axes=coupling.cache.ell_axes, box_extent=coupling.cache.box_extent, direct_kernel=coupling.cache.options.direct_kernel, m2l_strategy=coupling.cache.options.m2l_strategy)
    tuj = Float64[]; tsfs = Float64[]
    # Alternating order removes monotonic thermal/order bias while every arm
    # uses the identical field state and synchronized boundaries.
    for rep in 1:REPS
        if isodd(rep)
            push!(tuj, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=false)))
            push!(tsfs, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=true)))
        else
            push!(tsfs, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=true)))
            push!(tuj, gpu_seconds(() -> vpm.UJ_fmm(pf; sfs=false)))
        end
    end
    host_uj = @allocated vpm.UJ_fmm(pf; sfs=false)
    host_sfs = @allocated vpm.UJ_fmm(pf; sfs=true)
    dev_uj = CUDA.@allocated vpm.UJ_fmm(pf; sfs=false)
    dev_sfs = CUDA.@allocated vpm.UJ_fmm(pf; sfs=true)
    c = vpm._radix_fmm_couplings[pf].cache.state.counters
    a, b = median(tuj), median(tsfs)
    push!(rows, join((case, P, string(R), rho_t, n, SEED, REPS, a, b, b-a,
        host_uj, host_sfs, dev_uj, dev_sfs, c.body_uploads,
        c.expansion_host_copies), ','))
    @info "fm048 synchronized A/B" case P R rho_t n a b marginal=b-a tuj tsfs
end
open(OUT, "w") do io
    foreach(line -> println(io, line), rows)
end
println("FM048_AB_CSV=$OUT")
