"""
Compare two implementations of _accumulate_and_finalize_root!:
  - :stack  — 20 stack-local scalar accumulators (current approach)
  - :vector — a single heap-allocated Vector{Float64} of length 20

Run from the repo root with:
    julia scripts/benchmark_accumulate.jl
"""

import Pkg
# Activate a throw-away env layered on top of the root project so we can use
# BenchmarkTools without modifying Project.toml.
Pkg.activate(; temp=true)
Pkg.develop(; path=(@__DIR__) * "/..")
Pkg.add("BenchmarkTools")

using BenchmarkTools
using Printf
using Random
import FLOWVPM
const vpm = FLOWVPM

# ---------------------------------------------------------------------------
# Helpers shared by both variants
# ---------------------------------------------------------------------------

const ACC_GAMMA_X              = 1
const ACC_GAMMA_Y              = 2
const ACC_GAMMA_Z              = 3
const ACC_X_WEIGHTED_X         = 4
const ACC_X_WEIGHTED_Y         = 5
const ACC_X_WEIGHTED_Z         = 6
const ACC_C_WEIGHTED_X         = 7
const ACC_C_WEIGHTED_Y         = 8
const ACC_C_WEIGHTED_Z         = 9
const ACC_X_UNWEIGHTED_X       = 10
const ACC_X_UNWEIGHTED_Y       = 11
const ACC_X_UNWEIGHTED_Z       = 12
const ACC_C_UNWEIGHTED_X       = 13
const ACC_C_UNWEIGHTED_Y       = 14
const ACC_C_UNWEIGHTED_Z       = 15
const ACC_WEIGHT_SUM           = 16
const ACC_VOL_SUM              = 17
const ACC_SIGMA3_SUM           = 18
const ACC_CIRCULATION_WEIGHTED = 19
const ACC_SIGMA_SUM            = 20

# ---------------------------------------------------------------------------
# Vector variant: allocates a fresh Vector{Float64}(undef, 20) each call
# ---------------------------------------------------------------------------

function _accumulate_and_finalize_root_vector!(
    pfield::vpm.ParticleField,
    candidates_by_root::Vector{Int},
    range_start::Int,
    range_end::Int,
    representative::Int,
)
    R = eltype(pfield.particles)
    zeroR = zero(R)

    acc = zeros(R, 20)

    for k in range_start:range_end
        i = candidates_by_root[k]

        gamma_i_x = pfield.particles[vpm.GAMMA_INDEX.start,     i]
        gamma_i_y = pfield.particles[vpm.GAMMA_INDEX.start + 1, i]
        gamma_i_z = pfield.particles[vpm.GAMMA_INDEX.start + 2, i]
        pos_x     = pfield.particles[vpm.X_INDEX.start,          i]
        pos_y     = pfield.particles[vpm.X_INDEX.start + 1,      i]
        pos_z     = pfield.particles[vpm.X_INDEX.start + 2,      i]
        c_i_x     = pfield.particles[vpm.C_INDEX.start,          i]
        c_i_y     = pfield.particles[vpm.C_INDEX.start + 1,      i]
        c_i_z     = pfield.particles[vpm.C_INDEX.start + 2,      i]
        gamma_mag = sqrt(gamma_i_x^2 + gamma_i_y^2 + gamma_i_z^2)
        sigma     = pfield.particles[vpm.SIGMA_INDEX, i]

        acc[ACC_GAMMA_X]              += gamma_i_x
        acc[ACC_GAMMA_Y]              += gamma_i_y
        acc[ACC_GAMMA_Z]              += gamma_i_z
        acc[ACC_VOL_SUM]              += pfield.particles[vpm.VOL_INDEX, i]
        acc[ACC_SIGMA3_SUM]           += sigma^3
        acc[ACC_CIRCULATION_WEIGHTED] += sigma * pfield.particles[vpm.CIRCULATION_INDEX, i]
        acc[ACC_SIGMA_SUM]            += sigma
        acc[ACC_X_UNWEIGHTED_X]       += pos_x
        acc[ACC_X_UNWEIGHTED_Y]       += pos_y
        acc[ACC_X_UNWEIGHTED_Z]       += pos_z
        acc[ACC_C_UNWEIGHTED_X]       += c_i_x
        acc[ACC_C_UNWEIGHTED_Y]       += c_i_y
        acc[ACC_C_UNWEIGHTED_Z]       += c_i_z

        if gamma_mag > zeroR
            acc[ACC_WEIGHT_SUM]   += gamma_mag
            acc[ACC_X_WEIGHTED_X] += gamma_mag * pos_x
            acc[ACC_X_WEIGHTED_Y] += gamma_mag * pos_y
            acc[ACC_X_WEIGHTED_Z] += gamma_mag * pos_z
            acc[ACC_C_WEIGHTED_X] += gamma_mag * c_i_x
            acc[ACC_C_WEIGHTED_Y] += gamma_mag * c_i_y
            acc[ACC_C_WEIGHTED_Z] += gamma_mag * c_i_z
        end
    end

    n_members = range_end - range_start + 1
    vpm._finalize_merged_particle!(
        pfield, representative, n_members,
        acc[ACC_GAMMA_X],              acc[ACC_GAMMA_Y],              acc[ACC_GAMMA_Z],
        acc[ACC_X_WEIGHTED_X],         acc[ACC_X_WEIGHTED_Y],         acc[ACC_X_WEIGHTED_Z],
        acc[ACC_C_WEIGHTED_X],         acc[ACC_C_WEIGHTED_Y],         acc[ACC_C_WEIGHTED_Z],
        acc[ACC_X_UNWEIGHTED_X],       acc[ACC_X_UNWEIGHTED_Y],       acc[ACC_X_UNWEIGHTED_Z],
        acc[ACC_C_UNWEIGHTED_X],       acc[ACC_C_UNWEIGHTED_Y],       acc[ACC_C_UNWEIGHTED_Z],
        acc[ACC_WEIGHT_SUM],
        acc[ACC_VOL_SUM],
        acc[ACC_SIGMA3_SUM],
        acc[ACC_CIRCULATION_WEIGHTED],
        acc[ACC_SIGMA_SUM],
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Preallocated-vector variant: caller owns the buffer; function fill!s and reuses
# ---------------------------------------------------------------------------

function _accumulate_and_finalize_root_prealloc!(
    pfield::vpm.ParticleField,
    candidates_by_root::Vector{Int},
    range_start::Int,
    range_end::Int,
    representative::Int,
    acc::Vector,             # preallocated length-20 scratch buffer
)
    R = eltype(pfield.particles)
    zeroR = zero(R)
    fill!(acc, zeroR)

    for k in range_start:range_end
        i = candidates_by_root[k]

        gamma_i_x = pfield.particles[vpm.GAMMA_INDEX.start,     i]
        gamma_i_y = pfield.particles[vpm.GAMMA_INDEX.start + 1, i]
        gamma_i_z = pfield.particles[vpm.GAMMA_INDEX.start + 2, i]
        pos_x     = pfield.particles[vpm.X_INDEX.start,          i]
        pos_y     = pfield.particles[vpm.X_INDEX.start + 1,      i]
        pos_z     = pfield.particles[vpm.X_INDEX.start + 2,      i]
        c_i_x     = pfield.particles[vpm.C_INDEX.start,          i]
        c_i_y     = pfield.particles[vpm.C_INDEX.start + 1,      i]
        c_i_z     = pfield.particles[vpm.C_INDEX.start + 2,      i]
        gamma_mag = sqrt(gamma_i_x^2 + gamma_i_y^2 + gamma_i_z^2)
        sigma     = pfield.particles[vpm.SIGMA_INDEX, i]

        acc[ACC_GAMMA_X]              += gamma_i_x
        acc[ACC_GAMMA_Y]              += gamma_i_y
        acc[ACC_GAMMA_Z]              += gamma_i_z
        acc[ACC_VOL_SUM]              += pfield.particles[vpm.VOL_INDEX, i]
        acc[ACC_SIGMA3_SUM]           += sigma^3
        acc[ACC_CIRCULATION_WEIGHTED] += sigma * pfield.particles[vpm.CIRCULATION_INDEX, i]
        acc[ACC_SIGMA_SUM]            += sigma
        acc[ACC_X_UNWEIGHTED_X]       += pos_x
        acc[ACC_X_UNWEIGHTED_Y]       += pos_y
        acc[ACC_X_UNWEIGHTED_Z]       += pos_z
        acc[ACC_C_UNWEIGHTED_X]       += c_i_x
        acc[ACC_C_UNWEIGHTED_Y]       += c_i_y
        acc[ACC_C_UNWEIGHTED_Z]       += c_i_z

        if gamma_mag > zeroR
            acc[ACC_WEIGHT_SUM]   += gamma_mag
            acc[ACC_X_WEIGHTED_X] += gamma_mag * pos_x
            acc[ACC_X_WEIGHTED_Y] += gamma_mag * pos_y
            acc[ACC_X_WEIGHTED_Z] += gamma_mag * pos_z
            acc[ACC_C_WEIGHTED_X] += gamma_mag * c_i_x
            acc[ACC_C_WEIGHTED_Y] += gamma_mag * c_i_y
            acc[ACC_C_WEIGHTED_Z] += gamma_mag * c_i_z
        end
    end

    n_members = range_end - range_start + 1
    vpm._finalize_merged_particle!(
        pfield, representative, n_members,
        acc[ACC_GAMMA_X],              acc[ACC_GAMMA_Y],              acc[ACC_GAMMA_Z],
        acc[ACC_X_WEIGHTED_X],         acc[ACC_X_WEIGHTED_Y],         acc[ACC_X_WEIGHTED_Z],
        acc[ACC_C_WEIGHTED_X],         acc[ACC_C_WEIGHTED_Y],         acc[ACC_C_WEIGHTED_Z],
        acc[ACC_X_UNWEIGHTED_X],       acc[ACC_X_UNWEIGHTED_Y],       acc[ACC_X_UNWEIGHTED_Z],
        acc[ACC_C_UNWEIGHTED_X],       acc[ACC_C_UNWEIGHTED_Y],       acc[ACC_C_UNWEIGHTED_Z],
        acc[ACC_WEIGHT_SUM],
        acc[ACC_VOL_SUM],
        acc[ACC_SIGMA3_SUM],
        acc[ACC_CIRCULATION_WEIGHTED],
        acc[ACC_SIGMA_SUM],
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Build a particle field and synthetic cluster for benchmarking
# ---------------------------------------------------------------------------

function make_test_field(n::Int; rng=MersenneTwister(42), sigma=0.05)
    pfield = vpm.ParticleField(n)
    for _ in 1:n
        x     = rand(rng, 3)
        gamma = randn(rng, 3)
        vpm.add_particle(pfield, x, gamma, sigma)
    end
    return pfield
end

# Pretend all particles in 1:n form a single cluster, representative = 1.
function make_flat_cluster(n::Int)
    candidates = collect(1:n)   # candidates_by_root for one cluster
    range_start = 1
    range_end   = n
    representative = 1
    return candidates, range_start, range_end, representative
end

# ---------------------------------------------------------------------------
# Run benchmarks for several cluster sizes
# ---------------------------------------------------------------------------

println("=" ^ 60)
println("Benchmark: stack scalars vs heap vector (20 floats)")
println("=" ^ 60)

R = eltype(make_test_field(1).particles)
prealloc_buf = zeros(R, 20)

for cluster_size in [2, 5, 10, 50, 200]
    n_particles = max(cluster_size, 100)   # field must have enough particles
    pfield      = make_test_field(n_particles)
    cands, rs, re, rep = make_flat_cluster(cluster_size)

    # Warmup
    vpm._accumulate_and_finalize_root!(pfield, cands, rs, re, rep)
    _accumulate_and_finalize_root_vector!(pfield, cands, rs, re, rep)
    _accumulate_and_finalize_root_prealloc!(pfield, cands, rs, re, rep, prealloc_buf)

    b_stack = @benchmark vpm._accumulate_and_finalize_root!(
        $pfield, $cands, $rs, $re, $rep) samples=2000 evals=5

    b_vec = @benchmark _accumulate_and_finalize_root_vector!(
        $pfield, $cands, $rs, $re, $rep) samples=2000 evals=5

    b_prealloc = @benchmark _accumulate_and_finalize_root_prealloc!(
        $pfield, $cands, $rs, $re, $rep, $prealloc_buf) samples=2000 evals=5

    t_stack    = median(b_stack).time
    t_vec      = median(b_vec).time
    t_prealloc = median(b_prealloc).time

    println("\nCluster size: $cluster_size particles")
    @printf "  stack    : %8.1f ns   allocs=%d\n" t_stack    b_stack.allocs
    @printf "  vec/alloc: %8.1f ns   allocs=%d\n" t_vec      b_vec.allocs
    @printf "  prealloc : %8.1f ns   allocs=%d\n" t_prealloc b_prealloc.allocs
    @printf "  prealloc / stack = %.2fx\n" (t_prealloc / t_stack)
end
