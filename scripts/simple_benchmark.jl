using Random
using FLOWVPM

const vpm = FLOWVPM
const fmm = FLOWVPM.FastMultipole

"""
    build_random_particle_field(nparticles; rng=Random.default_rng(), sigma=0.05,
                                gamma_scale=1.0, fmm_options=vpm.FMM())

Build a `FLOWVPM.ParticleField` containing `nparticles` particles with positions
sampled uniformly over the unit cube.
"""
function build_random_particle_field(
    nparticles::Integer;
    rng=Random.default_rng(),
    sigma::Real=0.05,
    gamma_scale::Real=1.0,
    fmm_options::vpm.FMM=vpm.FMM(),
)
    nparticles < 0 && throw(ArgumentError("nparticles must be nonnegative"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    pfield = vpm.ParticleField(
        nparticles;
        fmm=fmm_options,
        UJ=vpm.UJ_fmm,
    )

    for _ in 1:nparticles
        x = rand(rng, 3)
        gamma = gamma_scale .* randn(rng, 3)
        vpm.add_particle(pfield, x, gamma, sigma)
    end

    return pfield
end

"""
    benchmark_fmm!(pfield; verbose=false, reset=true)

Call `FLOWVPM.FastMultipole.fmm!` on `pfield`, measure the elapsed wall-clock
time with `@elapsed`, and return the elapsed time in seconds.
"""
function benchmark_fmm!(pfield::vpm.ParticleField; verbose::Bool=false, reset::Bool=true)
    if reset
        vpm._reset_particles(pfield)
    end

    fmm_options = pfield.fmm

    elapsed = @elapsed optargs, _ = fmm.fmm!(
        pfield;
        expansion_order=fmm_options.p - 1,
        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit),
        multipole_acceptance=fmm_options.theta,
        error_tolerance=fmm.PowerRelativeGradient{
            fmm_options.relative_tolerance,
            fmm_options.absolute_tolerance,
            true,
        }(),
        tune=true,
        shrink_recenter=fmm_options.shrink_recenter,
        nearfield_device=(pfield.useGPU > 0),
        scalar_potential=false,
        hessian=true,
        silence_warnings=!verbose,
    )

    return elapsed, optargs
end

nparticles = 10_000
pfield = build_random_particle_field(
    nparticles;
    fmm_options=vpm.FMM(),
)

elapsed, optargs = benchmark_fmm!(pfield)
elapsed, optargs = benchmark_fmm!(pfield)
elapsed, optargs = benchmark_fmm!(pfield)

@show elapsed
@show optargs
