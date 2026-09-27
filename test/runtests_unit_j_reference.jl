# Independent velocity-gradient diagnostic for the Gaussian-erf vortex kernel.
#
# The primary oracle is deliberately pure: ForwardDiff differentiates the
# mathematical regularized Biot-Savart velocity with respect to the probe
# position.  It does not differentiate UJ_direct! or reuse FLOWVPM's analytic
# dg/dr implementation, so agreement checks both the derivative formula and
# the J storage orientation.
#
# Standalone:
#   julia --project=test test/runtests_unit_j_reference.jl
# Full radix localization sweep (about 25 s on the reference laptop):
#   FLOWVPM_JREF_RADIX=1 julia --project=test test/runtests_unit_j_reference.jl
# Focused rho_t sensitivity sweep (writes CSV when the path is supplied):
#   FLOWVPM_JREF_RHO_SWEEP=1 FLOWVPM_JREF_RHO_CSV=/path/to/results.csv \
#     julia --project=test test/runtests_unit_j_reference.jl

using Test
using ForwardDiff
using LinearAlgebra: norm
using Random: MersenneTwister, rand
using Statistics: quantile

if !isdefined(Main, :FLOWVPM)
    import FLOWVPM
end
const vpm_jref = FLOWVPM

const JREF_SQRT_2_OVER_PI = sqrt(2 / pi)
const JREF_INV_4PI = 1 / (4pi)

# Cancellation-free small-rho branch for
# erf(rho/sqrt(2)) - sqrt(2/pi)*rho*exp(-rho^2/2).  This is the same
# mathematical g used by gaussianerf, not FLOWVPM.g_gauserf itself.
function jref_gaussian_g(rho)
    if abs(rho) < 0.1
        rho2 = rho * rho
        return JREF_SQRT_2_OVER_PI * rho * rho2 *
            (1 / 3 + rho2 * (-1 / 10 + rho2 *
            (1 / 56 + rho2 * (-1 / 432 + rho2 / 4224))))
    end
    return vpm_jref.erf(rho / sqrt(2)) -
        JREF_SQRT_2_OVER_PI * rho * exp(-rho * rho / 2)
end

"Velocity at `x` from one Gaussian-erf vortex particle."
function jref_pair_velocity(x, xs, gamma, sigma)
    dx1 = x[1] - xs[1]
    dx2 = x[2] - xs[2]
    dx3 = x[3] - xs[3]
    r2 = dx1 * dx1 + dx2 * dx2 + dx3 * dx3
    iszero(r2) && return [zero(r2), zero(r2), zero(r2)]
    r = sqrt(r2)
    scale = -JREF_INV_4PI * jref_gaussian_g(r / sigma) / (r2 * r)
    return scale .* [dx2 * gamma[3] - dx3 * gamma[2],
                     dx3 * gamma[1] - dx1 * gamma[3],
                     dx1 * gamma[2] - dx2 * gamma[1]]
end

function jref_field_velocity(x, positions, gammas, sigmas; skip=0)
    u = zeros(eltype(x), 3)
    for source in axes(positions, 2)
        source == skip && continue
        u .+= jref_pair_velocity(x, view(positions, :, source),
            view(gammas, :, source), sigmas[source])
    end
    return u
end

function jref_pfield(maxparticles)
    opts = vpm_jref.FMM(; p=4, ncrit=50, theta=0.4,
        autotune_p=false, autotune_ncrit=false, autotune_reg_error=false)
    return vpm_jref.ParticleField(maxparticles, Float64;
        formulation=vpm_jref.rVPM, kernel=vpm_jref.gaussianerf,
        viscous=vpm_jref.Inviscid(), SFS=vpm_jref.noSFS,
        transposed=true, integration=vpm_jref.rungekutta3,
        UJ=vpm_jref.UJ_direct, fmm=opts)
end

function jref_add!(field, positions, gammas, sigmas; static_index=0)
    for i in axes(positions, 2)
        vpm_jref.add_particle(field, view(positions, :, i),
            view(gammas, :, i), sigmas[i]; static=(i == static_index))
    end
    return field
end

jref_matrix(field, i) = reshape(
    copy(field.particles[vpm_jref.J_INDEX, i]), 3, 3)

function jref_relative_rms(actual, reference)
    return norm(actual - reference) / max(norm(reference), eps(Float64))
end

@testset "Gaussian-erf J: single physical source and probe" begin
    xs = [0.31, -0.27, 0.19]
    gamma = [0.73, -1.11, 0.42]
    sigma = 0.37
    directions = ([1.0, 0.0, 0.0],
                  [0.2, -0.7, 0.5] ./ norm([0.2, -0.7, 0.5]),
                  [-0.4, 0.3, 0.8] ./ norm([-0.4, 0.3, 0.8]))

    # First isolate the scalar ingredients used by the analytic direct J.
    # The looser relative tolerance on g at rho=1e-3 accounts for subtracting
    # two O(rho) Float64 terms to obtain an O(rho^3) value in production.
    for rho in (1e-3, 0.3, 1.0, 3.0, 8.0)
        g, dg = vpm_jref.g_dgdr_gauserf(rho)
        gad = jref_gaussian_g(rho)
        dgad = ForwardDiff.derivative(jref_gaussian_g, rho)
        @test isapprox(g, gad; rtol=2e-6, atol=2e-16)
        @test isapprox(dg, dgad; rtol=2e-12, atol=2e-14)
    end

    # rho=r/sigma spans the near-origin limit, core/transition, and far field.
    for (rho, direction) in zip((1e-3, 0.3, 1.0, 3.0, 8.0),
                                (directions[1], directions[2], directions[3],
                                 directions[2], directions[1]))
        xprobe = xs .+ (rho * sigma) .* direction
        jad = ForwardDiff.jacobian(
            x -> jref_pair_velocity(x, xs, gamma, sigma), xprobe)

        # Keep one physical source and one zero-strength probe in the same
        # direct solve to cross-check self-field dispatch.
        pair = jref_pfield(2)
        vpm_jref.add_particle(pair, xs, gamma, sigma)
        vpm_jref.add_particle(pair, xprobe, zeros(3), sigma)
        vpm_jref.UJ_direct(pair)
        jdirect = jref_matrix(pair, 2)

        err = jref_relative_rms(jdirect, jad)
        @info "single-source J reference" rho err
        @test isapprox(jdirect, jad; rtol=2e-9, atol=2e-11)

        # Guard the documented column-major contract independently of reshape.
        @test pair.particles[vpm_jref.J_INDEX[1:3], 2] ≈ jad[:, 1]
        @test pair.particles[vpm_jref.J_INDEX[4:6], 2] ≈ jad[:, 2]
        @test pair.particles[vpm_jref.J_INDEX[7:9], 2] ≈ jad[:, 3]
    end

    # A literal one-particle self solve must exclude its r=0 self interaction.
    self = jref_pfield(1)
    vpm_jref.add_particle(self, xs, gamma, sigma)
    vpm_jref.UJ_direct(self)
    @test iszero(norm(self.particles[vpm_jref.U_INDEX, 1]))
    @test iszero(norm(self.particles[vpm_jref.J_INDEX, 1]))
end

@testset "Gaussian-erf J: separate source/target UJ_direct API" begin
    xs = [0.31, -0.27, 0.19]
    gamma = [0.73, -1.11, 0.42]
    sigma = 0.37
    xprobe = xs .+ sigma .* [0.2, -0.7, 0.5]
    source = jref_pfield(1)
    target = jref_pfield(1)
    vpm_jref.add_particle(source, xs, gamma, sigma)
    vpm_jref.add_particle(target, xprobe, zeros(3), sigma)

    jad = ForwardDiff.jacobian(
        x -> jref_pair_velocity(x, xs, gamma, sigma), xprobe)
    uref = jref_pair_velocity(xprobe, xs, gamma, sigma)
    vpm_jref.UJ_direct(source, target)

    @test isapprox(target.particles[vpm_jref.U_INDEX, 1], uref;
        rtol=2e-10, atol=2e-12)
    @test isapprox(jref_matrix(target, 1), jad;
        rtol=2e-9, atol=2e-11)
    @test !iszero(norm(target.particles[vpm_jref.J_INDEX, 1]))
end

@testset "Gaussian-erf J: aggregation, reset, and static-target semantics" begin
    positions = [0.08 0.31 0.77 0.54 -0.12 0.91;
                 0.15 0.82 0.24 0.61  0.43 0.06;
                 0.73 0.11 0.49 0.36  0.88 0.57]
    gammas = [ 0.13 -0.17  0.08  0.21 -0.11  0.05;
              -0.09  0.04  0.19 -0.07  0.16 -0.14;
               0.07  0.12 -0.15  0.10  0.03  0.18]
    sigmas = [0.22, 0.31, 0.27, 0.35, 0.24, 0.29]
    field = jref_add!(jref_pfield(6), positions, gammas, sigmas;
        static_index=3)

    jad = [ForwardDiff.jacobian(
        x -> jref_field_velocity(x, positions, gammas, sigmas; skip=i),
        positions[:, i]) for i in axes(positions, 2)]

    # Static targets retain their prior U/J during reset, then receive the new
    # direct contribution. Non-static targets are reset to exactly the field.
    sentinel_u = [0.7, -0.2, 0.4]
    sentinel_j = reshape(collect(0.01:0.01:0.09), 3, 3)
    field.particles[vpm_jref.U_INDEX, :] .= 9.0
    field.particles[vpm_jref.J_INDEX, :] .= 9.0
    field.particles[vpm_jref.U_INDEX, 3] .= sentinel_u
    field.particles[vpm_jref.J_INDEX, 3] .= vec(sentinel_j)
    vpm_jref.UJ_direct(field; reset=true)

    for i in axes(positions, 2)
        expected_u = jref_field_velocity(positions[:, i], positions,
            gammas, sigmas; skip=i)
        if i == 3
            expected_u .+= sentinel_u
            expected_j = jad[i] + sentinel_j
        else
            expected_j = jad[i]
        end
        @test isapprox(field.particles[vpm_jref.U_INDEX, i], expected_u;
            rtol=2e-10, atol=2e-12)
        @test isapprox(jref_matrix(field, i), expected_j;
            rtol=2e-9, atol=2e-11)
    end

    before_u = copy(field.particles[vpm_jref.U_INDEX, :])
    before_j = copy(field.particles[vpm_jref.J_INDEX, :])
    vpm_jref.UJ_direct(field; reset=false)
    for i in axes(positions, 2)
        expected_u = jref_field_velocity(positions[:, i], positions,
            gammas, sigmas; skip=i)
        @test field.particles[vpm_jref.U_INDEX, i] ≈ before_u[:, i] + expected_u
        @test jref_matrix(field, i) ≈ reshape(before_j[:, i], 3, 3) + jad[i]
    end
end

const JREF_RUN_RADIX = get(ENV, "FLOWVPM_JREF_RADIX", "0") == "1"
const JREF_RUN_RHO_SWEEP = get(ENV, "FLOWVPM_JREF_RHO_SWEEP", "0") == "1"

if vpm_jref._FMM_HAS_RADIX && (JREF_RUN_RADIX || JREF_RUN_RHO_SWEEP)
    @testset "Gaussian-erf J: radix localization against AD" begin
        rng = MersenneTwister(48117)
        n = 96
        positions = rand(rng, 3, n)
        gammas = (2 .* rand(rng, 3, n) .- 1) ./ n
        # Keep rho_t*sigma below the q=6 near-set geometric gap at ell=2;
        # otherwise the lifecycle correctly rejects the configuration before
        # any accuracy comparison can be made.
        sigmas = fill(0.08, n)
        jad = [ForwardDiff.jacobian(
            x -> jref_field_velocity(x, positions, gammas, sigmas; skip=i),
            positions[:, i]) for i in 1:n]
        jref = reduce(hcat, vec.(jad))

        if JREF_RUN_RADIX
            errors = Dict{Symbol,Float64}()
            cases = ((:P4_q6, 4, 6, :partitioned, nothing),
                     (:P8_q6, 8, 6, :partitioned, nothing),
                     (:P4_q20, 4, 20, :partitioned, nothing),
                     (:P8_q20, 8, 20, :partitioned, nothing),
                     # The shipped partitioned kernel switches to the singular
                     # law at rho_t=3.668. Extending the exact regularized law to
                     # rho=8 separates that cutoff error from FMM truncation.
                     (:P8_q20_rho8, 8, 20, :partitioned, 8.0))
            for (label, P, q, direct_kernel, rho_t) in cases
                field = jref_add!(jref_pfield(n), positions, gammas, sigmas)
                vpm_jref.radix_fmm_settings!(field;
                    expansion_order=P, ell=2, near_radius2=q,
                    direct_kernel=direct_kernel, rho_t=rho_t)
                vpm_jref.UJ_fmm_gpu!(field)
                jradix = Array(field.particles[vpm_jref.J_INDEX, 1:n])
                errors[label] = jref_relative_rms(jradix, jref)
                @info "radix J vs independent AD" label P q direct_kernel rho_t error=errors[label]
                @test errors[label] < 5e-2
            end

            # Diagnostic localization: report which lever matters. These are not
            # accuracy gates; the test's purpose is to distinguish derivative-law
            # errors from FMM truncation/direct-list errors without overfitting a
            # tiny synthetic case.
            @info "radix J localization ratios" (
                P8_over_P4_q6=errors[:P8_q6] / errors[:P4_q6],
                q20_over_q6_P4=errors[:P4_q20] / errors[:P4_q6],
                q20_over_q6_P8=errors[:P8_q20] / errors[:P8_q6],
                rho8_over_shipped=errors[:P8_q20_rho8] / errors[:P8_q20])
        end

        if JREF_RUN_RHO_SWEEP
            # Hold the particle state and every radix control except rho_t
            # fixed. The target-wise metric is Frobenius relative error; its
            # denominator floor only protects genuinely near-zero reference J.
            ref_norms = [norm(jad[i]) for i in 1:n]
            denominator_floor = sqrt(eps(Float64)) * maximum(ref_norms)
            # Include the theory-derived epsilon=1e-3 cutoffs exactly:
            # U RMS=3.668, U per-pair=4.211, J RMS=4.252, J per-pair=4.789.
            rho_values = sort!(unique!(vcat(collect(3.0:0.5:8.0),
                3.668, 4.211, 4.252, 4.789)))
            rows = NamedTuple[]
            for rho_t in rho_values
                field = jref_add!(jref_pfield(n), positions, gammas, sigmas)
                vpm_jref.radix_fmm_settings!(field;
                    expansion_order=8, ell=2, near_radius2=20,
                    direct_kernel=:partitioned, rho_t=rho_t)
                vpm_jref.UJ_fmm_gpu!(field)
                jradix = Array(field.particles[vpm_jref.J_INDEX, 1:n])
                target_errors = [norm(jradix[:, i] - vec(jad[i])) /
                    max(ref_norms[i], denominator_floor) for i in 1:n]
                row = (rho_t=rho_t,
                       global_relative_rms=jref_relative_rms(jradix, jref),
                       max_target_relative=maximum(target_errors),
                       p95_target_relative=quantile(target_errors, 0.95))
                push!(rows, row)
                @info "rho_t J sensitivity" row... denominator_floor
                @test row.global_relative_rms < 5e-2
            end

            csv_path = get(ENV, "FLOWVPM_JREF_RHO_CSV", "")
            if !isempty(csv_path)
                mkpath(dirname(csv_path))
                open(csv_path, "w") do io
                    println(io, "rho_t,global_relative_rms,max_target_frobenius_relative,p95_target_frobenius_relative")
                    for row in rows
                        println(io, join((row.rho_t, row.global_relative_rms,
                            row.max_target_relative, row.p95_target_relative), ','))
                    end
                end
                @info "wrote rho_t sensitivity CSV" csv_path
            end
        end
    end
elseif JREF_RUN_RADIX || JREF_RUN_RHO_SWEEP
    @info "FastMultipole radix lifecycle unavailable; skipping radix J localization"
end
