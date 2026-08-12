# Correctness tests for the radix FMM coupling (task 034): ParticleField
# driving FastMultipole's resident radix lifecycle.
#
# Part A (CPU, no GPU required): the transfer-based host-resident coupling
#   (`UJ_fmm_gpu!` on a Matrix-backed field) against FLOWVPM's own `UJ_direct`
#   on both Integration Phase cases (random unit cube at overlap 2; helical
#   wake cylinder, 033 definition), plus semantics tests (accumulate, cache
#   reuse, automatic recenter, varying particle count, loud-error paths).
#   Skipped with an @info message when the installed FastMultipole lacks the
#   radix device interface (registry releases).
#
# Part B (CUDA): the device-resident coupling end-to-end on a CuArray-backed
#   field — `UJ_fmm` routes to the resident lifecycle with zero per-step body
#   transfer (task 023 counter contract), Float64 and Float32, static
#   evaluation plus a multi-step RK3 dynamic run vs a CPU `UJ_direct`
#   reference. Auto-skips without functional CUDA; set
#   FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 (cluster) to make CUDA mandatory.
#
# Standalone:  julia --project=<env-with-FLOWVPM[-and-CUDA]> test/runtests_gpu_fmm.jl

using Test
import Random
using Random: MersenneTwister

if !isdefined(Main, :FLOWVPM)
    import FLOWVPM
end
vpm_fmm = FLOWVPM
const ffmm = FLOWVPM.fmm

const FMM034_REQUIRE_CUDA = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

# ------------------------------------------------------------------ case setup
# Integration Phase case definitions (benchmark_033_common.jl conventions):
# cube: n uniform in the unit cube, Gamma ~ U[-1,1]^3/n, sigma = 2*n^(-1/3)
#   (overlap 2), seed 33025+n.
# wake: helical wake cylinder, D=1, length 5D about z, uniform in volume,
#   strength tangent to a helix of pitch p=D with |Gamma| = (r/R)/n,
#   sigma = 2*(V_cyl/n)^(1/3), seed 33025+7919+n.
const FMM034_SEED = 33025
const FMM034_WAKE_SEED_OFFSET = 7919
const FMM034_WAKE_R = 0.5
const FMM034_WAKE_LEN = 5.0
const FMM034_WAKE_PITCH = 2 * FMM034_WAKE_R

fmm034_settings() = vpm_fmm.FMM(; p=4, ncrit=50, theta=0.4,
    autotune_p=false, autotune_ncrit=false, autotune_reg_error=false)

function fmm034_pfield(maxparticles, R=Float64; arraytype=Matrix, UJ=vpm_fmm.UJ_fmm)
    return vpm_fmm.ParticleField(maxparticles, R;
        formulation=vpm_fmm.rVPM,
        kernel=vpm_fmm.gaussianerf,
        viscous=vpm_fmm.Inviscid(),
        SFS=vpm_fmm.noSFS,
        transposed=true,
        integration=vpm_fmm.rungekutta3,
        UJ=UJ,
        fmm=fmm034_settings(),
        arraytype=arraytype)
end

function fmm034_build_cube(n; R=Float64, maxparticles=n, UJ=vpm_fmm.UJ_fmm)
    rng = MersenneTwister(FMM034_SEED + n)
    sigma = 2.0 * (1.0 / n)^(1 / 3)
    pfield = fmm034_pfield(maxparticles, R; UJ=UJ)
    for _ in 1:n
        X = rand(rng, 3)
        Gamma = (2 .* rand(rng, 3) .- 1) ./ n
        vpm_fmm.add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

function fmm034_build_wake(n; R=Float64, maxparticles=n, UJ=vpm_fmm.UJ_fmm)
    rng = MersenneTwister(FMM034_SEED + FMM034_WAKE_SEED_OFFSET + n)
    V = pi * FMM034_WAKE_R^2 * FMM034_WAKE_LEN
    sigma = 2.0 * (V / n)^(1 / 3)
    pfield = fmm034_pfield(maxparticles, R; UJ=UJ)
    for _ in 1:n
        r = FMM034_WAKE_R * sqrt(rand(rng))
        theta = 2pi * rand(rng)
        z = FMM034_WAKE_LEN * (rand(rng) - 0.5)
        X = [r * cos(theta), r * sin(theta), z]
        t = (-r * sin(theta), r * cos(theta), FMM034_WAKE_PITCH / (2pi))
        tnrm = sqrt(sum(abs2, t))
        mag = (r / FMM034_WAKE_R) / n
        Gamma = [mag * t[1] / tnrm, mag * t[2] / tnrm, mag * t[3] / tnrm]
        vpm_fmm.add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

fmm034_build(case, n; kwargs...) = case == "cube" ?
    fmm034_build_cube(n; kwargs...) : fmm034_build_wake(n; kwargs...)

# ------------------------------------------------------------------- metrics
# relative RMS of U (rows 10:12) and J (rows 16:24) against a reference
# particle matrix (both downloaded to host Arrays first)
function fmm034_uj_errors(particles, ref_particles, np)
    A = Array(particles)
    B = Array(ref_particles)
    u_err2 = u_ref2 = j_err2 = j_ref2 = 0.0
    for i in 1:np
        for r in vpm_fmm.U_INDEX
            d = Float64(A[r, i]) - Float64(B[r, i])
            u_err2 += d * d
            u_ref2 += Float64(B[r, i])^2
        end
        for r in vpm_fmm.J_INDEX
            d = Float64(A[r, i]) - Float64(B[r, i])
            j_err2 += d * d
            j_ref2 += Float64(B[r, i])^2
        end
    end
    return (u_rel_rms=sqrt(u_err2 / max(u_ref2, eps())),
            j_rel_rms=sqrt(j_err2 / max(j_ref2, eps())))
end

const FMM034_U_GATE = 1e-3   # fixed Integration Phase velocity tolerance

# =========================================================================
# Part A: host-resident (transfer-based) coupling, CPU only
# =========================================================================
if !FLOWVPM._FMM_HAS_RADIX
    @info "installed FastMultipole lacks the radix device interface " *
          "(RadixFMMCache); skipping the radix FMM coupling tests"
else

@testset "radix FMM coupling (host path) vs UJ_direct" begin
    for (case, n) in (("cube", 4000), ("wake", 1500))
        pfield = fmm034_build(case, n)
        ref = fmm034_build(case, n; UJ=vpm_fmm.UJ_direct)
        vpm_fmm.UJ_direct(ref)

        # transfer-based host-resident radix evaluation
        vpm_fmm.UJ_fmm_gpu!(pfield)
        err = fmm034_uj_errors(pfield.particles, ref.particles, n)
        @info "host radix coupling [$case n=$n]" err.u_rel_rms err.j_rel_rms
        @test err.u_rel_rms <= FMM034_U_GATE
        # J is a logged diagnostic, not a pass/fail gate (phase rule); it
        # should still be sane
        @test err.j_rel_rms < 1e-1

        # accumulate semantics: a second evaluation without reset doubles U
        U1 = copy(Array(pfield.particles)[vpm_fmm.U_INDEX, 1:n])
        vpm_fmm.UJ_fmm_gpu!(pfield; reset=false)
        U2 = Array(pfield.particles)[vpm_fmm.U_INDEX, 1:n]
        @test isapprox(U2, 2 .* U1; rtol=1e-12)

        # cache reuse: perturb positions in place (stay inside the box) and
        # re-evaluate against a fresh direct reference
        rng = MersenneTwister(1234)
        dx = 0.001 .* (2 .* rand(rng, 3, n) .- 1)
        pfield.particles[vpm_fmm.X_INDEX, 1:n] .+= dx
        ref.particles[vpm_fmm.X_INDEX, 1:n] .+= dx
        vpm_fmm.UJ_direct(ref)
        vpm_fmm.UJ_fmm_gpu!(pfield)
        err2 = fmm034_uj_errors(pfield.particles, ref.particles, n)
        @test err2.u_rel_rms <= FMM034_U_GATE

        # varying live count at fixed cache capacity: remove particles, reuse
        for _ in 1:min(50, n ÷ 10)
            vpm_fmm.remove_particle(pfield, pfield.np)
            vpm_fmm.remove_particle(ref, ref.np)
        end
        vpm_fmm.UJ_direct(ref)
        vpm_fmm.UJ_fmm_gpu!(pfield)
        err3 = fmm034_uj_errors(pfield.particles, ref.particles, pfield.np)
        @test pfield.np == ref.np
        @test err3.u_rel_rms <= FMM034_U_GATE
    end
end

@testset "radix FMM coupling: task-035 tuning settings (host path)" begin
    # The RadixFMMSettings tuning surface (task 035): nearfield-kernel and M2L
    # strategy selection must produce the same gate-passing answer as the
    # defaults. p = 4 throughout (P = 4 standing test rule).
    n = 1500
    ref = fmm034_build("wake", n; UJ=vpm_fmm.UJ_direct)
    vpm_fmm.UJ_direct(ref)
    for (kernel, strategy) in ((:partitioned, :concat),
                               (:partitioned, :precomputed_y),
                               (:regularized, :dense))
        pfield = fmm034_build("wake", n)
        FLOWVPM.radix_fmm_settings!(pfield;
            direct_kernel=kernel, m2l_strategy=strategy)
        vpm_fmm.UJ_fmm_gpu!(pfield)
        err = fmm034_uj_errors(pfield.particles, ref.particles, n)
        @info "035 settings [$kernel + $strategy]" err.u_rel_rms err.j_rel_rms
        @test err.u_rel_rms <= FMM034_U_GATE
        @test err.j_rel_rms < 1e-1
    end
    # rho_t override reaches the constructed kernel; shipped defaults differ
    # per kernel (4.789 regularized-everywhere, 4.252 split)
    s = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned)
    @test FLOWVPM._radix_direct_kernel(s).rho_t ≈ 4.252
    s2 = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned, rho_t=4.789)
    @test FLOWVPM._radix_direct_kernel(s2).rho_t ≈ 4.789
    # cycle-1 shipped defaults (task 035, user-approved 2026-08-12):
    # PartitionedVortex nearfield + DenseTranslationM2L + margin 1.15
    sdef = FLOWVPM.RadixFMMSettings()
    @test FLOWVPM._radix_direct_kernel(sdef) isa FLOWVPM.fmm.PartitionedVortex
    @test FLOWVPM._radix_m2l_strategy(sdef)[1] isa FLOWVPM.fmm.DenseTranslationM2L
    @test sdef.accuracy_margin ≈ 1.15
    # joint auto-geometry rule reproduces the measured 035 n=1e5 winners
    # (cube (ell=4, q=17), wake (ell=5, q=16)) from the case sigma/L
    sig_c = 2 * (1 / 1e5)^(1 / 3)
    @test FLOWVPM._radix_auto_geometry(1.2, sig_c, 100_000, 16, 4.252, 1.15) ==
        (4, 17)
    sig_w = 2 * (3.927 / 1e5)^(1 / 3)
    @test FLOWVPM._radix_auto_geometry(6.0, sig_w, 100_000, 16, 4.252, 1.15) ==
        (5, 16)
    # invalid selections fail loudly
    @test_throws ErrorException FLOWVPM._radix_direct_kernel(
        FLOWVPM.RadixFMMSettings(; direct_kernel=:nope))
    @test_throws ErrorException FLOWVPM._radix_m2l_strategy(
        FLOWVPM.RadixFMMSettings(; m2l_strategy=:nope))
    # an explicit uniform level schedule matching near_radius2 is accepted
    pfield = fmm034_build("wake", n)
    FLOWVPM.radix_fmm_settings!(pfield; ell=3, near_radius2=16,
        level_radii2=(16, 16))
    vpm_fmm.UJ_fmm_gpu!(pfield)
    err = fmm034_uj_errors(pfield.particles, ref.particles, n)
    @test err.u_rel_rms <= FMM034_U_GATE
end

@testset "radix FMM coupling: automatic recenter" begin
    n = 1500
    pfield = fmm034_build("wake", n)
    ref = fmm034_build("wake", n; UJ=vpm_fmm.UJ_direct)
    vpm_fmm.UJ_fmm_gpu!(pfield)  # builds the cache at the initial box

    # translate the whole field far outside the initial domain box; the
    # coupling must recenter (derived bounds) and stay accurate
    shift = [10.0, -3.0, 7.0]
    pfield.particles[vpm_fmm.X_INDEX, 1:n] .+= shift
    ref.particles[vpm_fmm.X_INDEX, 1:n] .+= shift
    vpm_fmm.UJ_direct(ref)
    vpm_fmm.UJ_fmm_gpu!(pfield)
    err = fmm034_uj_errors(pfield.particles, ref.particles, n)
    @test err.u_rel_rms <= FMM034_U_GATE

    # with user-fixed bounds, out-of-box must throw instead of recentering
    pfield2 = fmm034_build("wake", n)
    FLOWVPM.radix_fmm_settings!(pfield2;
        bounds=([-1.0, -1.0, -3.0], 6.0))
    vpm_fmm.UJ_fmm_gpu!(pfield2)
    pfield2.particles[vpm_fmm.X_INDEX, 1:n] .+= shift
    @test_throws ArgumentError vpm_fmm.UJ_fmm_gpu!(pfield2)
end

@testset "radix FMM coupling: loud unsupported configurations" begin
    n = 200
    # FMM autotuning on (the FLOWVPM FMM() defaults) must fail loudly
    pfield = vpm_fmm.ParticleField(n; UJ=vpm_fmm.UJ_fmm)  # default FMM(): autotune on
    for i in 1:n
        vpm_fmm.add_particle(pfield, rand(3), rand(3) ./ n, 0.05)
    end
    @test_throws ErrorException vpm_fmm.UJ_fmm_gpu!(pfield)

    # non-gaussianerf kernel must fail loudly
    pfield3 = vpm_fmm.ParticleField(n; kernel=vpm_fmm.winckelmans,
        UJ=vpm_fmm.UJ_fmm, fmm=fmm034_settings())
    for i in 1:n
        vpm_fmm.add_particle(pfield3, rand(3), rand(3) ./ n, 0.05)
    end
    @test_throws ErrorException vpm_fmm.UJ_fmm_gpu!(pfield3)

    # rbf/sfs are unsupported on this path (raised before any cache is built)
    pfield4 = fmm034_build_cube(500)
    @test_throws ErrorException vpm_fmm.UJ_fmm_gpu!(pfield4; rbf=true)
    @test_throws ErrorException vpm_fmm.UJ_fmm_gpu!(pfield4; sfs=true)
end

# =========================================================================
# Part B: device-resident coupling (CUDA)
# =========================================================================
fmm034_cuda_ok = try
    if !isdefined(Main, :CUDA)
        @eval Main import CUDA
    end
    Main.CUDA.functional()
catch
    false
end

if !fmm034_cuda_ok
    if FMM034_REQUIRE_CUDA
        error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA is not functional")
    end
    @info "CUDA unavailable; skipping the device-resident radix FMM tests"
else
    # Part B lives in its own file (runtime include): its CUDA.@allocated
    # macro must not be macro-expanded on CUDA-less machines.
    include("runtests_gpu_fmm_device.jl")
end

end # _FMM_HAS_RADIX
