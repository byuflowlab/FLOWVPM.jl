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
    # rho_t override reaches the constructed kernel. The partitioned coupling
    # default is the 031a velocity-RMS cutoff 3.668 (task 035 cycle 3);
    # :regularized keeps its constructor default 4.789.
    s = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned)
    @test FLOWVPM._radix_direct_kernel(s).rho_t ≈ 3.668
    s2 = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned, rho_t=4.252)
    @test FLOWVPM._radix_direct_kernel(s2).rho_t ≈ 4.252
    @test FLOWVPM._radix_direct_kernel(
        FLOWVPM.RadixFMMSettings(; direct_kernel=:regularized)).rho_t ≈ 4.789
    # cycle-3 shipped defaults (task 035, user-approved 2026-08-12):
    # literature P5 + PartitionedVortex(rho_t=3.668) + DenseTranslationM2L,
    # q floor 6, margin 1.03
    sdef = FLOWVPM.RadixFMMSettings()
    @test FLOWVPM._radix_direct_kernel(sdef) isa FLOWVPM.fmm.PartitionedVortex
    @test FLOWVPM._radix_m2l_strategy(sdef)[1] isa FLOWVPM.fmm.DenseTranslationM2L
    @test sdef.expansion_order == 4       # literature P = 5
    @test sdef.near_radius2 == 6
    @test sdef.accuracy_margin ≈ 1.03
    # joint auto-geometry rule at the shipped defaults reproduces every
    # measured cycle-3A P5 winner from the case sigma/L
    sig_c = 2 * (1 / 1e5)^(1 / 3)
    sig_w = 2 * (3.927 / 1e5)^(1 / 3)
    sig_c6 = 2 * (1 / 1e6)^(1 / 3)
    sig_w6 = 2 * (3.927 / 1e6)^(1 / 3)
    @test FLOWVPM._radix_auto_geometry(1.2, sig_c, 100_000, 6, 3.668, 1.03) ==
        (4, 12)
    @test FLOWVPM._radix_auto_geometry(6.0, sig_w, 100_000, 6, 3.668, 1.03) ==
        (5, 6)
    @test FLOWVPM._radix_auto_geometry(1.2, sig_c6, 1_000_000, 6, 3.668, 1.03) ==
        (5, 12)
    @test FLOWVPM._radix_auto_geometry(6.0, sig_w6, 1_000_000, 6, 3.668, 1.03) ==
        (6, 6)
    # the cycle-1 rule at the old explicit settings still reproduces the
    # cycle-1/2 winners (cube (4,17), wake (5,16)) — regression on the rule
    @test FLOWVPM._radix_auto_geometry(1.2, sig_c, 100_000, 16, 4.252, 1.15) ==
        (4, 17)
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

@testset "radix FMM coupling: rectangular bounds (task 037, host path)" begin
    n = 1500
    ref = fmm034_build("wake", n; UJ=vpm_fmm.UJ_direct)
    vpm_fmm.UJ_direct(ref)

    # settings round-trip (off by default; sticks when set)
    @test FLOWVPM.RadixFMMSettings().rectangular === false
    s = FLOWVPM.radix_fmm_settings!(fmm034_build("wake", 200); rectangular=true)
    @test s.rectangular === true
    tk = FLOWVPM._radix_direct_kernel(FLOWVPM.RadixFMMSettings(;
        direct_kernel=:twopass, rho_t=3.668, rho_c=1.75))
    @test tk isa vpm_fmm.fmm.TwoPassVortex
    @test tk.rho_t == 3.668
    @test tk.rho_c == 1.75
    @test FLOWVPM._radix_primary_reach(tk) == 1.75
    @test_throws ErrorException FLOWVPM._radix_direct_kernel(
        FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned, rho_c=1.75))

    # cubic derivation is unchanged (scalar box size); rectangular derivation
    # keeps per-axis padded tight extents (vector box size), same padding and
    # 4*sigma_max floor conventions per axis
    pfield = fmm034_build("wake", n)
    bc = FLOWVPM._radix_derive_bounds(pfield, 0.1)
    br = FLOWVPM._radix_derive_bounds(pfield, 0.1; rectangular=true)
    @test bc[2] isa Float64
    @test length(br[2]) == 3
    @test maximum(br[2]) ≈ bc[2]          # long axis reproduces the cubic side
    @test minimum(br[2]) < 0.5 * bc[2]    # wake: transverse axes much tighter
    lo = [minimum(pfield.particles[r, 1:n]) for r in vpm_fmm.X_INDEX]
    hi = [maximum(pfield.particles[r, 1:n]) for r in vpm_fmm.X_INDEX]
    floor4s = 4 * FLOWVPM._radix_sigma_max(pfield)
    ext_pad = (1 + 2 * 0.1) .* max.(hi .- lo, floor4s)
    @test all(isapprox.(collect(br[2]), ext_pad; rtol=1e-12))
    @test all(collect(br[1]) .≈ (lo .+ hi) ./ 2 .- ext_pad ./ 2)

    # rectangular wake evaluation: accuracy gate + cache introspection
    FLOWVPM.radix_fmm_settings!(pfield; rectangular=true)
    vpm_fmm.UJ_fmm_gpu!(pfield)
    err = fmm034_uj_errors(pfield.particles, ref.particles, n)
    @info "rectangular host radix coupling [wake n=$n]" err.u_rel_rms err.j_rel_rms
    @test err.u_rel_rms <= FMM034_U_GATE
    @test err.j_rel_rms < 1e-1
    cache = FLOWVPM._radix_fmm_couplings[pfield].cache
    @test maximum(cache.ell_axes) > minimum(cache.ell_axes)  # genuinely rectangular
    @test cache.ell == maximum(cache.ell_axes)
    # box_extent covers the padded per-axis extents, snapped up to whole leaf
    # cells of the shared cubic width delta = L_max/2^ell (minimal per axis)
    delta = maximum(br[2]) / 2^cache.ell
    @test all(collect(cache.box_extent) .≈ delta .* 2.0 .^ collect(cache.ell_axes))
    @test all(collect(cache.box_extent) .>= collect(br[2]) .* (1 - 1e-12))
    @test all((la == 0 || delta * 2.0^(la - 1) < La * (1 + 1e-12))
              for (la, La) in zip(cache.ell_axes, br[2]))
    # Task 037a: snap-up is symmetric, so the rectangular and raw derived boxes
    # have the same center rather than retaining the raw lower face.
    @test all(isapprox.(collect(cache.x_min + cache.box_extent / 2),
        collect(br[1] + br[2] / 2); rtol=1e-12, atol=1e-12))

    # regression: the cubic default still derives a cube
    pfield_c = fmm034_build("wake", n)
    vpm_fmm.UJ_fmm_gpu!(pfield_c)
    cache_c = FLOWVPM._radix_fmm_couplings[pfield_c].cache
    @test maximum(cache_c.ell_axes) == minimum(cache_c.ell_axes)
    @test maximum(cache_c.box_extent) ≈ minimum(cache_c.box_extent)
    # rectangular derived the same depth/leaf width (auto-geometry rule is
    # shape-independent: L = max extent)
    @test cache.ell == cache_c.ell
    # Same center, longest extent, ell, and leaf width imply the same occupied
    # leaf lattice and direct list; rectangular trimming may change coarse M2L.
    @test cache.state.grid.n_cells == cache_c.state.grid.n_cells
    @test cache.state.counts.n_direct == cache_c.state.counts.n_direct

    # explicit user bounds with a 3-vector box size pass through as-is:
    # rectangular cache, and out-of-box errors (user-owned box, no recenter)
    pfield_b = fmm034_build("wake", n)
    FLOWVPM.radix_fmm_settings!(pfield_b;
        bounds=([-1.0, -1.0, -3.0], [2.0, 2.0, 6.0]))
    vpm_fmm.UJ_fmm_gpu!(pfield_b)
    err_b = fmm034_uj_errors(pfield_b.particles, ref.particles, n)
    @test err_b.u_rel_rms <= FMM034_U_GATE
    cache_b = FLOWVPM._radix_fmm_couplings[pfield_b].cache
    @test maximum(cache_b.ell_axes) > minimum(cache_b.ell_axes)
    @test collect(cache_b.box_extent)[3] ≈ 6.0
    pfield_b.particles[vpm_fmm.X_INDEX, 1:n] .+= [10.0, -3.0, 7.0]
    @test_throws ArgumentError vpm_fmm.UJ_fmm_gpu!(pfield_b)
end

@testset "radix FMM coupling: automatic recenter (rectangular)" begin
    # the automatic-recenter Part A case with rectangular=true: the derived
    # rebuild must preserve rectangularity and stay inside the accuracy gate
    n = 1500
    pfield = fmm034_build("wake", n)
    ref = fmm034_build("wake", n; UJ=vpm_fmm.UJ_direct)
    FLOWVPM.radix_fmm_settings!(pfield; rectangular=true)
    vpm_fmm.UJ_fmm_gpu!(pfield)  # builds the rectangular cache at the initial box
    cache = FLOWVPM._radix_fmm_couplings[pfield].cache
    @test maximum(cache.ell_axes) > minimum(cache.ell_axes)
    axes0 = cache.ell_axes

    shift = [10.0, -3.0, 7.0]
    pfield.particles[vpm_fmm.X_INDEX, 1:n] .+= shift
    ref.particles[vpm_fmm.X_INDEX, 1:n] .+= shift
    vpm_fmm.UJ_direct(ref)
    vpm_fmm.UJ_fmm_gpu!(pfield)
    err = fmm034_uj_errors(pfield.particles, ref.particles, n)
    @test err.u_rel_rms <= FMM034_U_GATE
    # same particle cloud, same derivation rule: rectangularity (and here the
    # exact per-axis depths) survive the recenter
    cache2 = FLOWVPM._radix_fmm_couplings[pfield].cache
    @test cache2 === cache  # recenter! swaps in place; cache identity persists
    @test maximum(cache2.ell_axes) > minimum(cache2.ell_axes)
    @test cache2.ell_axes == axes0
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
