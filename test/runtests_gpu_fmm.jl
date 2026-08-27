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

function fmm034_pfield(maxparticles, R=Float64; arraytype=Matrix, UJ=vpm_fmm.UJ_fmm,
        transposed=true)
    return vpm_fmm.ParticleField(maxparticles, R;
        formulation=vpm_fmm.rVPM,
        kernel=vpm_fmm.gaussianerf,
        viscous=vpm_fmm.Inviscid(),
        SFS=vpm_fmm.noSFS,
        transposed=transposed,
        integration=vpm_fmm.rungekutta3,
        UJ=UJ,
        fmm=fmm034_settings(),
        arraytype=arraytype)
end

function fmm034_build_cube(n; R=Float64, maxparticles=n, UJ=vpm_fmm.UJ_fmm,
        transposed=true)
    rng = MersenneTwister(FMM034_SEED + n)
    sigma = 2.0 * (1.0 / n)^(1 / 3)
    pfield = fmm034_pfield(maxparticles, R; UJ=UJ, transposed=transposed)
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
# `skip`: static-particle columns to exclude. FLOWVPM's static convention
# (2026-08-22, H200 jobs 13299959/13302646): `_reset_particles` PRESERVES
# statics' U/J while every UJ evaluation delivers to all targets, so statics'
# U/J rows accumulate one full contribution per evaluation and are consumed
# by nothing (integration/relaxation skip statics). Comparing them across
# differing evaluation counts produces a spurious linear-growth "error"
# (sqrt(3/20000) per extra call reproduced job 13298230's replay j ~ 0.1098
# exactly — the CUDA-graph replay itself is bit-faithful).
function fmm034_uj_errors(particles, ref_particles, np; skip=())
    A = Array(particles)
    B = Array(ref_particles)
    u_err2 = u_ref2 = j_err2 = j_ref2 = 0.0
    for i in 1:np
        i in skip && continue
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
# Task 049: U_prev bookkeeping — CPU loop vs broadcast equivalence
# =========================================================================
# `nextstep`'s U_prev update forks on storage type (Matrix keeps the scalar
# loop verbatim; CuArray takes `_update_U_prev_broadcast!`). Assert the two
# implementations agree bit-for-bit on a Matrix field, above and below the
# MIN_MT_NP multithreading threshold. Needs neither radix nor CUDA.
@testset "nextstep U_prev: loop vs broadcast equivalence" begin
    for n in (200, 1500)   # single-thread and multi-thread loop branches
        rng = MersenneTwister(49000 + n)
        pfield = fmm034_build_cube(n)
        pfield.particles[vpm_fmm.U_INDEX, 1:n] .= randn(rng, 3, n)

        # reference: the loop implementation exactly as nextstep runs it
        ref = copy(pfield.particles)
        for i in 1:n
            Ux, Uy, Uz = ref[vpm_fmm.U_INDEX, i]
            ref[vpm_fmm.U_PREV_INDEX, i] = sqrt(Ux*Ux + Uy*Uy + Uz*Uz)
        end

        # broadcast implementation on the same Matrix field
        FLOWVPM._update_U_prev_broadcast!(pfield)
        @test pfield.particles[vpm_fmm.U_PREV_INDEX, 1:n] ==
              ref[vpm_fmm.U_PREV_INDEX, 1:n]

        # nextstep's Array branch still runs the loop: after a dt=0 step the
        # U_prev row must equal |U| of the post-integration U rows
        pfield.particles[vpm_fmm.U_PREV_INDEX, 1:n] .= 0
        vpm_fmm.nextstep(pfield, 0.0)
        @test all(isfinite, pfield.particles[vpm_fmm.U_PREV_INDEX, 1:n])
        expected = [sqrt(sum(abs2, pfield.particles[vpm_fmm.U_INDEX, i]))
                    for i in 1:n]
        @test pfield.particles[vpm_fmm.U_PREV_INDEX, 1:n] == expected
    end
end

@testset "euler sigma_guard: dt*Z cap + floor (052c trial 1)" begin
    # Reproduces the 052c acceptance step-1015 failure mode in miniature:
    # a strained outlier with dt*Z > 1 flips sigma's sign under the
    # unguarded Euler update. Under transposed rVPM (f=0, g=1/5) with
    # Gamma=(1,0,0) and only J[1] nonzero, Z = J[1]/5 exactly.
    sigma0 = 0.1
    dt = 0.1
    function guard_field(n=6)
        pf = fmm034_pfield(n; UJ=vpm_fmm.UJ_direct)
        for i in 1:n
            vpm_fmm.add_particle(pf, [Float64(i), 0.0, 0.0],
                                 [1.0, 0.0, 0.0], sigma0)
        end
        P = pf.particles
        P[vpm_fmm.U_INDEX, 1:n] .= 0
        P[vpm_fmm.J_INDEX, 1:n] .= 0
        P[first(vpm_fmm.J_INDEX), 1] = 100.0  # dt*Z = 2.0  -> would flip sign
        P[first(vpm_fmm.J_INDEX), 2] = 10.0   # dt*Z = 0.2  -> mild contraction
        return pf
    end
    args(pf) = (pf.formulation.f, pf.formulation.g, pf.kernel.zeta(0))
    Uinf = zeros(3)

    # (a) unguarded update flips the strained particle's sigma negative
    pf = guard_field()
    f, g, zeta0 = args(pf)
    vpm_fmm._euler_cpu_reformulated!(pf, dt, Uinf, f, g, zeta0)
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 1] < 0

    # (b) cap prevents the flip (floor low enough to stay out of the way);
    # untriggered particles keep the exact unguarded update
    guard = (dtz_cap=0.5, floor=0.001)
    pf = guard_field()
    vpm_fmm._euler_cpu_reformulated!(pf, dt, Uinf, f, g, zeta0;
                                     sigma_guard=guard)
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 1] ≈ sigma0 * (1 - 0.5)  # capped
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 2] ≈ 0.08                # mild, uncapped
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 3] ≈ sigma0              # untouched (Z=0)

    # (c) floor engages after the cap: both the capped outlier (0.05 raw)
    # and the mild contraction (0.08 raw) land on the floor
    pf = guard_field()
    vpm_fmm._euler_cpu_reformulated!(pf, dt, Uinf, f, g, zeta0;
                                     sigma_guard=(dtz_cap=0.5, floor=0.085))
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 1] ≈ 0.085
    @test pf.particles[vpm_fmm.SIGMA_INDEX, 2] ≈ 0.085

    # (d) explicit (Inf, -Inf) guard is bit-identical to the empty guard
    pf1, pf2 = guard_field(), guard_field()
    vpm_fmm._euler_cpu_reformulated!(pf1, dt, Uinf, f, g, zeta0)
    vpm_fmm._euler_cpu_reformulated!(pf2, dt, Uinf, f, g, zeta0;
                                     sigma_guard=(dtz_cap=Inf, floor=-Inf))
    @test pf1.particles[vpm_fmm.SIGMA_INDEX, 1:6] ==
          pf2.particles[vpm_fmm.SIGMA_INDEX, 1:6]

    # (e) broadcast twin matches the scalar loop with the guard armed
    pf3 = guard_field()
    vpm_fmm._euler_broadcast_reformulated!(pf3, dt, Uinf, f, g, zeta0;
                                           sigma_guard=guard)
    pf4 = guard_field()
    vpm_fmm._euler_cpu_reformulated!(pf4, dt, Uinf, f, g, zeta0;
                                     sigma_guard=guard)
    @test pf3.particles[vpm_fmm.SIGMA_INDEX, 1:6] ≈
          pf4.particles[vpm_fmm.SIGMA_INDEX, 1:6]

    # (f) unknown guard keys are rejected loudly
    pf5 = guard_field()
    @test_throws ArgumentError vpm_fmm._euler_cpu_reformulated!(
        pf5, dt, Uinf, f, g, zeta0; sigma_guard=(bogus=1.0,))
end

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
    # A multi-setting update is transactional across semantic local validation,
    # global GPU settings, the per-field registry, and an already-live cache.
    # Geometry pinned explicitly (052c): at n=200 the auto pick at the shipped
    # rho_t=4.789 is (ell=2, near_radius2=12), which on the rectangular 2x2x4
    # leaf grid puts every offset inside the near ball — a ZERO-M2L degenerate
    # cache that evaluates pure direct. The explicit pin keeps this block
    # independent of auto-geometry defaults (the 048 rho_t bump 3.668->4.789
    # is what moved q from 6 to 12 and broke the unpinned build) and doubles
    # as host-path coverage of the degenerate pure-direct support.
    atomic_field = fmm034_build("wake", 200)
    atomic_before = FLOWVPM.radix_fmm_settings!(atomic_field;
        rectangular=true, ell=2, near_radius2=12)
    vpm_fmm.UJ_fmm_gpu!(atomic_field)
    # degenerate pure-direct correctness: no M2L anywhere, answer ~= UJ_direct
    atomic_ref = fmm034_build("wake", 200; UJ=vpm_fmm.UJ_direct)
    vpm_fmm.UJ_direct(atomic_ref)
    atomic_err = fmm034_uj_errors(atomic_field.particles, atomic_ref.particles, 200)
    @test atomic_err.u_rel_rms <= FMM034_U_GATE
    @test haskey(FLOWVPM._radix_fmm_couplings, atomic_field)
    coupling_before = FLOWVPM._radix_fmm_couplings[atomic_field]
    cache_before = coupling_before.cache
    gh_before = vpm_fmm.fmm.radix_setting(:CUDA_NEARFIELD_GH_MODE)
    gh_proposed = gh_before === :shipped ? :fp32 : :shipped
    @test_throws ArgumentError FLOWVPM.radix_fmm_settings!(atomic_field;
        gpu=(; CUDA_NEARFIELD_GH_MODE=gh_proposed),
        direct_kernel=:bogus, rectangular=false)
    @test vpm_fmm.fmm.radix_setting(:CUDA_NEARFIELD_GH_MODE) === gh_before
    @test FLOWVPM._radix_fmm_settings[atomic_field] === atomic_before
    @test FLOWVPM._radix_fmm_couplings[atomic_field] === coupling_before
    @test FLOWVPM._radix_fmm_couplings[atomic_field].cache === cache_before

    # Every settings field with a downstream semantic contract is rejected at
    # the eager boundary, before the valid proposed GPU change can leak.
    invalid_local = (
        (; expansion_order=-1),
        (; ell=1),
        (; near_radius2=-1),
        (; window_classes=0),
        (; padding=0.0),
        (; bounds=([0.0, 0.0, 0.0], -1.0)),
        (; precision=Float16),
        (; direct_kernel=:bogus),
        (; direct_kernel=:regularized, rho_t=0.0),
        (; direct_kernel=:regularized, rho_c=2.0),
        (; m2l_strategy=:bogus),
        (; ell=3, near_radius2=6, level_radii2=(6,)),
        # n=200 resolves auto ell=2, so the legacy 2:ell schedule has length 1.
        (; near_radius2=6, level_radii2=(6, 6)),
        (; accuracy_margin=0.0),
    )
    for local_kwargs in invalid_local
        @test_throws ArgumentError FLOWVPM.radix_fmm_settings!(atomic_field;
            gpu=(; CUDA_NEARFIELD_GH_MODE=gh_proposed), local_kwargs...)
        @test vpm_fmm.fmm.radix_setting(:CUDA_NEARFIELD_GH_MODE) === gh_before
        @test FLOWVPM._radix_fmm_settings[atomic_field] === atomic_before
        @test FLOWVPM._radix_fmm_couplings[atomic_field] === coupling_before
        @test FLOWVPM._radix_fmm_couplings[atomic_field].cache === cache_before
    end

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
    # default is the conservative Jacobian-per-pair cutoff 4.789 (task 048
    # production selection, 2026-08-22; supersedes 035's 3.668);
    # :regularized keeps its constructor default 4.789.
    s = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned)
    @test FLOWVPM._radix_direct_kernel(s).rho_t ≈ 4.789
    s2 = FLOWVPM.RadixFMMSettings(; direct_kernel=:partitioned, rho_t=4.252)
    @test FLOWVPM._radix_direct_kernel(s2).rho_t ≈ 4.252
    @test FLOWVPM._radix_direct_kernel(
        FLOWVPM.RadixFMMSettings(; direct_kernel=:regularized)).rho_t ≈ 4.789
    # shipped defaults (task 048 production selection, user-approved
    # 2026-08-22): P6 + PartitionedVortex(rho_t=4.789) + DenseTranslationM2L,
    # derived near shell (q floor 6), margin 1.03 — passes the strict 5e-4
    # F64 delivered-E_str gate on the p018 production field (job 13303399)
    sdef = FLOWVPM.RadixFMMSettings()
    @test FLOWVPM._radix_direct_kernel(sdef) isa FLOWVPM.fmm.PartitionedVortex
    @test FLOWVPM._radix_m2l_strategy(sdef)[1] isa FLOWVPM.fmm.DenseTranslationM2L
    @test sdef.expansion_order == 6       # literature P = 7
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

@testset "radix FMM coupling: depth rebuild on particle growth (052c)" begin
    # A wake growing from hundreds to thousands of particles must not keep
    # the shallow grid depth frozen at first build (052 stage-d root cause:
    # ell=2 chosen at np=330 served 242k particles near-dense). Fixed sigma
    # so only np drives the occupancy cap: build at n0=300 (ell=2), grow to
    # n1=8000 (occupancy admits ell=4) and expect a strictly deeper rebuild.
    n0, n1 = 300, 8000
    sigma = 0.03
    rng = MersenneTwister(FMM034_SEED + 52)
    pfield = fmm034_pfield(n1)
    ref = fmm034_pfield(n1; UJ=vpm_fmm.UJ_direct)
    Xs = [rand(rng, 3) for _ in 1:n1]
    Gs = [(2 .* rand(rng, 3) .- 1) ./ n1 for _ in 1:n1]
    for i in 1:n0
        vpm_fmm.add_particle(pfield, Xs[i], Gs[i], sigma)
        vpm_fmm.add_particle(ref, Xs[i], Gs[i], sigma)
    end
    vpm_fmm.UJ_fmm_gpu!(pfield)
    st0 = FLOWVPM._radix_fmm_couplings[pfield]
    ell0 = st0.cache.ell
    for i in (n0 + 1):n1
        vpm_fmm.add_particle(pfield, Xs[i], Gs[i], sigma)
        vpm_fmm.add_particle(ref, Xs[i], Gs[i], sigma)
    end
    vpm_fmm.UJ_direct(ref)
    vpm_fmm.UJ_fmm_gpu!(pfield)
    st1 = FLOWVPM._radix_fmm_couplings[pfield]
    @test st1.cache !== st0.cache
    @test st1.cache.ell > ell0
    err = fmm034_uj_errors(pfield.particles, ref.particles, n1)
    @info "depth rebuild on growth" ell0 st1.cache.ell err.u_rel_rms
    @test err.u_rel_rms <= FMM034_U_GATE

    # unchanged np must not churn the coupling
    vpm_fmm.UJ_fmm_gpu!(pfield)
    @test FLOWVPM._radix_fmm_couplings[pfield] === st1

    # a user-fixed ell is a promise: growth must never rebuild it
    pfield2 = fmm034_pfield(n1)
    for i in 1:n0
        vpm_fmm.add_particle(pfield2, Xs[i], Gs[i], sigma)
    end
    FLOWVPM.radix_fmm_settings!(pfield2; ell=2, near_radius2=6)
    vpm_fmm.UJ_fmm_gpu!(pfield2)
    st2 = FLOWVPM._radix_fmm_couplings[pfield2]
    for i in (n0 + 1):n1
        vpm_fmm.add_particle(pfield2, Xs[i], Gs[i], sigma)
    end
    vpm_fmm.UJ_fmm_gpu!(pfield2)
    @test FLOWVPM._radix_fmm_couplings[pfield2] === st2
    @test FLOWVPM._radix_fmm_couplings[pfield2].cache.ell == 2
end

@testset "radix FMM coupling: sigma-outgrown rebuild (052c near-peak)" begin
    # Merge-produced oversize particles grow sigma_max between cache builds;
    # the cached grid must rebuild to an admissible geometry (shallower ell
    # and/or larger near set) instead of tripping FastMultipole's runtime
    # adequacy gate (job 13497184: ell=4 admissible at sigma_max=0.0198 near
    # step 473, refused at 0.02137 by step 502).
    n = 8000
    sigma0 = 0.005
    rng = MersenneTwister(FMM034_SEED + 53)
    pfield = fmm034_pfield(n)
    ref = fmm034_pfield(n; UJ=vpm_fmm.UJ_direct)
    Xs = [rand(rng, 3) for _ in 1:n]
    Gs = [(2 .* rand(rng, 3) .- 1) ./ n for _ in 1:n]
    for i in 1:n
        vpm_fmm.add_particle(pfield, Xs[i], Gs[i], sigma0)
        vpm_fmm.add_particle(ref, Xs[i], Gs[i], sigma0)
    end
    vpm_fmm.UJ_fmm_gpu!(pfield)
    st0 = FLOWVPM._radix_fmm_couplings[pfield]
    ell0 = st0.cache.ell
    @test isfinite(st0.sigma_limit)
    # a single oversize particle (the fat tail) past the cached limit
    sigma_big = 1.05 * st0.sigma_limit
    vpm_fmm.get_sigma(pfield, 1) .= sigma_big
    vpm_fmm.get_sigma(ref, 1) .= sigma_big
    vpm_fmm.UJ_direct(ref)
    vpm_fmm.UJ_fmm_gpu!(pfield)   # must rebuild, not throw
    st1 = FLOWVPM._radix_fmm_couplings[pfield]
    @test st1.cache !== st0.cache
    @test sigma_big <= st1.sigma_limit  # rebuilt geometry admits the new sigma
    err = fmm034_uj_errors(pfield.particles, ref.particles, n)
    @info "sigma-outgrown rebuild" ell0 st1.cache.ell st0.sigma_limit st1.sigma_limit err.u_rel_rms
    @test err.u_rel_rms <= FMM034_U_GATE

    # steady sigma: no churn
    vpm_fmm.UJ_fmm_gpu!(pfield)
    @test FLOWVPM._radix_fmm_couplings[pfield] === st1

    # user-fixed ell is a promise: no auto rebuild — the runtime adequacy
    # gate reports instead of a silent geometry change
    pfield2 = fmm034_pfield(n)
    for i in 1:n
        vpm_fmm.add_particle(pfield2, Xs[i], Gs[i], sigma0)
    end
    FLOWVPM.radix_fmm_settings!(pfield2; ell=2, near_radius2=6)
    vpm_fmm.UJ_fmm_gpu!(pfield2)
    st2 = FLOWVPM._radix_fmm_couplings[pfield2]
    lim2 = FLOWVPM._radix_sigma_limit(st2.cache, st2.settings)
    vpm_fmm.get_sigma(pfield2, 1) .= 1.5 * lim2
    @test_throws ArgumentError vpm_fmm.UJ_fmm_gpu!(pfield2)
    @test FLOWVPM._radix_fmm_couplings[pfield2] === st2
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

    # rbf remains unsupported on this path (raised before any cache is built);
    # sfs is supported since task 048 (see the dedicated SFS testset below)
    pfield4 = fmm034_build_cube(500)
    @test_throws ErrorException vpm_fmm.UJ_fmm_gpu!(pfield4; rbf=true)
    # sfs no longer throws (n=1500: the 500-particle cube's σ overlap is too
    # large for the radix auto-geometry rule, unrelated to SFS)
    pfield5 = fmm034_build_cube(1500)
    vpm_fmm.UJ_fmm_gpu!(pfield5; sfs=true)
    @test any(!iszero, pfield5.particles[vpm_fmm.SFS_INDEX, 1:1500])
end

# relative RMS of the SFS rows (40:42) against a reference particle matrix
function fmm034_sfs_relrms(particles, ref_particles, np)
    A = Array(particles)
    B = Array(ref_particles)
    e2 = r2 = 0.0
    for i in 1:np
        for r in vpm_fmm.SFS_INDEX
            d = Float64(A[r, i]) - Float64(B[r, i])
            e2 += d * d
            r2 += Float64(B[r, i])^2
        end
    end
    return sqrt(e2 / max(r2, eps()))
end

# all-pairs ζ brute force of E_str from the J currently ON the particles
# (rows J_INDEX), FLOWVPM Estr_direct algebra, saturation cutoff matched to
# the framework (ρ² ≤ 81 in F64, ≤ 42.25 in F32 — contributions ≤ 1e-18/1e-9
# of ζ(0)). Isolates the SFS pass from the FMM's own U/J error.
function fmm034_sfs_bruteforce(pfield, np; transposed=true,
        honor_static=true)
    P = Array(pfield.particles)
    K1 = 1 / (2pi)^1.5
    rc2 = eltype(pfield.particles) === Float32 ? 42.25 : 81.0
    E = zeros(3, np)
    for i in 1:np
        honor_static && P[vpm_fmm.STATIC_INDEX, i] != 0 && continue
        Ji = view(P, vpm_fmm.J_INDEX, i)
        for j in 1:np
            i == j && continue
            honor_static && P[vpm_fmm.STATIC_INDEX, j] != 0 && continue
            dx = P[1, i] - P[1, j]; dy = P[2, i] - P[2, j]; dz = P[3, i] - P[3, j]
            sig = Float64(P[vpm_fmm.SIGMA_INDEX, j])
            rho2 = (dx^2 + dy^2 + dz^2) / sig^2
            rho2 <= rc2 || continue
            z = K1 * exp(-rho2 / 2) / sig^3
            Jj = view(P, vpm_fmm.J_INDEX, j)
            G1, G2, G3 = P[vpm_fmm.GAMMA_INDEX, j]
            if transposed
                E[1, i] += z * ((Ji[1] - Jj[1]) * G1 + (Ji[2] - Jj[2]) * G2 + (Ji[3] - Jj[3]) * G3)
                E[2, i] += z * ((Ji[4] - Jj[4]) * G1 + (Ji[5] - Jj[5]) * G2 + (Ji[6] - Jj[6]) * G3)
                E[3, i] += z * ((Ji[7] - Jj[7]) * G1 + (Ji[8] - Jj[8]) * G2 + (Ji[9] - Jj[9]) * G3)
            else
                E[1, i] += z * ((Ji[1] - Jj[1]) * G1 + (Ji[4] - Jj[4]) * G2 + (Ji[7] - Jj[7]) * G3)
                E[2, i] += z * ((Ji[2] - Jj[2]) * G1 + (Ji[5] - Jj[5]) * G2 + (Ji[8] - Jj[8]) * G3)
                E[3, i] += z * ((Ji[3] - Jj[3]) * G1 + (Ji[6] - Jj[6]) * G2 + (Ji[9] - Jj[9]) * G3)
            end
        end
    end
    return E
end

fmm034_sfs_vs_matrix(pfield, E, np) = begin
    S = Array(pfield.particles)[vpm_fmm.SFS_INDEX, 1:np]
    sqrt(sum(abs2, S .- E) / max(sum(abs2, E), eps()))
end
fmm034_matrix_relrms(A, B) = sqrt(sum(abs2, Float64.(A) .- Float64.(B)) /
                                   max(sum(abs2, Float64.(B)), eps()))

@testset "radix FMM coupling: SFS host path (task 048)" begin
    # Host-radix SFS (UJ_fmm_gpu! sfs=true) on the cube case at FM expansion
    # orders 4 AND 8 (standing P=4 rule), two-tier gates:
    #
    # (1) MECHANICAL parity (what task 048 owns): radix SFS vs an all-pairs
    #     ζ brute force built from the radix-delivered J itself — gate 1e-6.
    #     Requires the widened shell ell=2/near_radius2=20 (min ρ ≈ 5.2) so
    #     the U-list covers every non-negligible ζ pair; at the DERIVED
    #     defaults (ell=2, q=12, min ρ ≈ 3.84) the U-list ζ truncation alone
    #     measures ≈ 3.0e-3 rel RMS (recorded default-list gap, 2026-08-20 —
    #     the design-anticipated risk; widened here per its instruction).
    # (2) PHYSICS vs the exact-erf references (Estr_direct!/Estr_fmm!): E is
    #     built from J, and the radix J carries the 031a erf-free g/h
    #     nearfield approximation (j_rel_rms ≈ 1.9e-3 on this overlap-2 cube,
    #     P- and shell-independent; U meets its own 1e-3 gate). The E error is
    #     J-bound (measured e/j ratio ≈ 1.9 HERE — but the ratio is
    #     field-dependent: 2.05 on cube n=2e4, 14.1 on wake n=2e4, measured
    #     2026-08-21; the device testset uses 20x headroom for that reason),
    #     so the gate is max(1e-3, 3 · j_rel_rms): it tightens automatically
    #     if the g/h arithmetic ever improves, and a mechanism regression
    #     (not J-bound) still fails loudly. A flat 1e-3 vs exact-erf
    #     references is unattainable at any radix setting while J is ≈ 2e-3.
    n = 1500
    # Required matrix: P=4/P=8 x Float32/Float64. The original delivery only
    # covered P=4 in Float32 and substituted a J-scaled mechanical gate for
    # the required CPU physics gate.
    for R in (Float64, Float32), P in (4, 8), rho_t in (4.211, 4.789)
        pfield = fmm034_build_cube(n; R)
        FLOWVPM.radix_fmm_settings!(pfield; expansion_order=P, ell=2,
            near_radius2=20, rho_t)
        vpm_fmm.UJ_fmm_gpu!(pfield; reset=true, reset_sfs=true, sfs=true)

        # (1) mechanical parity from the radix-delivered J
        E_mech = fmm034_sfs_bruteforce(pfield, n)
        e_mech = fmm034_sfs_vs_matrix(pfield, E_mech, n)

        # exact reference: direct U/J then the direct pairwise Estr
        ref = fmm034_build_cube(n; R, UJ=vpm_fmm.UJ_direct)
        vpm_fmm.UJ_direct(ref)
        vpm_fmm.Estr_direct!(ref)

        # legacy-octree FMM SFS reference (UJ_fmm runs Estr_fmm! when sfs=true)
        # The legacy octree Tree constructor is not Float32-clean in this
        # stack, so use its Float64 CPU Estr_fmm! result as the reference for
        # both delivered precisions (as with the usual high-precision oracle).
        fmmref = fmm034_build_cube(n)
        vpm_fmm.UJ_fmm(fmmref; sfs=true, reset=true)

        j_rel = fmm034_uj_errors(pfield.particles, ref.particles, n).j_rel_rms
        e_direct = fmm034_sfs_relrms(pfield.particles, ref.particles, n)
        e_fmm = fmm034_sfs_relrms(pfield.particles, fmmref.particles, n)
        e_sanity = fmm034_sfs_relrms(fmmref.particles, ref.particles, n)
        # Conservative per-pair candidates were derived at ε=1e-3 with half
        # reserved for the omitted tail. F64 can enforce that ε/2 budget;
        # F32 retains the phase-wide 1e-3 delivered gate.
        strict_gate = R === Float64 ? 5e-4 : 1e-3
        mech_gate = R === Float64 ? 1e-6 : 1e-4
        @info "SFS host radix [cube n=$n P=$P $R rho_t=$rho_t]" e_mech e_direct e_fmm e_sanity j_rel strict_gate
        @test e_mech <= mech_gate
        # Required delivered-physics gates; mechanical parity above is a
        # separate mechanism gate and never substitutes for CPU Estr parity.
        @test e_direct <= strict_gate
        @test e_fmm <= strict_gate

        # accumulate semantics: without reset_sfs a second sfs evaluation
        # doubles the SFS rows
        S1 = copy(Array(pfield.particles)[vpm_fmm.SFS_INDEX, 1:n])
        vpm_fmm.UJ_fmm_gpu!(pfield; reset=true, reset_sfs=false, sfs=true)
        S2 = Array(pfield.particles)[vpm_fmm.SFS_INDEX, 1:n]
        @test isapprox(S2, 2 .* S1; rtol=R === Float64 ? 1e-10 : 1e-5)

        # sfs=false evaluations leave the SFS rows untouched
        sfs_ctx = FLOWVPM._radix_fmm_couplings[pfield].cache.state.sfs
        om_before = copy(sfs_ctx.om)
        q_before = copy(sfs_ctx.q)
        vpm_fmm.UJ_fmm_gpu!(pfield; reset=true, reset_sfs=false, sfs=false)
        @test Array(pfield.particles)[vpm_fmm.SFS_INDEX, 1:n] == S2
        @test sfs_ctx.om == om_before
        @test sfs_ctx.q == q_before
    end

    # classic (transposed=false) scheme, Float64, P = 4: mechanical parity
    # (scheme-flag correctness) + the J-bound physics gate
    pfield_c = fmm034_build_cube(n; transposed=false)
    FLOWVPM.radix_fmm_settings!(pfield_c; expansion_order=4, ell=2,
        near_radius2=20)
    vpm_fmm.UJ_fmm_gpu!(pfield_c; reset=true, reset_sfs=true, sfs=true)
    e_mech_c = fmm034_sfs_vs_matrix(pfield_c,
        fmm034_sfs_bruteforce(pfield_c, n; transposed=false), n)
    ref_c = fmm034_build_cube(n; UJ=vpm_fmm.UJ_direct, transposed=false)
    vpm_fmm.UJ_direct(ref_c)
    vpm_fmm.Estr_direct!(ref_c)
    j_rel_c = fmm034_uj_errors(pfield_c.particles, ref_c.particles, n).j_rel_rms
    e_classic = fmm034_sfs_relrms(pfield_c.particles, ref_c.particles, n)
    @info "SFS host radix classic scheme [cube n=$n P=4]" e_mech_c e_classic j_rel_c
    @test e_mech_c <= 1e-6
    @test e_classic <= max(1e-3, 3 * j_rel_c)
    # the two schemes genuinely differ on this field (guards against the
    # transposed flag being silently ignored)
    ref_t = fmm034_build_cube(n; UJ=vpm_fmm.UJ_direct)
    vpm_fmm.UJ_direct(ref_t)
    vpm_fmm.Estr_direct!(ref_t)
    @test fmm034_sfs_relrms(ref_c.particles, ref_t.particles, n) > 1e-3

    # capacity > np: the cache's SFS scatter buffers are capacity-wide, so
    # delivery passes the SubArray prefix view to sfs_to_target! — regression
    # for the buf::Matrix over-pinning (2026-08-21). Identical answer to the
    # exact-capacity run.
    pfield_cap = fmm034_build_cube(n; maxparticles=n + 128)
    FLOWVPM.radix_fmm_settings!(pfield_cap; expansion_order=4, ell=2,
        near_radius2=20)
    vpm_fmm.UJ_fmm_gpu!(pfield_cap; reset=true, reset_sfs=true, sfs=true)
    pfield_eq = fmm034_build_cube(n)
    FLOWVPM.radix_fmm_settings!(pfield_eq; expansion_order=4, ell=2,
        near_radius2=20)
    vpm_fmm.UJ_fmm_gpu!(pfield_eq; reset=true, reset_sfs=true, sfs=true)
    @test fmm034_sfs_relrms(pfield_cap.particles, pfield_eq.particles, n) < 1e-12

    # CPU Estr semantics: static particles are neither SFS sources nor SFS
    # targets. Their pre-existing SFS rows remain untouched.
    pfield_static = fmm034_build_cube(n)
    ref_static = fmm034_build_cube(n; UJ=vpm_fmm.UJ_direct)
    for i in (2, 17, 201)
        vpm_fmm.set_static(pfield_static, i, 1.0)
        vpm_fmm.set_static(ref_static, i, 1.0)
    end
    sentinel = [3.0, -2.0, 1.0]
    for i in (2, 17, 201)
        pfield_static.particles[vpm_fmm.SFS_INDEX, i] .= sentinel
        ref_static.particles[vpm_fmm.SFS_INDEX, i] .= sentinel
    end
    S_static_before = copy(pfield_static.particles[vpm_fmm.SFS_INDEX, 1:n])
    FLOWVPM.radix_fmm_settings!(pfield_static; expansion_order=4, ell=2,
        near_radius2=20)
    vpm_fmm.UJ_fmm_gpu!(pfield_static; reset=true, reset_sfs=false, sfs=true)
    vpm_fmm.UJ_direct(ref_static)
    vpm_fmm.Estr_direct!(ref_static)
    static_indices = [2, 17, 201]
    active_indices = setdiff(collect(1:n), static_indices)
    S_static_delta = pfield_static.particles[vpm_fmm.SFS_INDEX, 1:n] .-
        S_static_before
    E_masked = fmm034_sfs_bruteforce(pfield_static, n)
    E_all_active = fmm034_sfs_bruteforce(pfield_static, n;
        honor_static=false)
    @test all(iszero, S_static_delta[:, static_indices])
    @test fmm034_matrix_relrms(S_static_delta[:, active_indices],
                               E_masked[:, active_indices]) < 1e-6
    # Explicitly proves static SOURCE removal matters, not just target masking.
    @test E_masked[:, active_indices] != E_all_active[:, active_indices]
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
