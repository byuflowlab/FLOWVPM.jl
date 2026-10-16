using Test
import Random
import LinearAlgebra
import FLOWVPM
import Enzyme
import ForwardDiff

# Enzyme reverse and forward mode (default settings) through one UJ_direct time
# step, checked against central FD only: reverse gradient along random directions,
# forward tangents (repeated: deterministic, the zeroed-shadow workaround in
# FLOWVPMEnzymeExt) and ForwardDiff (SFS on).
# The field is built inside the differentiated function (FLOWVPMEnzymeExt marks
# its filament workspaces and the rho/sigma tuning inactive). Objective: weights on
# [X; Gamma; sigma] after the step; inputs x = [X; Gamma; sigma] before it.
# Two compiled configurations; with the rules test below, the core-spreading full
# steps (rVPM, gaussianerf, Pedrizzetti, Euler + constant SFS, RK3 + no SFS) and the
# threaded step, every option of each axis runs at least once. `relax` is a runtime
# flag (same compiled code): relax=false is the no-relaxation case.

const ENZ_N = 50

function enz_step(x::AbstractVector{T}, w, cfg) where T
    n = length(x) ÷ 7
    pf = FLOWVPM.ParticleField(n, T; formulation=cfg.formulation, kernel=cfg.kernel, UJ=FLOWVPM.UJ_direct,
                               SFS=cfg.SFS, relaxation=cfg.relaxation, integration=cfg.integration,
                               fmm=FLOWVPM.FMM(autotune_reg_error=false))
    for i in 1:n
        FLOWVPM.add_particle(pf, (x[3i-2], x[3i-1], x[3i]), (x[3n+3i-2], x[3n+3i-1], x[3n+3i]), x[6n+i])
    end
    FLOWVPM.nextstep(pf, 0.01; relax=cfg.relax)
    s = zero(T)
    for i in 1:n
        X = FLOWVPM.get_X(pf, i); G = FLOWVPM.get_Gamma(pf, i)
        s += w[3i-2]*X[1] + w[3i-1]*X[2] + w[3i]*X[3] +
             w[3n+3i-2]*G[1] + w[3n+3i-1]*G[2] + w[3n+3i]*G[3] + w[6n+i]*FLOWVPM.get_sigma(pf, i)[]
    end
    return s
end

# relative error of a directional derivative d against central FD of f along v;
# FD truncation error is O(h^2), roundoff O(1/h): best of the steps
# (scale: |d| by default; the gradient's norm where a random direction can make d small)
enz_fd_err(f, x, v, d; hs=(1e-4, 1e-5, 1e-6, 1e-7), scale=abs(d)) = minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - d)/scale, hs)

@testset verbose=true "Enzyme: one UJ_direct step" begin
    pedr = FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3)
    cpedr = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3)
    cfgs = (
        (name="classic, winckelmans, correctedpedrizzetti, dynamic SFS, rungekutta3", formulation=FLOWVPM.formulation_classic,
         kernel=FLOWVPM.kernel_winckelmans, relaxation=cpedr, relaxes=(true,), SFS=FLOWVPM.SFS_Cd_twolevel_nobackscatter,
         integration=FLOWVPM.rungekutta3, fwd=false, fdiff=false),
        (name="rVPM, gaussianerf, pedrizzetti and none, two-level dynamic SFS, euler", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=pedr, relaxes=(true, false),
         SFS=FLOWVPM.SFS_Cd_twolevel_analytic_nobackscatter, integration=FLOWVPM.euler, fwd=true, fdiff=true),
    )
    n = ENZ_N
    Random.seed!(1)
    x = vcat(rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
    w = randn(7n)
    for cfg0 in cfgs, relax in cfg0.relaxes
        cfg = (; cfg0.formulation, cfg0.kernel, cfg0.relaxation, relax, cfg0.SFS, cfg0.integration)
        @testset "$(cfg0.name), relax=$relax" begin
            f = x -> enz_step(x, w, cfg)
            g = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, enz_step, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w), Enzyme.Const(cfg))
            @test all(isfinite, g)
            for k in 1:2
                v = randn(7n); v ./= LinearAlgebra.norm(v)
                @test enz_fd_err(f, x, v, LinearAlgebra.dot(g, v)) < 1e-5
            end
            v = randn(7n)
            if cfg0.fwd && relax
                fwd = [Enzyme.autodiff(Enzyme.Forward, enz_step, Enzyme.Duplicated(x, v), Enzyme.Const(w), Enzyme.Const(cfg))[1]
                       for _ in 1:3]
                @test all(==(fwd[1]), fwd)
                @test enz_fd_err(f, x, v, fwd[1]) < 1e-5
            end
            if cfg0.fdiff && relax
                @test enz_fd_err(f, x, v, ForwardDiff.derivative(t -> f(x .+ t .* v), 0.0)) < 1e-5
            end
        end
    end
end

# Above FLOWVPM.MIN_MT_NP particles every per-particle loop of the step takes its
# Threads.@threads branch (integrator, relaxation, SFS clipping, U_prev, Estr). Julia
# must be started with -t >= 2 for this to exercise threads (else it is skipped).
@testset "Enzyme: one step through the threaded branches (N > MIN_MT_NP)" begin
    if Threads.nthreads() == 1
        @info "Threaded Enzyme step test skipped: start Julia with -t 2 or more"
    else
        cfg = (formulation=FLOWVPM.formulation_rVPM, kernel=FLOWVPM.kernel_gaussianerf,
               relaxation=FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3), relax=true,
               SFS=FLOWVPM.ConstantSFS(FLOWVPM.Estr_fmm; Cs=1.0, clippings=(FLOWVPM.clipping_backscatter,)),
               integration=FLOWVPM.rungekutta3)
        n = FLOWVPM.MIN_MT_NP + 100
        Random.seed!(4)
        x = vcat(0.3 * n^(1/3) .* rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
        w = randn(7n)
        g = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, enz_step, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w), Enzyme.Const(cfg))
        @test all(isfinite, g)
        f = x -> enz_step(x, w, cfg)
        v = randn(7n); v ./= LinearAlgebra.norm(v)
        gv = LinearAlgebra.dot(g, v)
        @test minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - gv)/abs(gv), (1e-4, 1e-5, 1e-6, 1e-7)) < 1e-5
    end
end

# FLOWVPMEnzymeExt's reverse rules for the host direct sum (U, J) and Estr_direct
# (SFS), against central FD. Every kernel (one compile each), both stretching
# schemes (a runtime flag), one static particle (Estr skips it).
function enz_uj(x::AbstractVector{T}, w, kernel, transposed) where T
    n = length(x) ÷ 7
    pf = FLOWVPM.ParticleField(n, T; kernel, transposed, UJ=FLOWVPM.UJ_direct, fmm=FLOWVPM.FMM(autotune_reg_error=false))
    for i in 1:n
        FLOWVPM.add_particle(pf, (x[3i-2], x[3i-1], x[3i]), (x[3n+3i-2], x[3n+3i-1], x[3n+3i]), x[6n+i]; static=(i == 2))
    end
    FLOWVPM.UJ_direct(pf; sfs=true)
    rows = vcat(collect(FLOWVPM.U_INDEX), collect(FLOWVPM.J_INDEX), collect(FLOWVPM.SFS_INDEX))
    s = zero(T)
    for i in 1:n, (k, row) in enumerate(rows)
        s += w[15*(i-1)+k] * pf.particles[row, i]
    end
    return s
end

@testset "Enzyme: direct-sum U/J and Estr reverse rules" begin
    n = 30
    Random.seed!(3)
    x = vcat(rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
    w = randn(15n)
    for kernel in (FLOWVPM.kernel_gaussianerf, FLOWVPM.kernel_gaussian, FLOWVPM.kernel_winckelmans, FLOWVPM.kernel_singular),
        transposed in (true, false)
        g = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, enz_uj, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w),
                        Enzyme.Const(kernel), Enzyme.Const(transposed))
        @test all(isfinite, g)
        f = x -> enz_uj(x, w, kernel, transposed)
        for k in 1:2
            v = randn(7n); v ./= LinearAlgebra.norm(v)
            @test enz_fd_err(f, x, v, LinearAlgebra.dot(g, v); hs=(1e-4, 1e-5, 1e-6), scale=LinearAlgebra.norm(g)) < 1e-7
        end
    end
end

# The analytic two-level SFS derivatives (`dsigma_direct!` rule) after the direct
# sum, objective on M[1:6] (both schemes, one static particle), against central FD.
function enz_dsigma(x::AbstractVector{T}, w, transposed) where T
    n = length(x) ÷ 7
    pf = FLOWVPM.ParticleField(n, T; transposed, UJ=FLOWVPM.UJ_direct, fmm=FLOWVPM.FMM(autotune_reg_error=false))
    for i in 1:n
        FLOWVPM.add_particle(pf, (x[3i-2], x[3i-1], x[3i]), (x[3n+3i-2], x[3n+3i-1], x[3n+3i]), x[6n+i]; static=(i == 2))
    end
    FLOWVPM.UJ_direct(pf)
    FLOWVPM.dsigma_direct!(pf)
    s = zero(T)
    for i in 1:n, k in 1:6
        s += w[6*(i-1)+k] * pf.particles[FLOWVPM.M_INDEX[k], i]
    end
    return s
end

@testset "Enzyme: dsigma_direct! reverse rule (two-level analytic SFS)" begin
    n = 30
    Random.seed!(5)
    x = vcat(rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
    w = randn(6n)
    for transposed in (true, false)
        g = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, enz_dsigma, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w), Enzyme.Const(transposed))
        f = x -> enz_dsigma(x, w, transposed)
        for k in 1:2
            v = randn(7n); v ./= LinearAlgebra.norm(v)
            @test enz_fd_err(f, x, v, LinearAlgebra.dot(g, v); hs=(1e-4, 1e-5, 1e-6), scale=LinearAlgebra.norm(g)) < 1e-7
        end
    end
end

# FLOWVPMEnzymeExt's FMM reverse rule (`_fmm_UJ!`: adjoint FMM calls) and the
# Estr_fmm! rule, through one UJ_fmm step, against central FD of the FMM
# (agreement set by the FMM error, here ~1e-5).
function enz_fmm_step(x::AbstractVector{T}, w, UJ, SFS) where T
    n = length(x) ÷ 7
    pf = FLOWVPM.ParticleField(n, T; UJ, SFS, integration=FLOWVPM.euler,
                               fmm=FLOWVPM.FMM(p=8, ncrit=20, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false,
                                               min_ncrit=1, relative_tolerance=1e-12, absolute_tolerance=1e-12))
    for i in 1:n
        FLOWVPM.add_particle(pf, (x[3i-2], x[3i-1], x[3i]), (x[3n+3i-2], x[3n+3i-1], x[3n+3i]), x[6n+i])
    end
    FLOWVPM.nextstep(pf, 0.01)
    s = zero(T)
    for i in 1:n, k in 1:7
        s += w[7(i-1)+k] * pf.particles[k, i]
    end
    return s
end

@testset "Enzyme: UJ_fmm step through the FMM adjoint rule" begin
    n = 1500
    Random.seed!(4)
    L = (n/500)^(1/3)
    x = vcat(L .* rand(3n), 0.1 .* randn(3n) ./ L^2, 0.06 .+ 0.02 .* rand(n))
    w = randn(7n)
    sfs = FLOWVPM.ConstantSFS(FLOWVPM.Estr_fmm; Cs=1.0, clippings=(FLOWVPM.clipping_backscatter,))
    grad(UJ, SFS) = (g = zero(x); Enzyme.autodiff(Enzyme.Reverse, enz_fmm_step, Enzyme.Active, Enzyme.Duplicated(x, g),
                                                  Enzyme.Const(w), Enzyme.Const(UJ), Enzyme.Const(SFS)); g)
    for SFS in (FLOWVPM.SFS_none, sfs)
        g = grad(FLOWVPM.UJ_fmm, SFS)
        @test all(isfinite, g)
        f = x -> enz_fmm_step(x, w, FLOWVPM.UJ_fmm, SFS)
        v = randn(7n); v ./= LinearAlgebra.norm(v); gv = LinearAlgebra.dot(g, v)
        @test minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - gv)/abs(gv), (1e-4, 1e-5, 1e-6)) < 1e-3
    end
end
