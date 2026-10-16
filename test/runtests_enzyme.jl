using Test
import Random
import LinearAlgebra
import FLOWVPM
import Enzyme
import ForwardDiff

# Enzyme reverse and forward mode (default settings) through one UJ_direct time
# step: dot-product test of the gradient against central FD along random
# directions, forward tangents against the gradient (repeated: deterministic),
# and ForwardDiff against the gradient.
# The field is built inside the differentiated function (FLOWVPMEnzymeExt marks
# its filament workspaces and the ρ/σ tuning inactive). Objective: weights on
# [X; Gamma; sigma] after the step; inputs x = [X; Gamma; sigma] before it.
# Four configurations cover every option of each axis at least once.

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

@testset verbose=true "Enzyme: one UJ_direct step" begin
    pedr = FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3)
    cpedr = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3)
    constsfs = FLOWVPM.ConstantSFS(FLOWVPM.Estr_fmm; Cs=1.0, clippings=(FLOWVPM.clipping_backscatter,))
    cfgs = (
        (name="classic, gaussianerf, pedrizzetti, constant SFS, euler", formulation=FLOWVPM.formulation_classic,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=pedr, relax=true, SFS=constsfs, integration=FLOWVPM.euler, fwd=true),
        (name="rVPM, winckelmans, correctedpedrizzetti, dynamic SFS, rungekutta3", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_winckelmans, relaxation=cpedr, relax=true, SFS=FLOWVPM.SFS_Cd_twolevel_nobackscatter,
         integration=FLOWVPM.rungekutta3, fwd=true),
        (name="rVPM, gaussianerf, no relaxation, no SFS, rungekutta3", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=cpedr, relax=false, SFS=FLOWVPM.SFS_none,
         integration=FLOWVPM.rungekutta3, fwd=false),
        (name="rVPM, gaussianerf, pedrizzetti, two-level dynamic SFS, euler", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=pedr, relax=true,
         SFS=FLOWVPM.SFS_Cd_twolevel_analytic_nobackscatter, integration=FLOWVPM.euler, fwd=false),
    )
    n = ENZ_N
    Random.seed!(1)
    x = vcat(rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
    w = randn(7n)
    for cfg in cfgs
        @testset "$(cfg.name)" begin
            f = x -> enz_step(x, w, cfg)
            g = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, enz_step, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w), Enzyme.Const(cfg))
            @test all(isfinite, g)
            for k in 1:3
                v = randn(7n); v ./= LinearAlgebra.norm(v)
                gv = LinearAlgebra.dot(g, v)
                # FD truncation error is O(h^2), roundoff O(1/h): best of four steps
                err = minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - gv)/abs(gv), (1e-4, 1e-5, 1e-6, 1e-7))
                @test err < 1e-5
            end
            v = randn(7n)
            if cfg.fwd   # two configurations (both integrators and formulations), to bound compile time
                fwd = [Enzyme.autodiff(Enzyme.Forward, enz_step, Enzyme.Duplicated(x, v), Enzyme.Const(w), Enzyme.Const(cfg))[1]
                       for _ in 1:3]
                @test all(==(fwd[1]), fwd)
                @test isapprox(fwd[1], LinearAlgebra.dot(g, v); rtol=1e-10)
                @test minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - fwd[1])/abs(fwd[1]), (1e-4, 1e-5, 1e-6, 1e-7)) < 1e-5
            end
            @test isapprox(ForwardDiff.derivative(t -> f(x .+ t .* v), 0.0), LinearAlgebra.dot(g, v); rtol=1e-10)
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

# Many steps with checkpoints (FLOWVPM.checkpointed_gradient!), with the freestream
# as a design variable: a typed AmbientHolder in the field carries the uniform
# velocity's parameter, whose gradient lands in the shadow field.
struct EnzUinf{T}
    U::T
end
(u::EnzUinf)(t) = FLOWVPM.SVector(u.U, u.U / 2, zero(u.U))

@testset "Enzyme: checkpointed_gradient! over many steps, freestream parameter" begin
    n = 30; nsteps = 20; dt = 0.01
    Random.seed!(5)
    P0 = vcat(rand(3, n), 0.1 .* randn(3, n), 0.15 .+ 0.02 .* rand(1, n))
    rows = vcat(FLOWVPM.X_INDEX, FLOWVPM.GAMMA_INDEX, FLOWVPM.SIGMA_INDEX)
    W = randn(7, n)
    function build(P0, U)
        pf = FLOWVPM.ParticleField(n, Float64; formulation=FLOWVPM.formulation_rVPM, kernel=FLOWVPM.kernel_gaussianerf,
                                   UJ=FLOWVPM.UJ_direct, Uinf=FLOWVPM.AmbientHolder(FLOWVPM.UniformAmbient(EnzUinf(U))),
                                   integration=FLOWVPM.rungekutta3, fmm=FLOWVPM.FMM(autotune_reg_error=false))
        for i in 1:n
            FLOWVPM.add_particle(pf, (P0[1, i], P0[2, i], P0[3, i]), (P0[4, i], P0[5, i], P0[6, i]), P0[7, i])
        end
        return pf
    end
    function obj(pf)
        s = 0.0
        for i in 1:pf.np, k in 1:7
            s += W[k, i] * pf.particles[rows[k], i]
        end
        return s
    end
    step(pf, dt) = FLOWVPM.nextstep(pf, dt; relax=true)
    run(pf) = (for _ in 1:nsteps; step(pf, dt); end; obj(pf))
    F(P0, U) = run(build(P0, U))
    U = 0.8
    pf = build(P0, U); dpf = Enzyme.make_zero(pf)
    J = FLOWVPM.checkpointed_gradient!(dpf, obj, pf, dt, nsteps; every=5, step)
    @test J == F(P0, U)
    @test obj(pf) == J && pf.nt == nsteps                 # left at the final state
    # same numbers as Enzyme over the whole loop (all steps on one tape; bit-equal
    # at -t 1, threaded reverse accumulation reorders sums at -t > 1)
    pf2 = build(P0, U); dpf2 = Enzyme.make_zero(pf2)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(run), Enzyme.Active, Enzyme.Duplicated(pf2, dpf2))
    @test isapprox(dpf.particles[rows, 1:n], dpf2.particles[rows, 1:n]; rtol=1e-12)
    @test isapprox(dpf.Uinf.ambient.velocity.U, dpf2.Uinf.ambient.velocity.U; rtol=1e-12)
    # against central FD: two directions in the initial particles, and the freestream
    G = dpf.particles[rows, 1:n]
    for k in 1:2
        V = randn(7, n); V ./= LinearAlgebra.norm(V)
        gv = sum(G .* V)
        @test minimum(h -> abs((F(P0 .+ h .* V, U) - F(P0 .- h .* V, U))/(2h) - gv)/abs(gv), (1e-4, 1e-5, 1e-6, 1e-7)) < 1e-6
    end
    dU = dpf.Uinf.ambient.velocity.U
    @test minimum(h -> abs((F(P0, U + h) - F(P0, U - h))/(2h) - dU)/abs(dU), (1e-4, 1e-5, 1e-6)) < 1e-6
end
