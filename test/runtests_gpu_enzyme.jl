# Enzyme reverse mode through device (CUDA) time steps on the direct path, against
# central finite differences (ext/FLOWVPMEnzymeGPUExt.jl: every device operation of
# a step has a reverse rule). Included by runtests.jl when CUDA is functional.
# The step mutates the field in place: its cotangent is seeded with the weights W
# on rows 1:7 of every column and comes back as the gradient with respect to the
# state before the step (the objective is the weighted sum over the whole particle
# matrix, so it is defined when a reset changes the particle count; the order is
# deterministic under the FD steps). FD weighs the same on the host.
using Test, Random, LinearAlgebra
import CUDA, Enzyme, FLOWVPM
const gvpm = FLOWVPM

@testset "Enzyme on CUDA fields (direct path) vs FD" begin
    N = 48; MAXP = 4N
    rng = Random.MersenneTwister(7)
    X0 = vcat(rand(rng, 3, N) .* (N / 100)^(1/3), 0.05 .* randn(rng, 3, N), 0.25 .+ 0.1 .* rand(rng, 1, N))
    W = randn(rng, 7, MAXP)
    function field(x; kw...)
        pf = gvpm.ParticleField(MAXP, Float64; arraytype = CUDA.CuArray, UJ = gvpm.UJ_direct,
                                formulation = gvpm.formulation_rVPM, kernel = gvpm.kernel_gaussianerf, kw...)
        blk = zeros(gvpm.nfields, N); blk[1:7, :] .= x; blk[gvpm.CIRCULATION_INDEX, :] .= 1
        gvpm.add_particles!(pf, blk)
        return pf
    end
    weigh(pf) = sum(Array(pf.particles)[1:7, :] .* W)
    runop(op, pf) = (op(pf); nothing)
    function check(kw, op)
        pf = field(X0; kw...); dpf = Enzyme.make_zero(pf)
        copyto!(view(dpf.particles, 1:7, :), CUDA.CuArray(W))
        Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), runop, Enzyme.Const, Enzyme.Const(op),
                        Enzyme.Duplicated(pf, dpf))
        g = Array(dpf.particles)[1:7, 1:N]
        worst = 0.0
        for _ in 1:2
            v = randn(rng, 7, N); v ./= norm(v); h = 1e-6
            fp = (p = field(X0 .+ h .* v; kw...); op(p); weigh(p))
            fm = (p = field(X0 .- h .* v; kw...); op(p); weigh(p))
            worst = max(worst, abs(dot(g, v) - (fp - fm) / 2h) / norm(g))
        end
        return worst
    end
    dt = 0.01
    relax = gvpm.Relaxation(gvpm.relax_pedrizzetti, 1, 0.3)
    cs = gvpm.CoreSpreading(1e-3, 0.25; beta = 10.0)
    @test check((; integration = gvpm.euler, viscous = cs), pf -> gvpm.nextstep(pf, dt)) < 1e-6
    @test check((; relaxation = relax, SFS = gvpm.SFS_Cs_nobackscatter, viscous = cs),
                pf -> (gvpm.nextstep(pf, dt); gvpm.nextstep(pf, dt))) < 1e-6
    @test check((; relaxation = relax, SFS = gvpm.SFS_Cd_twolevel_analytic_nobackscatter),
                pf -> gvpm.nextstep(pf, dt)) < 1e-6
    # one preconditioner block over every unknown: the reset's CG solves exactly in one
    # iteration, so FD does not see the stopping rule (an iterative solve put ~1e-6 of
    # stopping noise into FD)
    exact = gvpm.GalerkinReset(; block_cell = 1e3, block_cap = 10^6, tol = 1e-10, itmax = 50)
    @test check((; viscous = gvpm.CoreSpreading(0.2, 0.25; beta = 1.02, galerkin = exact)), pf -> gvpm.nextstep(pf, dt)) < 1e-6
end
