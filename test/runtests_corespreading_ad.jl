using Test
import Random
import LinearAlgebra
import FLOWVPM
import Enzyme

# Reverse-mode AD through one Galerkin core-size reset (implicit rule for the linear
# solve, ext/FLOWVPMEnzymeExt.jl): Enzyme gradient vs central FD.
# Field: a perturbed 4x4x3 lattice (spacing 0.6 sgm0), cores 1.0-1.4 sgm0; the cores
# above 1.2 sgm0 are reset, so the solve has kept and thinned subset particles, a halo
# and edge points. The selection does not change under the FD steps (same np).
# FD oracle: the same reset with one preconditioner block over every unknown (a dense
# exact solve; FD through the CG's stopping rule would jump). The AD run uses small
# blocks and a tight tol, so both the primal and the adjoint CG iterate. Both evaluate
# the Gram matrix on the fly: its stored pairs are Float32 (steps of ~1e-7 in M, too
# coarse for FD), checked separately against the on-the-fly gradient.
# Objective: weights on [X; Gamma; sigma] of the field after the reset; inputs
# x = [X; Gamma; sigma] before.

const CSAD_N = 48
const CSAD_SGM0 = 1.0

function csad_field(exact; gram_mem=0)
    gr = exact ? FLOWVPM.GalerkinReset(; block_cell=1e3, block_cap=10^6, tol=1e-6, itmax=50, gram_mem) :
                 FLOWVPM.GalerkinReset(; block_cap=8, tol=1e-9, itmax=2000, gram_mem)
    cs = FLOWVPM.CoreSpreading(0.0, CSAD_SGM0; galerkin=gr)
    pf = FLOWVPM.ParticleField(4CSAD_N, Float64; viscous=cs, UJ=FLOWVPM.UJ_direct)
    for i in 1:CSAD_N
        FLOWVPM.add_particle(pf, zeros(3), zeros(3), 1.0)
    end
    return pf, copy(pf.particles[:, 1:CSAD_N])
end

# The field is built outside the differentiated function (the ParticleField
# constructor is not Enzyme-differentiable); csad_f restores the first N columns from
# the template T and the inputs x, so every call starts from the same state.
function csad_f(x, pf, T, sub, w)
    N = CSAD_N; P = pf.particles
    pf.np = N
    for j in 1:N
        for r in axes(T, 1)
            P[r, j] = T[r, j]
        end
        for k in 1:3
            P[FLOWVPM.X_INDEX[k], j] = x[3j-3+k]
            P[FLOWVPM.GAMMA_INDEX[k], j] = x[3N+3j-3+k]
        end
        P[FLOWVPM.SIGMA_INDEX, j] = x[6N+j]
    end
    FLOWVPM.corespreading_reset_subset!(pf, pf.viscous, sub)
    s = zero(eltype(x))
    for j in 1:FLOWVPM.get_np(pf), k in 1:7
        s += w[k, j] * P[k, j]
    end
    return s
end

@testset "Galerkin reset: implicit AD rule (Enzyme reverse vs FD)" begin
    N = CSAD_N
    rng = Random.MersenneTwister(3)
    X0 = Float64[]
    for i in 1:4, j in 1:4, k in 1:3, c in (i, j, k)
        push!(X0, 10 + 0.6c + 0.1 * (rand(rng) - 0.5))   # positive: one block in the oracle
    end
    S0 = 1.0 .+ 0.4 .* rand(rng, N)
    x0 = vcat(X0, 0.1 .* randn(rng, 3N), S0)
    sub = findall(>(1.2), S0)
    w = randn(rng, 7, 4N)
    pf, T = csad_field(false)
    pfd, Td = csad_field(true)

    f0 = csad_f(x0, pf, T, sub, w); np0 = FLOWVPM.get_np(pf)
    @test np0 != N                                     # edge points in or thinned out
    @test isapprox(f0, csad_f(x0, pfd, Td, sub, w); rtol=1e-10)

    grad(pf, T) = (dx = zero(x0); Enzyme.autodiff(Enzyme.Reverse, csad_f, Enzyme.Active, Enzyme.Duplicated(x0, dx),
                   Enzyme.Duplicated(pf, Enzyme.make_zero(pf)), Enzyme.Const(T), Enzyme.Const(sub), Enzyme.Const(w)); dx)
    dx = grad(pf, T)
    @test all(isfinite, dx)
    @test any(!iszero, dx[1:3N]) && any(!iszero, dx[3N+1:6N]) && any(!iszero, dx[6N+1:end])
    @test isapprox(grad(csad_field(false; gram_mem=2^30)...), dx; rtol=1e-4)     # stored Gram pairs

    for blk in (:all, :X, :Gamma, :sigma)
        v = randn(rng, 7N)
        blk == :X && (v[3N+1:end] .= 0)
        blk == :Gamma && (v[1:3N] .= 0; v[6N+1:end] .= 0)
        blk == :sigma && (v[1:6N] .= 0)
        v ./= LinearAlgebra.norm(v)
        gv = LinearAlgebra.dot(dx, v)
        # error relative to the gradient's norm in the block (|v| = 1; a random
        # direction can make gv itself small)
        err = minimum((1e-4, 1e-5, 1e-6)) do h
            fp = csad_f(x0 .+ h .* v, pfd, Td, sub, w); np1 = FLOWVPM.get_np(pfd)
            fm = csad_f(x0 .- h .* v, pfd, Td, sub, w); np2 = FLOWVPM.get_np(pfd)
            np1 == np2 == np0 || return Inf
            abs(gv - (fp - fm) / 2h) / LinearAlgebra.norm(dx .* (v .!= 0))
        end
        @info "Galerkin reset AD vs FD" blk err
        @test err <= 1e-6
    end
end

# Reverse-mode AD through full time steps of a CoreSpreading field (UJ_direct, rVPM,
# Pedrizzetti relaxation): core growth and a Galerkin reset of every particle inside
# the differentiated steps, Euler and RK3, SFS on and off. Enzyme vs central FD.
# The reset changes the particle count (edge points added, one particle of a close
# pair thinned) and order, so the objective is a sum over the final particle set of
# a smooth function of each particle's state, sum_j a(X_j) . [Gamma_j; sigma_j]: it
# does not depend on the count or order. The selection is piecewise constant (no
# derivative), so FD steps are used only where they keep the same particle count.
# The AD run iterates the CG (default blocks, tol 1e-10); the FD oracle solves
# exactly (one block), as above.
const CSSTEP_N = 48

function csstep_field(exact, integration, sfs; nu, beta, growth_beta=0.0)
    gr = exact ? FLOWVPM.GalerkinReset(; block_cell=1e3, block_cap=10^6, tol=1e-10, itmax=50, gram_mem=0) :
                 FLOWVPM.GalerkinReset(; tol=1e-10, itmax=2000, gram_mem=0)
    cs = FLOWVPM.CoreSpreading(nu, CSAD_SGM0; beta, growth_beta, galerkin=gr)
    SFS = sfs ? FLOWVPM.ConstantSFS(FLOWVPM.Estr_fmm; Cs=1.0, clippings=(FLOWVPM.clipping_backscatter,)) : FLOWVPM.SFS_none
    pf = FLOWVPM.ParticleField(4CSSTEP_N, Float64; viscous=cs, UJ=FLOWVPM.UJ_direct, formulation=FLOWVPM.formulation_rVPM,
                               kernel=FLOWVPM.kernel_gaussianerf, SFS, integration,
                               relaxation=FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3),
                               fmm=FLOWVPM.FMM(autotune_reg_error=false))
    for i in 1:CSSTEP_N
        FLOWVPM.add_particle(pf, zeros(3), zeros(3), 1.0)
    end
    return pf, copy(pf.particles[:, 1:CSSTEP_N])
end

@inline csstep_a(X1, X2, X3) = (cos(0.7X1 + 0.2X2), sin(0.5X2 - 0.3X3), cos(0.4X3 + 0.6X1), sin(0.3X1 + 0.8X3))

# restores the field (state, clock, growth timer) from T and x, then steps
function csstep_f(x, pf, T, nsteps, dt)
    N = CSSTEP_N; P = pf.particles
    pf.np = N; pf.t = 0.0; pf.nt = 0; pf.viscous.t_sgm = 0.0
    for j in 1:N
        for r in axes(T, 1)
            P[r, j] = T[r, j]
        end
        for k in 1:3
            P[FLOWVPM.X_INDEX[k], j] = x[3j-3+k]
            P[FLOWVPM.GAMMA_INDEX[k], j] = x[3N+3j-3+k]
        end
        P[FLOWVPM.SIGMA_INDEX, j] = x[6N+j]
    end
    for _ in 1:nsteps
        FLOWVPM.nextstep(pf, dt; relax=true)
    end
    s = zero(eltype(x))
    for j in 1:FLOWVPM.get_np(pf)
        a = csstep_a(P[1, j], P[2, j], P[3, j])
        s += a[1] * P[4, j] + a[2] * P[5, j] + a[3] * P[6, j] + a[4] * P[7, j]
    end
    return s
end

@testset verbose=true "Core spreading + Galerkin reset: full steps (Enzyme reverse vs FD)" begin
    N = CSSTEP_N
    rng = Random.MersenneTwister(5)
    X0 = Float64[]
    for i in 1:4, j in 1:4, k in 1:3, c in (i, j, k)
        push!(X0, 10 + 0.6c + 0.1 * (rand(rng) - 0.5))
    end
    X0[4:6] .= X0[1:3] .+ (0.05, 0.05, 0.3)            # a close pair: one of it is thinned
    x0 = vcat(X0, 0.1 .* randn(rng, 3N), CSAD_SGM0 .* (1.0 .+ 0.05 .* rand(rng, N)))
    # nu = 0.05: one step of 0.25 grows sqrt(2 nu t) / sgm0 to 1.012 (beta 1.01: reset in
    # step 1); steps of 0.1 reach 1.005, 1.00995 (beta 1.008: reset in step 2), then 1.005.
    # growth_beta 1.03: the largest core (1.05 sgm0) resets in step 1, not again in step 2
    cfgs = ((integ=FLOWVPM.euler, sfs=false, nsteps=1, dt=0.25, beta=1.01, growth_beta=0.0),
            (integ=FLOWVPM.rungekutta3, sfs=true, nsteps=1, dt=0.25, beta=1.01, growth_beta=0.0),
            (integ=FLOWVPM.euler, sfs=true, nsteps=3, dt=0.1, beta=1.008, growth_beta=0.0),
            # reset by the largest core (growth_beta), not the timer
            (integ=FLOWVPM.rungekutta3, sfs=false, nsteps=2, dt=0.1, beta=10.0, growth_beta=1.03))
    for cfg in cfgs
        @testset "$(nameof(cfg.integ)), SFS $(cfg.sfs), $(cfg.nsteps) step(s)$(cfg.growth_beta > 0 ? ", growth_beta" : "")" begin
            kw = (; nu=0.05, cfg.beta, cfg.growth_beta)
            pf, T = csstep_field(false, cfg.integ, cfg.sfs; kw...)
            pfd, Td = csstep_field(true, cfg.integ, cfg.sfs; kw...)
            f0 = csstep_f(x0, pf, T, cfg.nsteps, cfg.dt); np0 = FLOWVPM.get_np(pf)
            # the reset fired inside the steps: count changed, timer restarted at it
            @test np0 != N
            @test pf.viscous.t_sgm ≈ (cfg.nsteps == 1 ? 0.0 : cfg.dt)    # reset in step 1 (1 step, growth_beta) or 2
            @test isapprox(f0, csstep_f(x0, pfd, Td, cfg.nsteps, cfg.dt); rtol=1e-10)
            dx = zero(x0)
            Enzyme.autodiff(Enzyme.Reverse, csstep_f, Enzyme.Active, Enzyme.Duplicated(x0, dx),
                            Enzyme.Duplicated(pf, Enzyme.make_zero(pf)), Enzyme.Const(T), Enzyme.Const(cfg.nsteps),
                            Enzyme.Const(cfg.dt))
            @test all(isfinite, dx)
            for blk in (:all, :X, :Gamma, :sigma)
                v = randn(rng, 7N)
                blk == :X && (v[3N+1:end] .= 0)
                blk == :Gamma && (v[1:3N] .= 0; v[6N+1:end] .= 0)
                blk == :sigma && (v[1:6N] .= 0)
                v ./= LinearAlgebra.norm(v)
                gv = LinearAlgebra.dot(dx, v)
                err = minimum((1e-4, 1e-5, 1e-6)) do h
                    fp = csstep_f(x0 .+ h .* v, pfd, Td, cfg.nsteps, cfg.dt); np1 = FLOWVPM.get_np(pfd)
                    fm = csstep_f(x0 .- h .* v, pfd, Td, cfg.nsteps, cfg.dt); np2 = FLOWVPM.get_np(pfd)
                    np1 == np2 == np0 || return Inf
                    abs(gv - (fp - fm) / 2h) / LinearAlgebra.norm(dx .* (v .!= 0))
                end
                @info "CS step AD vs FD" cfg.integ cfg.sfs cfg.nsteps blk err
                @test err <= 1e-6
            end
        end
    end
end
