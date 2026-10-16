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
# coarse for FD), so the stored-Gram gradient is checked against FD at 1e-5.
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

    # `local`: a closure assigning `dx` would rebind the testset's dx below
    grad(pf, T) = (local g = zero(x0); Enzyme.autodiff(Enzyme.Reverse, csad_f, Enzyme.Active, Enzyme.Duplicated(x0, g),
                   Enzyme.Duplicated(pf, Enzyme.make_zero(pf)), Enzyme.Const(T), Enzyme.Const(sub), Enzyme.Const(w)); g)
    dx = grad(pf, T)
    dxs = grad(csad_field(false; gram_mem=2^30)...)    # stored (Float32) Gram pairs
    @test all(isfinite, dx) && all(isfinite, dxs)
    @test any(!iszero, dx[1:3N]) && any(!iszero, dx[3N+1:6N]) && any(!iszero, dx[6N+1:end])

    # X and Gamma through the reset run in the full-step test below; this one keeps
    # what is unique to it: the subset reset called directly, small preconditioner
    # blocks, and the stored-Gram adjoint (its Float32 pairs: ~1e-7 from FD)
    for blk in (:all, :sigma)
        v = randn(rng, 7N)
        blk == :sigma && (v[1:6N] .= 0)
        v ./= LinearAlgebra.norm(v)
        fd = map((1e-4, 1e-5, 1e-6)) do h
            fp = csad_f(x0 .+ h .* v, pfd, Td, sub, w); np1 = FLOWVPM.get_np(pfd)
            fm = csad_f(x0 .- h .* v, pfd, Td, sub, w); np2 = FLOWVPM.get_np(pfd)
            np1 == np2 == np0 ? (fp - fm) / 2h : Inf
        end
        # error relative to the gradient's norm in the block (|v| = 1; a random
        # direction can make the directional derivative itself small)
        err(d) = minimum(q -> abs(LinearAlgebra.dot(d, v) - q), fd) / LinearAlgebra.norm(d .* (v .!= 0))
        @info "Galerkin reset AD vs FD" blk err(dx) err(dxs)
        @test err(dx) <= 1e-6
        @test err(dxs) <= 1e-5
    end
end

# Reverse-mode AD through full time steps of a CoreSpreading field (UJ_direct, rVPM,
# Pedrizzetti relaxation): core growth and a Galerkin reset inside the differentiated
# steps, Euler with SFS (timer reset) and RK3 without (growth_beta reset). Enzyme vs
# central FD.
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
    # nu = 0.05: steps of 0.1 grow sqrt(2 nu t) / sgm0 to 1.005, 1.00995 (beta 1.008:
    # reset in step 2), then 1.005.
    # growth_beta 1.03: the largest core (1.05 sgm0) resets in step 1, not again in step 2
    # Euler with SFS: timer reset in step 2 of 3; RK3 without SFS: reset by the largest
    # core (growth_beta), not the timer, in step 1 of 2
    cfgs = ((integ=FLOWVPM.euler, sfs=true, nsteps=3, dt=0.1, beta=1.008, growth_beta=0.0),
            (integ=FLOWVPM.rungekutta3, sfs=false, nsteps=2, dt=0.1, beta=10.0, growth_beta=1.03))
    for cfg in cfgs
        @testset "$(nameof(cfg.integ)), SFS $(cfg.sfs), $(cfg.nsteps) step(s)$(cfg.growth_beta > 0 ? ", growth_beta" : "")" begin
            kw = (; nu=0.05, cfg.beta, cfg.growth_beta)
            pf, T = csstep_field(false, cfg.integ, cfg.sfs; kw...)
            pfd, Td = csstep_field(true, cfg.integ, cfg.sfs; kw...)
            f0 = csstep_f(x0, pf, T, cfg.nsteps, cfg.dt); np0 = FLOWVPM.get_np(pf)
            # the reset fired inside the steps: count changed, timer restarted at it
            @test np0 != N
            @test pf.viscous.t_sgm ≈ cfg.dt                 # reset in step 2 of 3 (timer) or 1 of 2 (growth_beta)
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

# The same field through FLOWVPM.checkpointed_gradient!, with the freestream as a
# design variable (a typed AmbientHolder carries the uniform velocity's parameter,
# whose gradient lands in the shadow field): 4 steps, a checkpoint every 2, resets in
# steps 2 and 4, so the particle count differs between checkpoints (48 -> 155 -> 308).
# Against central FD (exact-solve oracle): positions, strengths and the freestream.
struct EnzUinf{T}
    U::T
end
(u::EnzUinf)(t) = FLOWVPM.SVector(u.U, u.U / 2, zero(u.U))

@testset "checkpointed_gradient! across resets (np changes between checkpoints), freestream parameter" begin
    N = CSSTEP_N; nsteps = 4; dt = 0.1; U = 0.8
    function cs_build(x, U, exact)
        gr = exact ? FLOWVPM.GalerkinReset(; block_cell=1e3, block_cap=10^6, tol=1e-10, itmax=50, gram_mem=0) :
                     FLOWVPM.GalerkinReset(; tol=1e-10, itmax=2000, gram_mem=0)
        pf = FLOWVPM.ParticleField(8N, Float64; viscous=FLOWVPM.CoreSpreading(0.05, CSAD_SGM0; beta=1.008, galerkin=gr),
                                   UJ=FLOWVPM.UJ_direct, formulation=FLOWVPM.formulation_rVPM, kernel=FLOWVPM.kernel_gaussianerf,
                                   Uinf=FLOWVPM.AmbientHolder(FLOWVPM.UniformAmbient(EnzUinf(U))),
                                   integration=FLOWVPM.rungekutta3, relaxation=FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3),
                                   fmm=FLOWVPM.FMM(autotune_reg_error=false))
        for j in 1:N
            FLOWVPM.add_particle(pf, (x[3j-2], x[3j-1], x[3j]), (x[3N+3j-2], x[3N+3j-1], x[3N+3j]), x[6N+j])
        end
        return pf
    end
    function cs_obj(pf)
        P = pf.particles; s = 0.0
        for j in 1:FLOWVPM.get_np(pf)
            a = csstep_a(P[1, j], P[2, j], P[3, j])
            s += a[1] * P[4, j] + a[2] * P[5, j] + a[3] * P[6, j] + a[4] * P[7, j]
        end
        return s
    end
    cs_step(pf, dt) = FLOWVPM.nextstep(pf, dt; relax=true)
    cs_run(pf) = (for _ in 1:nsteps; cs_step(pf, dt); end; cs_obj(pf))
    rng = Random.MersenneTwister(5)
    X0 = Float64[]
    for i in 1:4, j in 1:4, k in 1:3, c in (i, j, k)
        push!(X0, 10 + 0.6c + 0.1 * (rand(rng) - 0.5))
    end
    X0[4:6] .= X0[1:3] .+ (0.05, 0.05, 0.3)
    x0 = vcat(X0, 0.1 .* randn(rng, 3N), CSAD_SGM0 .* (1.0 .+ 0.05 .* rand(rng, N)))
    pf = cs_build(x0, U, false); dpf = Enzyme.make_zero(pf)
    J = FLOWVPM.checkpointed_gradient!(dpf, cs_obj, pf, dt, nsteps; every=2, step=cs_step)
    @test FLOWVPM.get_np(pf) > 2N                          # resets inside the run
    @test J == cs_run(cs_build(x0, U, false))
    @test cs_obj(pf) == J && pf.nt == nsteps               # left at the final state
    rows = vcat(FLOWVPM.X_INDEX, FLOWVPM.GAMMA_INDEX, FLOWVPM.SIGMA_INDEX)
    G = dpf.particles[rows, 1:N]
    dx = vcat(vec(G[1:3, :]), vec(G[4:6, :]), vec(G[7, :]))
    F(x, U) = (p = cs_build(x, U, true); (cs_run(p), FLOWVPM.get_np(p)))
    np0 = F(x0, U)[2]
    for blk in (:X, :Gamma)
        v = randn(rng, 7N)
        blk == :X ? (v[3N+1:end] .= 0) : (v[1:3N] .= 0; v[6N+1:end] .= 0)
        v ./= LinearAlgebra.norm(v)
        gv = LinearAlgebra.dot(dx, v)
        err = minimum((1e-4, 1e-5, 1e-6)) do h
            (fp, n1) = F(x0 .+ h .* v, U); (fm, n2) = F(x0 .- h .* v, U)
            n1 == n2 == np0 || return Inf
            abs(gv - (fp - fm) / 2h) / LinearAlgebra.norm(dx .* (v .!= 0))
        end
        @test err <= 1e-6
    end
    dU = dpf.Uinf.ambient.velocity.U
    errU = minimum((1e-4, 1e-5, 1e-6)) do h
        (fp, n1) = F(x0, U + h); (fm, n2) = F(x0, U - h)
        n1 == n2 == np0 || return Inf
        abs((fp - fm) / 2h - dU) / abs(dU)
    end
    @info "checkpointed freestream AD vs FD" errU
    @test errU <= 1e-6
end
