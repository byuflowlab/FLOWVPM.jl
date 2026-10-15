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
