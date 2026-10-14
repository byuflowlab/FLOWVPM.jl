using Test
import Random
import LinearAlgebra
import FLOWVPM
import Enzyme
import ImplicitAD

# Reverse-mode AD through one core-spreading reset with rbf = rbf_implicitad
# (implicit-function rule, FLOWVPMImplicitADExt): Enzyme gradient vs central FD.
# FD oracle: the same reset with an exact dense Cholesky solve of the same A
# (FD of the CG primal is not usable: its stopping test makes it jump).
# Objective: weights on [X; Gamma] after the reset; inputs x = [X; Gamma_old; sigma].

const CSAD_N = 27
const CSAD_SGM0 = 1.0

function csad_rbf_dense(pfield, cs)
    n = pfield.np; zeta = pfield.kernel.zeta; s0 = cs.sgm0
    A = [zeta(sqrt(sum(abs2, FLOWVPM.get_X(pfield, i) .- FLOWVPM.get_X(pfield, j)))/s0)/s0^3 for i in 1:n, j in 1:n]
    B = [FLOWVPM.get_M(pfield, i)[6+k] for i in 1:n, k in 1:3]
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A)) \ B
    for i in 1:n, k in 1:3
        FLOWVPM.get_Gamma(pfield, i)[k] = G[i, k]
    end
    return nothing
end

# The field is built outside the differentiated function (the ParticleField
# constructor is not Enzyme-differentiable, FilamentEdgeGraph BitArray); csad_f
# overwrites every input slot from x, so its shadow starts from zero each call.
function csad_field(rbf, tol)
    cs = FLOWVPM.CoreSpreading(1e-2, CSAD_SGM0, FLOWVPM.zeta_direct_singlethreaded;
                               rbf=rbf, tol=tol, itmax=500, beta=1.5)
    pf = FLOWVPM.ParticleField(CSAD_N, Float64; viscous=cs, UJ=FLOWVPM.UJ_direct)
    for i in 1:CSAD_N
        FLOWVPM.add_particle(pf, zeros(3), zeros(3), 1.0)
    end
    return pf
end

function csad_reset!(pf, x)
    N = CSAD_N
    for i in 1:N
        for k in 1:3
            FLOWVPM.get_X(pf, i)[k] = x[3i-3+k]
            FLOWVPM.get_Gamma(pf, i)[k] = x[3N+3i-3+k]
        end
        FLOWVPM.get_sigma(pf, i)[] = x[6N+i]
    end
    FLOWVPM._corespreading_reset!(pf, pf.viscous)
    return pf
end

function csad_f(x, pf, w)
    csad_reset!(pf, x)
    s = zero(eltype(x)); N = CSAD_N
    for i in 1:N
        X = FLOWVPM.get_X(pf, i); G = FLOWVPM.get_Gamma(pf, i)
        s += w[3i-2]*X[1] + w[3i-1]*X[2] + w[3i]*X[3] + w[3N+3i-2]*G[1] + w[3N+3i-1]*G[2] + w[3N+3i]*G[3]
    end
    return s
end

@testset "CoreSpreading reset: implicit-AD rule (Enzyme reverse vs FD)" begin
    N = CSAD_N
    rng = Random.MersenneTwister(3)
    X0 = Float64[]
    for i in 1:3, j in 1:3, k in 1:3, c in (i, j, k)
        push!(X0, c + 0.15*(rand(rng) - 0.5))
    end
    x0 = vcat(X0, 0.1 .* randn(rng, 3N), 1.15 .+ 0.1 .* rand(rng, N))
    w = randn(rng, 6N)
    tol = 1e-10
    pf = csad_field(FLOWVPM.rbf_implicitad, tol)
    pfd = csad_field(csad_rbf_dense, tol)

    # primal: bit-for-bit the stock CG
    G(pf) = [FLOWVPM.get_Gamma(pf, i)[k] for k in 1:3, i in 1:N]
    @test G(csad_reset!(pf, x0)) == G(csad_reset!(csad_field(FLOWVPM.rbf_conjugategradient, tol), x0))
    # and the CG at this tol is the exact solve
    @test isapprox(csad_f(x0, pf, w), csad_f(x0, pfd, w); rtol=1e-9)

    dx = zero(x0)
    Enzyme.autodiff(Enzyme.Reverse, csad_f, Enzyme.Active, Enzyme.Duplicated(x0, dx),
                    Enzyme.Duplicated(pf, Enzyme.make_zero(pf)), Enzyme.Const(w))
    @test all(isfinite, dx)
    @test any(!iszero, dx[1:3N]) && any(!iszero, dx[3N+1:6N]) && any(!iszero, dx[6N+1:end])

    h = 1e-6
    for blk in (:all, :X, :Gamma, :sigma)
        v = randn(rng, 7N)
        blk == :X && (v[3N+1:end] .= 0)
        blk == :Gamma && (v[1:3N] .= 0; v[6N+1:end] .= 0)
        blk == :sigma && (v[1:6N] .= 0)
        v ./= LinearAlgebra.norm(v)
        fd = (csad_f(x0 .+ h .* v, pfd, w) - csad_f(x0 .- h .* v, pfd, w)) / (2h)
        gv = LinearAlgebra.dot(dx, v)
        @test isapprox(gv, fd; rtol=1e-6)
    end
end
