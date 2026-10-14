using Test
import Random
import FLOWVPM
vpm = FLOWVPM

# Core-spreading RBF solve (rbf_conjugategradient): A*Gamma = b with
# A_ij = zeta(r_ij/sigma)/sigma^3 and b the target vorticity in M[7:9].
# Guards the initial guess (the strengths on entry, not omega*vol, which is 0 for
# particles added without a volume) and the stopping test (||r||/||b||, not ||r||/||r0||).

function cs_field(; n=4, h=1.0, sgm=1.3, seed=1)
    rng = Random.MersenneTwister(seed)
    pfield = vpm.ParticleField(n^3, Float64)
    for i in 1:n, j in 1:n, k in 1:n
        X = h .* ([i, j, k] .+ 0.1 .* (rand(rng, 3) .- 0.5))
        vpm.add_particle(pfield, X, zeros(3), sgm)    # vol = 0, as every shed particle
    end
    return pfield
end

get_Gammas(pfield) = [vpm.get_Gamma(pfield, j)[i] for i in 1:3, j in 1:vpm.get_np(pfield)]
set_Gammas!(pfield, G) = for (j, P) in enumerate(vpm.iterator(pfield)); vpm.get_Gamma(P) .= G[:, j]; end

# A*G, evaluated by zeta_direct
function cs_apply(pfield, G)
    set_Gammas!(pfield, G)
    vpm.zeta_direct(pfield)
    return [vpm.get_J(pfield, j)[i] for i in 1:3, j in 1:vpm.get_np(pfield)]
end

function cs_solve!(pfield, b, G0; tol=1e-6, itmax=200)
    ncalls = Ref(0)
    zeta = pf -> (ncalls[] += 1; vpm.zeta_direct(pf))
    cs = vpm.CoreSpreading(0.0, 1.3, zeta; tol, itmax)
    for (j, P) in enumerate(vpm.iterator(pfield))
        vpm.get_M(P)[7:9] .= b[:, j]
    end
    set_Gammas!(pfield, G0)
    vpm.rbf_conjugategradient(pfield, cs)
    return cs, ncalls[]
end

relres(pfield, G, b) = [sqrt(sum(abs2, cs_apply(pfield, G)[i, :] .- b[i, :]) / sum(abs2, b[i, :])) for i in 1:3]

@testset "Core-spreading RBF solve" begin
    pfield = cs_field()
    rng = Random.MersenneTwister(2)
    Gtrue = randn(rng, 3, vpm.get_np(pfield))
    b = cs_apply(pfield, Gtrue)
    tol = 1e-6

    # From zero: true residual meets ||A*Gamma - b||/||b|| <= tol, and the
    # residual the solver reports is that same quantity
    cs, n0 = cs_solve!(pfield, b, zeros(size(Gtrue)); tol)
    G = get_Gammas(pfield)
    rr = relres(pfield, G, b)
    @test all(rr .<= tol)
    @test sqrt.(cs.rrs ./ cs.rr0s) ≈ rr rtol=1e-4

    # Entry strengths already solve the system: one zeta evaluation, no iteration
    cs, n1 = cs_solve!(pfield, b, Gtrue; tol)
    @test n1 == 1
    @test get_Gammas(pfield) == Gtrue

    # A good initial guess is used (fewer iterations than from zero) and the
    # target stays ||r||/||b||, not the stricter ||r||/||r0||
    Gnear = Gtrue .+ 1e-3 .* randn(rng, size(Gtrue)...)
    cs, n2 = cs_solve!(pfield, b, Gnear; tol)
    @test n2 < n0
    @test all(relres(pfield, get_Gammas(pfield), b) .<= tol)
end

# zeta_fmm (direct list of the FMM tree, far field neglected) matches zeta_direct
# and overwrites, not accumulates, what J[1:3] held
@testset "Core-spreading zeta_fmm vs zeta_direct" begin
    pfield = cs_field(; n=8)
    rng = Random.MersenneTwister(4)
    G = randn(rng, 3, vpm.get_np(pfield))
    Wd = cs_apply(pfield, G)
    for j in 1:vpm.get_np(pfield)
        vpm.get_J(pfield, j)[1:3] .= randn(rng, 3)
    end
    vpm.zeta_fmm(pfield)
    Wf = [vpm.get_J(pfield, j)[i] for i in 1:3, j in 1:vpm.get_np(pfield)]
    @test sqrt(sum(abs2, Wf .- Wd) / sum(abs2, Wd)) < 1e-3
end
