using Test
import Random
import FLOWVPM

# Core-spreading RBF solve (rbf_conjugategradient): A*Gamma = b with
# A_ij = zeta(r_ij/sigma)/sigma^3 and b the target vorticity in M[7:9].
# Guards the initial guess (the strengths on entry, not omega*vol, which is 0 for
# particles added without a volume) and the stopping test (||r||/||b||, not ||r||/||r0||).

function cs_field(; n=4, h=1.0, sgm=1.3, seed=1)
    rng = Random.MersenneTwister(seed)
    pfield = FLOWVPM.ParticleField(n^3, Float64)
    for i in 1:n, j in 1:n, k in 1:n
        X = h .* ([i, j, k] .+ 0.1 .* (rand(rng, 3) .- 0.5))
        FLOWVPM.add_particle(pfield, X, zeros(3), sgm)    # vol = 0, as every shed particle
    end
    return pfield
end

get_Gammas(pfield) = [FLOWVPM.get_Gamma(pfield, j)[i] for i in 1:3, j in 1:FLOWVPM.get_np(pfield)]
set_Gammas!(pfield, G) = for (j, P) in enumerate(FLOWVPM.iterator(pfield)); FLOWVPM.get_Gamma(P) .= G[:, j]; end

# A*G, evaluated by zeta_direct (into the vorticity slot)
function cs_apply(pfield, G)
    set_Gammas!(pfield, G)
    FLOWVPM.zeta_direct(pfield)
    return [FLOWVPM.get_vorticity(pfield, j)[i] for i in 1:3, j in 1:FLOWVPM.get_np(pfield)]
end

function cs_solve!(pfield, b, G0; tol=1e-6, itmax=200, precondition=true)
    ncalls = Ref(0)
    zeta = pf -> (ncalls[] += 1; FLOWVPM.zeta_direct(pf))
    cs = FLOWVPM.CoreSpreading(0.0, 1.3, zeta; tol, itmax, precondition)
    for (j, P) in enumerate(FLOWVPM.iterator(pfield))
        FLOWVPM.get_M(P)[7:9] .= b[:, j]
    end
    set_Gammas!(pfield, G0)
    FLOWVPM.rbf_conjugategradient(pfield, cs)
    return cs, ncalls[]
end

relres(pfield, G, b) = [sqrt(sum(abs2, cs_apply(pfield, G)[i, :] .- b[i, :]) / sum(abs2, b[i, :])) for i in 1:3]

@testset "Core-spreading RBF solve, precondition=$precondition" for precondition in (false, true)
    pfield = cs_field()
    rng = Random.MersenneTwister(2)
    Gtrue = randn(rng, 3, FLOWVPM.get_np(pfield))
    b = cs_apply(pfield, Gtrue)
    tol = 1e-6

    # From zero: true residual meets ||A*Gamma - b||/||b|| <= tol, and the
    # residual the solver reports is that same quantity
    cs, n0 = cs_solve!(pfield, b, zeros(size(Gtrue)); tol, precondition)
    G = get_Gammas(pfield)
    rr = relres(pfield, G, b)
    @test all(rr .<= tol)
    @test sqrt.(cs.rrs ./ cs.rr0s) ≈ rr rtol=1e-4

    # Entry strengths already solve the system: one zeta evaluation, no iteration
    cs, n1 = cs_solve!(pfield, b, Gtrue; tol, precondition)
    @test n1 == 1
    @test get_Gammas(pfield) == Gtrue

    # A good initial guess is used (fewer iterations than from zero) and the
    # target stays ||r||/||b||, not the stricter ||r||/||r0||
    Gnear = Gtrue .+ 1e-3 .* randn(rng, size(Gtrue)...)
    cs, n2 = cs_solve!(pfield, b, Gnear; tol, precondition)
    @test n2 < n0
    @test all(relres(pfield, get_Gammas(pfield), b) .<= tol)
end

# The device CG (ext/FLOWVPMGPUExt.jl) on the same field, with a device backend
# loaded via the FLOWVPM_CS_DEVICE env (value = array type module: Metal or CUDA)
if haskey(ENV, "FLOWVPM_CS_DEVICE")
    @testset "Core-spreading RBF solve, device ($(ENV["FLOWVPM_CS_DEVICE"]))" begin
        mod = ENV["FLOWVPM_CS_DEVICE"]
        @eval using KernelAbstractions, $(Symbol(mod))
        AT = mod == "Metal" ? Metal.MtlArray : CUDA.CuArray
        TF = mod == "Metal" ? Float32 : Float64
        tol = TF == Float32 ? 1e-3 : 1e-6

        ph = cs_field()
        np = FLOWVPM.get_np(ph)
        Gtrue = randn(Random.MersenneTwister(2), 3, np)
        b = cs_apply(ph, Gtrue)
        Gnear = Gtrue .+ 1e-3 .* randn(Random.MersenneTwister(3), 3, np)
        pd = FLOWVPM.ParticleField(np, TF; arraytype=AT)
        pd.np = np

        function solve_device(G0)
            Ph = copy(ph.particles)
            Ph[FLOWVPM.M_INDEX[7:9], 1:np] .= b
            Ph[FLOWVPM.GAMMA_INDEX, 1:np] .= G0
            copyto!(pd.particles, TF.(Ph))
            ncalls = Ref(0)
            zeta = pf -> (ncalls[] += 1; FLOWVPM.zeta_direct(pf))
            cs = FLOWVPM.CoreSpreading(0.0, 1.3, zeta; tol, itmax=200)
            FLOWVPM.rbf_conjugategradient(pd, cs)
            G = Float64.(Array(pd.particles)[FLOWVPM.GAMMA_INDEX, 1:np])
            return G, ncalls[]
        end

        G, n0 = solve_device(zeros(3, np))
        @test all(relres(ph, G, b) .<= 2tol)          # true residual, Float64 A on the host
        G, n1 = solve_device(Gtrue)
        @test n1 == 1
        G, n2 = solve_device(Gnear)
        @test n2 < n0
    end
end
