using Test
import Random
import FLOWVPM

# Galerkin core-size reset (`galerkin_shrink!`, the reset of `CoreSpreading`) on a
# Lamb-Oseen tube: a Gaussian vortex (core 1.5) sampled on an h = 0.5 lattice in
# a finite box, sgm0 = 1. A reset only changes the representation, so the
# reference is the field before it (cores grown = the exactly diffused field).
# Thresholds from the 2026-10-14 study (measured values in brackets there).

const GAL_SGM0 = 1.0
const GAL_C3 = (2pi)^-1.5

function gal_tube(c, sigf; h=0.5, s0v=1.5)
    X = Float64[]; G = Float64[]; S = Float64[]
    for x in -4:h:4, y in -4:h:4, z in -2:h:2
        append!(X, (x + c[1], y + c[2], z))
        append!(G, (0.0, 0.0, exp(-(x^2 + y^2) / (2s0v^2)) / (2pi * s0v^2) * h^3))
        push!(S, sigf(x^2 + y^2))
    end
    return reshape(X, 3, :), reshape(G, 3, :), S
end

function gal_field(X, G, S; maxparticles=size(X, 2), beta=1.5, gal...)
    cs = FLOWVPM.CoreSpreading(0.0, GAL_SGM0; beta, galerkin=FLOWVPM.GalerkinReset(; gal...))
    pf = FLOWVPM.ParticleField(maxparticles, Float64; viscous=cs)
    for j in axes(X, 2)
        FLOWVPM.add_particle(pf, X[:, j], G[:, j], S[j])
    end
    return pf, cs
end

gal_state(pf) = (n = FLOWVPM.get_np(pf); P = pf.particles;
                 (Float64.(P[1:3, 1:n]), Float64.(P[4:6, 1:n]), Float64.(P[7, 1:n])))

# Gaussian vorticity of the particles at the points Y
function gal_omega(X, G, S, Y)
    W = zeros(3, size(Y, 2))
    for i in axes(Y, 2), j in axes(X, 2)
        r2 = (Y[1, i] - X[1, j])^2 + (Y[2, i] - X[2, j])^2 + (Y[3, i] - X[3, j])^2
        w = GAL_C3 * exp(-r2 / (2S[j]^2)) / S[j]^3
        W[:, i] .+= w .* G[:, j]
    end
    return W
end

gal_probes(c) = (rng = Random.MersenneTwister(5);
                 hcat([[2rand(rng) - 1, 2rand(rng) - 1, rand(rng) - 0.5] .* [1.8, 1.8, 2.0] .+ [c[1], c[2], 0.0] for _ in 1:400]...))

# probe L2 error (%), max |Gamma| ratio, circulation change (% of the reset part's)
function gal_metrics(X0, G0, S0, Sref, X, G, S, Yp, sub)
    W = gal_omega(X0, G0, Sref, Yp)
    E = gal_omega(X, G, S, Yp) .- W
    gn(A) = maximum(sqrt.(sum(abs2, A; dims=1)))
    circ(A) = vec(sum(A; dims=2))
    return (probe = 100 * sqrt(sum(abs2, E) / sum(abs2, W)),
            gratio = gn(G) / gn(G0),
            dcirc = 100 * maximum(abs.(circ(G) .- circ(G0))) / maximum(abs.(circ(G0[:, sub]))))
end

const GAL_UNI = r2 -> 1.4
const GAL_VAR = r2 -> GAL_SGM0 * (1 + 0.7 * exp(-r2 / (2 * 1.5^2)))
const GAL_CASES = [("A centred", (0.0, 0.0), GAL_UNI, (x, y, s) -> x^2 + y^2 < 4),
                   ("B off-centre", (2.5, 1.0), GAL_UNI, (x, y, s) -> x^2 + y^2 < 4),
                   ("C half-cylinder", (2.5, 1.0), GAL_UNI, (x, y, s) -> x^2 + y^2 < 6.25 && x > 0),
                   ("D variable, beta 1.25", (2.5, 1.0), GAL_VAR, (x, y, s) -> s > 1.25),
                   ("E variable, beta 1.5", (2.5, 1.0), GAL_VAR, (x, y, s) -> s > 1.5)]

@testset "Galerkin reset, precision conversion, iterror, preconditioner blocks" begin
    cs = FLOWVPM.CoreSpreading(0.0, 1.0; galerkin=FLOWVPM.GalerkinReset(; thin=0.7))
    pf = FLOWVPM.ParticleField(10, Float32; viscous=cs)        # the field converts its scheme
    @test pf.viscous.galerkin isa FLOWVPM.GalerkinReset{Float32}
    @test pf.viscous.galerkin.thin == 0.7f0
    @test FLOWVPM.CoreSpreading(0.0, 1.0).galerkin isa FLOWVPM.GalerkinReset

    # not converged at itmax with the field error above tol: warn, or throw with iterror
    X0, G0, S0 = gal_tube((2.5, 1.0), GAL_UNI)
    sub = findall(j -> (X0[1, j] - 2.5)^2 + (X0[2, j] - 1.0)^2 < 4, axes(X0, 2))
    pf, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2), itmax=2, block_cap=0)
    @test_logs (:warn, r"not converged") FLOWVPM.corespreading_reset_subset!(pf, cs, sub)
    pf, _ = gal_field(X0, G0, S0; maxparticles=2size(X0, 2), itmax=2, block_cap=0)
    @test_throws ErrorException FLOWVPM.galerkin_shrink!(pf, sub, GAL_SGM0; iterror=true,
                                                          galerkin=FLOWVPM.GalerkinReset(; itmax=2, block_cap=0))

    # each stored block is the inverse of its block of M + D
    U, _ = FLOWVPM._gal_set(X0, S0 .^ 2, 2.0)
    d = 1e-2 * GAL_C3 .* (2 .* U.v) .^ -1.5
    blk = FLOWVPM._gal_blocks(U, d, 25.0, 3.0, 64, Float64)
    @test sort(Int.(blk.perm)) == collect(axes(X0, 2))
    b = argmax(diff(blk.start)); s = Int(blk.start[b]); m = Int(blk.start[b + 1]) - s
    ids = Int.(blk.perm[s:s + m - 1])
    A = [(vv = U.v[a] + U.v[c]; r2 = sum(abs2, U.x[:, a] .- U.x[:, c]);
          (r2 <= 25vv) * GAL_C3 * exp(-r2 / 2vv) / vv^1.5 + (a == c) * d[a]) for a in ids, c in ids]
    Minv = reshape(blk.minv[blk.moff[b]:blk.moff[b + 1] - 1], m, m)
    @test maximum(abs.(Minv * A .- [a == c ? 1.0 : 0.0 for a in 1:m, c in 1:m])) < 1e-8
end

# a spread of 1000 thinning distances per axis would need 1e9 cells: the set's cells
# are enlarged (2^26 cap) and the thinning must index them by the set's cell size
@testset "Galerkin thinning over a very large spread" begin
    q = 0.5
    X = [0.0 0.1 300.0 300.1 500.0 500.0;
         0.0 0.0 300.0 300.0 500.0 500.0;
         0.0 0.0 300.0 300.0 0.0 0.3]
    @test FLOWVPM._gal_set(X, zeros(6), q)[1].h > q
    @test FLOWVPM._gal_thin(X, [2.0, 1.0, 1.0, 2.0, 1.0, 1.0], q) == [true, false, false, true, true, false]
end

@testset "Galerkin reset, Lamb-Oseen single subset reset: $name" for (name, c, sigf, insub) in GAL_CASES
    X0, G0, S0 = gal_tube(c, sigf)
    sub = [j for j in axes(X0, 2) if insub(X0[1, j] - c[1], X0[2, j] - c[2], S0[j])]
    pf, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2))
    FLOWVPM.corespreading_reset_subset!(pf, cs, sub)
    X, G, S = gal_state(pf)
    m = gal_metrics(X0, G0, S0, S0, X, G, S, gal_probes(c), sub)
    @test m.probe <= 0.15
    @test m.gratio <= 2.5
    @test m.dcirc <= 1.0
    @test count(S .> GAL_SGM0 * (1 + 1e-6)) == count(S0 .> GAL_SGM0 * (1 + 1e-6)) - length(sub)
end

@testset "Galerkin reset, Lamb-Oseen full reset (F)" begin
    c = (2.5, 1.0)
    X0, G0, S0 = gal_tube(c, GAL_UNI)
    pf, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2))
    FLOWVPM._corespreading_reset!(pf, cs)
    X, G, S = gal_state(pf)
    m = gal_metrics(X0, G0, S0, S0, X, G, S, gal_probes(c), collect(axes(X0, 2)))
    @test m.probe <= 0.15
    @test m.gratio <= 2.5
    @test all(S .== GAL_SGM0)
    @test cs.t_sgm == 0
end

@testset "Galerkin reset, full reset leaves static particles alone" begin
    c = (2.5, 1.0)
    X0, G0, S0 = gal_tube(c, GAL_UNI)
    pf, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2))
    st = 1:7:size(X0, 2)
    pf.particles[FLOWVPM.STATIC_INDEX, st] .= 1
    FLOWVPM._corespreading_reset!(pf, cs)
    X, G, S = gal_state(pf)
    new = Set(eachcol(vcat(X, G, S')))
    @test all(j -> vcat(X0[:, j], G0[:, j], S0[j]) in new, st)     # untouched (cores still 1.4)
    @test count(S .> GAL_SGM0 * (1 + 1e-6)) == length(st)
end

@testset "Galerkin shrink on an Inviscid field (no viscous scheme)" begin
    c = (2.5, 1.0)
    X0, G0, S0 = gal_tube(c, GAL_VAR)
    pf = FLOWVPM.ParticleField(2size(X0, 2), Float64; viscous=FLOWVPM.Inviscid())
    for j in axes(X0, 2); FLOWVPM.add_particle(pf, X0[:, j], G0[:, j], S0[j]); end
    sub = findall(>(1.5), S0)
    gr = FLOWVPM.GalerkinReset()
    st = FLOWVPM.galerkin_shrink!(pf, sub, GAL_SGM0; galerkin=gr)
    @test st.K == length(sub) && st.its > 0
    X, G, S = gal_state(pf)
    m = gal_metrics(X0, G0, S0, S0, X, G, S, gal_probes(c), sub)
    @test m.probe <= 0.15
    @test all(S .<= 1.5)
    @test (@inferred FLOWVPM._galerkin_shrink!(pf, Int[1], GAL_SGM0, gr, false, 0)) isa NamedTuple
end

# Repeated resets against exact diffusion (sigma^2 + t on the initial particles):
# uniform init, every reset a full one; variable init, partial resets of the cores
# above beta sgm0 as they cross
@testset "Galerkin reset, Lamb-Oseen multi-reset, $init" for (init, beta, dt2, tol) in
        (("uniform", 1.25, 0.5625 + 1e-9, 0.1), ("variable", 1.5, 0.06, 0.3))
    c = (2.5, 1.0)
    X0, G0, S0 = gal_tube(c, init == "uniform" ? (r2 -> GAL_SGM0) : (r2 -> GAL_SGM0 * (1 + 0.3 * exp(-r2 / (2 * 1.5^2)))))
    pf, cs = gal_field(X0, G0, S0; maxparticles=12size(X0, 2), beta)
    t = 0.0; nres = 0
    while t < 3.84 - 1e-9
        dt = min(dt2, 3.84 - t); t += dt
        P = pf.particles; n = FLOWVPM.get_np(pf)
        P[FLOWVPM.SIGMA_INDEX, 1:n] .= sqrt.(P[FLOWVPM.SIGMA_INDEX, 1:n] .^ 2 .+ dt)
        sub = findall(>(beta * GAL_SGM0), P[FLOWVPM.SIGMA_INDEX, 1:n])
        if !isempty(sub)
            FLOWVPM.corespreading_reset_subset!(pf, cs, sub); nres += 1
        end
    end
    X, G, S = gal_state(pf)
    m = gal_metrics(X0, G0, S0, sqrt.(S0 .^ 2 .+ 3.84), X, G, S, gal_probes(c), collect(axes(X0, 2)))
    @test nres >= 6
    @test m.probe <= tol
    @test m.gratio <= 2.5
end

@testset "Galerkin reset, stored and on-the-fly Gram agree: $name" for (name, c, sigf, insub) in GAL_CASES[[2, 5]]
    X0, G0, S0 = gal_tube(c, sigf)
    sub = [j for j in axes(X0, 2) if insub(X0[1, j] - c[1], X0[2, j] - c[2], S0[j])]
    out = map((typemax(Int), 0)) do mem          # every pair stored / none
        pf, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2), gram_mem=mem)
        FLOWVPM._galerkin_reset!(pf, cs, sub)
        gal_state(pf)
    end
    # stored values are Float32: the CG paths part in the Gram matrix's near-null
    # directions (strengths; the edge-point prune may differ by a point), not in
    # the field (difference well below the reset's own ~4e-4)
    @test abs(size(out[1][1], 2) - size(out[2][1], 2)) <= 2
    Yp = gal_probes(c); W1 = gal_omega(out[1]..., Yp); W2 = gal_omega(out[2]..., Yp)
    @test sqrt(sum(abs2, W1 .- W2) / sum(abs2, W2)) <= 5e-4
end

# The device reset (ext/FLOWVPMGPUExt.jl) against the host Float64 one, with a
# device backend loaded via the FLOWVPM_CS_DEVICE env (Metal or CUDA)
if haskey(ENV, "FLOWVPM_CS_DEVICE")
    @testset "Galerkin reset, device ($(ENV["FLOWVPM_CS_DEVICE"])) vs host" begin
        mod = ENV["FLOWVPM_CS_DEVICE"]
        @eval using KernelAbstractions, $(Symbol(mod))
        AT = mod == "Metal" ? Metal.MtlArray : CUDA.CuArray
        TF = mod == "Metal" ? Float32 : Float64
        for (name, c, sigf, insub) in GAL_CASES[[2, 5]]
            X0, G0, S0 = gal_tube(c, sigf)
            sub = [j for j in axes(X0, 2) if insub(X0[1, j] - c[1], X0[2, j] - c[2], S0[j])]
            ph, cs = gal_field(X0, G0, S0; maxparticles=2size(X0, 2))
            FLOWVPM.corespreading_reset_subset!(ph, cs, sub)
            pd = FLOWVPM.ParticleField(2size(X0, 2), TF; arraytype=AT, viscous=cs)
            blk = zeros(TF, FLOWVPM.nfields, size(X0, 2))
            blk[1:3, :] .= X0; blk[4:6, :] .= G0; blk[7, :] .= S0; blk[FLOWVPM.CIRCULATION_INDEX, :] .= 1
            FLOWVPM.add_particles!(pd, blk)
            FLOWVPM.corespreading_reset_subset!(pd, cs, sub)
            @test abs(FLOWVPM.get_np(pd) - FLOWVPM.get_np(ph)) <= 2   # edge prune may differ by a point
            Xh, Gh, Sh = gal_state(ph)
            Pd = Array(pd.particles); n = FLOWVPM.get_np(pd)
            Xd = Float64.(Pd[1:3, 1:n]); Gd = Float64.(Pd[4:6, 1:n]); Sd = Float64.(Pd[7, 1:n])
            Yp = gal_probes(c)
            mh = gal_metrics(X0, G0, S0, S0, Xh, Gh, Sh, Yp, sub)
            md = gal_metrics(X0, G0, S0, S0, Xd, Gd, Sd, Yp, sub)
            Wh = gal_omega(Xh, Gh, Sh, Yp); Wd = gal_omega(Xd, Gd, Sd, Yp)
            @test abs(md.probe - mh.probe) <= 0.02
            @test sqrt(sum(abs2, Wd .- Wh) / sum(abs2, Wh)) <= 5e-4  # reset error itself ~4-6e-4
            # (strengths are not compared: Float32 and Float64 PCG part in the Gram
            # matrix's near-null directions and stop at different iterations)
        end
    end
end

# Opt-in: the 5MW step-2880 subset (beta 1.5) from a 7-row Float32 dump
# (X, Gamma, sigma; see the 2026-10-15 notes) named by FLOWVPM_CS_5MW
if haskey(ENV, "FLOWVPM_CS_5MW")
    include(joinpath(@__DIR__, "galerkin_5mw.jl"))
end
