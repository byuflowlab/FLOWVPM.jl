# Opt-in (included from runtests_corespreading_galerkin.jl when FLOWVPM_CS_5MW is set):
# Galerkin subset reset (beta 1.5) of the NREL 5MW step-2880 wake, 1.88M particles,
# read from a 7-row Float32 dump (header Int64 nrows, ncols, 32; then X, Gamma, sigma).
# On a device with FLOWVPM_CS_DEVICE, else on the host (FLOWVPM_CS_GRAM_MEM bytes for
# the stored Gram; the beta-1.5 subset needs ~9 GB, below it the Gram is on the fly).
import Statistics
@testset "Galerkin reset, 5MW step 2880 subset, beta 1.5" begin
    Q = open(ENV["FLOWVPM_CS_5MW"]) do io
        nr = read(io, Int64); nc = read(io, Int64); read(io, Int64)
        A = Matrix{Float32}(undef, nr, nc); read!(io, A); A
    end
    N = size(Q, 2); sg0 = Statistics.median(Float64.(Q[7, :]))
    idx = findall(>(1.5sg0), Float64.(Q[7, :]))
    dev = get(ENV, "FLOWVPM_CS_DEVICE", "")
    TF = dev == "" ? Float64 : Float32
    pkw = dev == "" ? (;) : (; arraytype=(dev == "Metal" ? Metal.MtlArray : CUDA.CuArray))
    gmem = parse(Int, get(ENV, "FLOWVPM_CS_GRAM_MEM", string(Int(Sys.total_memory() ÷ 4))))
    cs = FLOWVPM.CoreSpreading(0.0, sg0, FLOWVPM.zeta_fmm; reset=:galerkin, galerkin=FLOWVPM.GalerkinReset(; gram_mem=gmem))
    pf = FLOWVPM.ParticleField(ceil(Int, 1.02N), TF; viscous=cs, pkw...)
    blk = zeros(TF, FLOWVPM.nfields, N); blk[1:7, :] .= Q; blk[FLOWVPM.CIRCULATION_INDEX, :] .= 1
    FLOWVPM.add_particles!(pf, blk); blk = nothing
    FLOWVPM.corespreading_reset_subset!(pf, cs, idx)
    n = FLOWVPM.get_np(pf)
    A0 = Float64.(Q); A1 = Float64.(FLOWVPM._radix_oversize_gather(pf.particles, collect(1:n), 1:7))

    # field at probes near the subset (within ~sgm0) and on a shell 1-3 sgm0 out (cut 6 sigma)
    function field(A, Y)
        v = A[7, :] .^ 2
        S, perm = FLOWVPM._gal_set(A[1:3, :], v, 6 * sqrt(maximum(v)) / 2)
        FLOWVPM._gal_pairsum!(zeros(3, size(Y, 2)), Y, zeros(size(Y, 2)), S, A[4:6, perm], 36.0)
    end
    Xs = A0[1:3, idx]; K = length(idx)
    rng = Random.MersenneTwister(1)
    Yp = hcat([Xs[:, rand(rng, 1:K)] .+ sg0 .* randn(rng, 3) ./ sqrt(3) for _ in 1:4000]...)
    rng = Random.MersenneTwister(2)
    Yo = hcat([(u = randn(rng, 3); Xs[:, rand(rng, 1:K)] .+ (sg0 * (1 + 2rand(rng))) .* u ./ sqrt(sum(abs2, u))) for _ in 1:4000]...)
    # error over the reset subset's own old field at the probes (the 2026-10-14 study's norm)
    rel(A, B, Y) = sqrt(sum(abs2, field(B, Y) .- field(A, Y)) / sum(abs2, field(A[:, idx], Y)))
    @test rel(A0, A1, Yp) <= 5e-3
    @test rel(A0, A1, Yo) <= 5e-3

    # strengths: every changed particle against the old maximum of the same region
    gn(A) = sqrt.(vec(sum(abs2, A; dims=1)))
    old = Dict((A0[1, j], A0[2, j], A0[3, j]) => j for j in axes(A0, 2))
    ko = [get(old, (A1[1, j], A1[2, j], A1[3, j]), 0) for j in axes(A1, 2)]
    ch = findall(j -> ko[j] == 0 || A1[4:6, j] != A0[4:6, ko[j]], 1:size(A1, 2))
    @test maximum(gn(A1[4:6, ch])) <= 1.5 * maximum(gn(A0[4:6, vcat(idx, filter(>(0), ko[ch]))]))

    # circulation and impulse (about the subset centroid) against the subset's own
    xc = vec(sum(Xs; dims=2)) ./ K
    mom(A, js) = (m = zeros(6); for j in js
        x = A[1:3, j] .- xc; g = A[4:6, j]
        m .+= (g[1], g[2], g[3], x[2]*g[3] - x[3]*g[2], x[3]*g[1] - x[1]*g[3], x[1]*g[2] - x[2]*g[1])
    end; m)
    ms = mom(A0, idx); dm = mom(A1, axes(A1, 2)) .- mom(A0, axes(A0, 2))
    @test sqrt(sum(abs2, dm[1:3]) / sum(abs2, ms[1:3])) <= 1e-2
    @test sqrt(sum(abs2, dm[4:6]) / sum(abs2, ms[4:6])) <= 2e-2    # measured 1.6 % (Metal F32)
    @test n - N <= 0.01N
end
