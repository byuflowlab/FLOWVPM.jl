using Test
import Random
import FLOWVPM

# CoreSpreading through `viscousdiffusion` (euler): cores grow as sigma^2 + 2 nu dt
# and go back to sgm0 by the Galerkin reset once sigma/sgm0 reaches beta. A Gaussian
# vortex tube (core 1.5, h = 0.5 lattice, sgm0 = 1) under pure diffusion: the exact
# field after a time t is the initial particles with every core sqrt(sgm0^2 + 2 nu t).

const CS_SGM0 = 1.0
const CS_C3 = (2pi)^-1.5

function cs_tube(; h=0.5, s0v=1.5)
    X = Float64[]; G = Float64[]
    for x in -4:h:4, y in -4:h:4, z in -2:h:2
        append!(X, (x, y, z))
        append!(G, (0.0, 0.0, exp(-(x^2 + y^2) / (2s0v^2)) / (2pi * s0v^2) * h^3))
    end
    return reshape(X, 3, :), reshape(G, 3, :)
end

function cs_field(X, G; TF=Float64, pkw=(;), static=Int[])
    cs = FLOWVPM.CoreSpreading(0.5, CS_SGM0; beta=1.25)
    pf = FLOWVPM.ParticleField(4size(X, 2), TF; viscous=cs, integration=FLOWVPM.euler, pkw...)
    blk = zeros(TF, FLOWVPM.nfields, size(X, 2))
    blk[1:3, :] .= X; blk[4:6, :] .= G; blk[7, :] .= CS_SGM0; blk[FLOWVPM.CIRCULATION_INDEX, :] .= 1
    blk[FLOWVPM.STATIC_INDEX, static] .= 1
    FLOWVPM.add_particles!(pf, blk)
    return pf
end

cs_state(pf) = (n = FLOWVPM.get_np(pf); P = Float64.(Array(pf.particles)[:, 1:n]); (P[1:3, :], P[4:6, :], P[7, :], P[FLOWVPM.STATIC_INDEX, :]))

function cs_omega(X, G, S, Y)
    W = zeros(3, size(Y, 2))
    for i in axes(Y, 2), j in axes(X, 2)
        r2 = (Y[1, i] - X[1, j])^2 + (Y[2, i] - X[2, j])^2 + (Y[3, i] - X[3, j])^2
        W[:, i] .+= CS_C3 * exp(-r2 / (2S[j]^2)) / S[j]^3 .* G[:, j]
    end
    return W
end

const CS_PROBES = (rng = Random.MersenneTwister(5); [3.6, 3.6, 2.0] .* (2 .* rand(rng, 3, 400) .- 1))

# 7 steps of 2 nu dt = 0.25 sgm0^2 with beta 1.25: full resets after steps 3 and 6
function cs_run!(pf; nsteps=7, dt=0.25)
    nres = 0
    for _ in 1:nsteps
        t0 = pf.viscous.t_sgm
        FLOWVPM.viscousdiffusion(pf, dt)
        nres += pf.viscous.t_sgm < t0 + dt / 2
    end
    return nres
end

@testset "CoreSpreading: Lamb-Oseen diffusion with Galerkin resets" begin
    X0, G0 = cs_tube()
    pf = cs_field(X0, G0)
    @test cs_run!(pf) == 2
    X, G, S, _ = cs_state(pf)
    @test all(S .≈ sqrt(CS_SGM0^2 + 0.25))           # one step since the last reset
    W = cs_omega(X0, G0, fill(sqrt(CS_SGM0^2 + 7 * 0.25), size(X0, 2)), CS_PROBES)
    E = cs_omega(X, G, S, CS_PROBES) .- W
    @test sqrt(sum(abs2, E) / sum(abs2, W)) <= 2e-3
    @test abs(sum(G[3, :]) - sum(G0[3, :])) <= 1e-3 * sum(G0[3, :])
end

@testset "CoreSpreading: static particles are neither grown nor reset" begin
    X0, G0 = cs_tube()
    st = collect(1:7:size(X0, 2))
    pf = cs_field(X0, G0; static=st)
    @test cs_run!(pf) == 2
    X, G, S, St = cs_state(pf)
    ks = findall(==(1), St)
    @test length(ks) == length(st)
    @test sort(collect(eachcol(vcat(X[:, ks], G[:, ks], S[ks]')))) ==
          sort(collect(eachcol(vcat(X0[:, st], G0[:, st], fill(CS_SGM0, 1, length(st))))))
    @test all(S[St .== 0] .≈ sqrt(CS_SGM0^2 + 0.25))
end

@testset "CoreSpreading: the integrator that runs picks the growth" begin
    # euler and rungekutta3 each grow the cores by sigma^2 += 2 nu dt per step, also
    # on a field whose `integration` names the other one (a driver may step a field
    # with either; Euler on an RK3-flagged field grew nothing before)
    X0, G0 = cs_tube(; h=1.0)
    for (flag, step!) in ((FLOWVPM.rungekutta3, FLOWVPM.euler), (FLOWVPM.euler, FLOWVPM.rungekutta3))
        pf = FLOWVPM.ParticleField(size(X0, 2), Float64; viscous=FLOWVPM.CoreSpreading(0.5, CS_SGM0; beta=10.0), integration=flag, UJ=FLOWVPM.UJ_direct,
                                   formulation=FLOWVPM.formulation_classic)
        blk = zeros(FLOWVPM.nfields, size(X0, 2))
        blk[1:3, :] .= X0; blk[4:6, :] .= G0; blk[7, :] .= CS_SGM0; blk[FLOWVPM.CIRCULATION_INDEX, :] .= 1
        FLOWVPM.add_particles!(pf, blk)
        for _ in 1:2; step!(pf, 0.1); end
        @test all(cs_state(pf)[3] .≈ sqrt(CS_SGM0^2 + 2 * 2 * 0.5 * 0.1))
        @test pf.viscous.t_sgm ≈ 0.2
    end
end

# The device path (ext/FLOWVPMGPUExt.jl) against the host Float64 run, with a device
# backend loaded via the FLOWVPM_CS_DEVICE env (Metal or CUDA)
if haskey(ENV, "FLOWVPM_CS_DEVICE")
    @testset "CoreSpreading: device ($(ENV["FLOWVPM_CS_DEVICE"])) vs host" begin
        mod = ENV["FLOWVPM_CS_DEVICE"]
        @eval using KernelAbstractions, $(Symbol(mod))
        AT = mod == "Metal" ? Metal.MtlArray : CUDA.CuArray
        TF = mod == "Metal" ? Float32 : Float64
        X0, G0 = cs_tube()
        ph = cs_field(X0, G0); cs_run!(ph)
        pd = cs_field(X0, G0; TF, pkw=(; arraytype=AT))
        @test cs_run!(pd) == 2
        Wh = cs_omega(cs_state(ph)[1:3]..., CS_PROBES); Wd = cs_omega(cs_state(pd)[1:3]..., CS_PROBES)
        @test sqrt(sum(abs2, Wd .- Wh) / sum(abs2, Wh)) <= 1e-3
    end
end
