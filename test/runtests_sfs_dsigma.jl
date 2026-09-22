# Analytic core-scaling derivatives of the dynamic SFS procedure (two-level form).
# (1) The all-pairs reference `dsigma_direct!` against a central finite
#     difference of UJ_direct + Estr_direct! over a uniform core scaling.
# (2) The radix host sweeps (fmm!(...; sfs_dsigma=true)) against the reference.
# (3) The two-level procedure's coefficient against the pseudo-three-level
#     one at alpha = 0.999 (which converges to it as alpha -> 1).
# (4) With a device backend loaded (Metal/CUDA via the AL_DSIGMA_DEVICE env,
#     value = array type module), the device sweeps against the reference.
using Test, Random, LinearAlgebra
import FLOWVPM
const vpm = FLOWVPM

function dsigma_field(n; seed=11, TF=Float64, procedure=vpm.pseudo3level_positive,
        UJ=vpm.UJ_direct, arraytype=Array)
    rng = MersenneTwister(seed)
    sigma = 1.5 * (1.0 / n)^(1 / 3)
    sfs = vpm.DynamicSFS(vpm.Estr_fmm, procedure; alpha=0.999, maxC=1.0,
                         clippings=[vpm.clipping_backscatter])
    pf = vpm.ParticleField(n, TF; formulation=vpm.rVPM, kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(), SFS=sfs, transposed=true, integration=vpm.rungekutta3,
        UJ=UJ, arraytype,
        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    for k in 1:n
        vpm.add_particle(pf, rand(rng, 3), (2 .* rand(rng, 3) .- 1) ./ n, sigma * (0.9 + 0.2rand(rng)))
    end
    return pf
end
relerr(A, B) = sqrt(sum((A .- B) .^ 2) / sum(B .^ 2))
stretch(P, n) = [sum(P[vpm.J_INDEX[(i-1)*3+j], k] * P[vpm.GAMMA_INDEX[j], k] for j in 1:3) for i in 1:3, k in 1:n]  # transposed op: J[1:3]⋅Γ

@testset "dsigma reference vs finite difference" begin
    n = 250
    pf = dsigma_field(n)
    P = pf.particles
    h = 1e-4
    function TE(alpha)
        P[vpm.SIGMA_INDEX, 1:n] .*= alpha
        vpm.UJ_direct(pf; reset=true, reset_sfs=true, sfs=true)
        P[vpm.SIGMA_INDEX, 1:n] ./= alpha
        return stretch(P, n), copy(P[vpm.SFS_INDEX, 1:n])
    end
    Tp, Ep = TE(1 + h); Tm, Em = TE(1 - h)
    dT_fd = (Tp .- Tm) ./ (2h); dE_fd = (Ep .- Em) ./ (2h)
    vpm.UJ_direct(pf; reset=true, reset_sfs=true, sfs=true)
    vpm.dsigma_direct!(pf)
    dT = P[vpm.M_INDEX[1:3], 1:n]; dE = P[vpm.M_INDEX[4:6], 1:n]
    eT = relerr(dT, dT_fd); eE = relerr(dE, dE_fd)
    println("  L (stretching derivative) vs FD: $(round(eT, sigdigits=3));  dE vs FD: $(round(eE, sigdigits=3))")
    @test eT < 1e-6
    @test eE < 1e-6
end

@testset "radix host sweeps vs reference" begin
    n = 6000
    pf = dsigma_field(n; UJ=vpm.UJ_fmm)
    vpm.radix_fmm_settings!(pf; oversize_count=-1)
    vpm._sfs_dsigma_request!(pf, true)
    vpm.UJ_fmm_gpu!(pf; reset=true, reset_sfs=true, sfs=true)
    vpm._sfs_dsigma_request!(pf, false)
    @test vpm._sfs_dsigma_delivered(pf)
    P = pf.particles
    Ms = copy(P[vpm.M_INDEX[1:6], 1:n])
    vpm.dsigma_direct!(pf)
    Md = P[vpm.M_INDEX[1:6], 1:n]
    eT = relerr(Ms[1:3, :], Md[1:3, :]); eE = relerr(Ms[4:6, :], Md[4:6, :])
    println("  radix host L vs reference: $(round(eT, sigdigits=3));  dE: $(round(eE, sigdigits=3))")
    @test eT < 1e-6
    @test eE < 1e-6
    # the repass delivers the same channel from the same pairs
    vpm._sfs_dsigma_request!(pf, true)
    vpm.sfs_repass!(pf)
    vpm._sfs_dsigma_request!(pf, false)
    Mr = P[vpm.M_INDEX[1:6], 1:n]
    println("  repass vs pass: $(round(relerr(Mr, Ms), sigdigits=3))")
    @test relerr(Mr, Ms) < 1e-10
end

@testset "octree CPU list (dsigma_fmm!) vs reference" begin
    n = 4000
    pf = dsigma_field(n; UJ=vpm.UJ_fmm)
    vpm._sfs_dsigma_request!(pf, true)
    vpm.UJ_fmm(pf; reset=true, reset_sfs=true, sfs=true)
    vpm._sfs_dsigma_request!(pf, false)
    @test vpm._sfs_dsigma_delivered(pf)
    P = pf.particles
    Mo = copy(P[vpm.M_INDEX[1:6], 1:n])
    vpm.dsigma_direct!(pf)
    Md = P[vpm.M_INDEX[1:6], 1:n]
    eT = relerr(Mo[1:3, :], Md[1:3, :]); eE = relerr(Mo[4:6, :], Md[4:6, :])
    println("  octree list L vs reference: $(round(eT, sigdigits=3));  dE: $(round(eE, sigdigits=3))")
    @test eT < 1e-5
    @test eE < 1e-5
end

@testset "two-level coefficient vs pseudo-three-level (alpha 0.999)" begin
    n = 250
    C = map((vpm.pseudo3level_positive, vpm.twolevel_positive)) do proc
        pf = dsigma_field(n; procedure=proc)
        vpm.nextstep(pf, 1e-3)
        copy(pf.particles[vpm.C_INDEX, 1:n])
    end
    # the pseudo3level numerator/denominator carry the un-normalised
    # σ(α − 1)(3α − 2) finite-difference scaling, which cancels in C only.
    # Per particle the one-sided difference is within 0.3% of the analytic
    # derivative; the coefficient differs by 0.5% at the median and by up to
    # ~10% where the finite-difference denominator nearly cancels (the
    # conditioning the analytic form removes), so the gate is the median.
    d = abs.(C[2][1, :] .- C[1][1, :]) ./ max.(abs.(C[1][1, :]), 1e-300)
    e = sort(d)[cld(n, 2)]
    println("  C two-level vs pseudo3level(0.999): median rel diff $(round(e, sigdigits=3)), max $(round(maximum(d), sigdigits=3))")
    @test e < 1e-2
    @test all(isfinite, C[2])
end

if haskey(ENV, "AL_DSIGMA_DEVICE")
    @testset "device sweeps vs reference ($(ENV["AL_DSIGMA_DEVICE"]))" begin
        mod = ENV["AL_DSIGMA_DEVICE"]
        @eval using KernelAbstractions, $(Symbol(mod))
        AT = mod == "Metal" ? Metal.MtlArray : CUDA.CuArray
        TF = mod == "Metal" ? Float32 : Float64
        n = 6000
        pf = dsigma_field(n; UJ=vpm.UJ_fmm)
        pd = dsigma_field(n; UJ=vpm.UJ_fmm, TF, arraytype=AT)
        vpm.radix_fmm_settings!(pd; oversize_count=-1)
        vpm._sfs_dsigma_request!(pd, true)
        vpm.UJ_fmm_gpu!(pd; reset=true, reset_sfs=true, sfs=true)
        vpm._sfs_dsigma_request!(pd, false)
        Md = Array(pd.particles[vpm.M_INDEX[1:6], 1:n])
        # the reference on the device pass's own J (dE carries J; the FMM's J
        # differs from the direct sum by its expansion error, ~5e-5)
        pf.particles[vpm.J_INDEX, 1:n] .= Array(pd.particles[vpm.J_INDEX, 1:n])
        vpm.dsigma_direct!(pf)
        Mh = pf.particles[vpm.M_INDEX[1:6], 1:n]
        eT = relerr(Md[1:3, :], Mh[1:3, :]); eE = relerr(Md[4:6, :], Mh[4:6, :])
        println("  device L vs reference: $(round(eT, sigdigits=3));  dE: $(round(eE, sigdigits=3))")
        tol = TF == Float32 ? 5e-3 : 1e-6
        @test eT < tol
        @test eE < tol
    end
end
