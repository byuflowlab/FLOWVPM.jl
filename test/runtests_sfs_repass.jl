# sfs_repass!: the SFS estimator recomputed from the particles' current U/J over
# the last radix evaluation's direct pairs. (1) Repass right after the pass
# reproduces the pass's estimator to roundoff. (2) With the gradient rows
# modified (a synthetic extra source field), the repass matches FLOWVPM's
# direct all-pairs estimator evaluated on the same modified rows.
using Test, Random, LinearAlgebra
import FLOWVPM
const vpm = FLOWVPM

function repass_field(n; seed=11)
    rng = MersenneTwister(seed)
    sigma = 1.5 * (1.0 / n)^(1 / 3)
    sfs = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive; alpha=0.999, maxC=1.0,
                         clippings=[vpm.clipping_backscatter])
    pf = vpm.ParticleField(n, Float64; formulation=vpm.rVPM, kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(), SFS=sfs, transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    for k in 1:n
        vpm.add_particle(pf, rand(rng, 3), (2 .* rand(rng, 3) .- 1) ./ n, sigma * (0.9 + 0.2rand(rng)))
    end
    return pf
end
relerr(A, B, rows, n) = sqrt(sum((A[r, i] - B[r, i])^2 for i in 1:n, r in rows) / sum(B[r, i]^2 for i in 1:n, r in rows))

@testset "sfs_repass!" begin
    n = 8000
    pf = repass_field(n)
    vpm.radix_fmm_settings!(pf; oversize_count=-1)
    vpm.UJ_fmm_gpu!(pf; reset=true, reset_sfs=true, sfs=true)
    E1 = copy(pf.particles[vpm.SFS_INDEX, 1:n])
    vpm.sfs_repass!(pf)
    E2 = pf.particles[vpm.SFS_INDEX, 1:n]
    e = sqrt(sum((E1 .- E2) .^ 2) / sum(E1 .^ 2))
    println("  repass vs pass: $(round(e, sigdigits=3))")
    @test e < 1e-10
    # a synthetic extra field: scale the gradient rows by a smooth factor
    P = pf.particles
    for i in 1:n
        f = 1 + 0.3 * P[1, i]
        P[vpm.J_INDEX, i] .*= f
    end
    vpm.sfs_repass!(pf)
    Er = copy(P[vpm.SFS_INDEX, 1:n])
    # reference: FLOWVPM's all-pairs estimator on the same rows
    P[vpm.SFS_INDEX, 1:n] .= 0
    vpm.Estr_direct!(pf)
    Ed = P[vpm.SFS_INDEX, 1:n]
    e2 = sqrt(sum((Er .- Ed) .^ 2) / sum(Ed .^ 2))
    println("  repass vs direct estimator on modified J: $(round(e2, sigdigits=3))")
    @test e2 < 5e-3
end
