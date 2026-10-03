# Oversize-core handling on the radix path (RadixFMMSettings.oversize_count):
# a random cube with a handful of cores several times the rest. The masked
# evaluation (K largest cores out of the tree, all-pairs onto every target)
# must match UJ_direct as closely as the unmasked one, leave the field's
# strengths and cores untouched, and let the geometry rule pick a deeper grid.
using Test, Random, LinearAlgebra
import FLOWVPM
const vpm = FLOWVPM
const fmm = FLOWVPM.fmm

function oversize_field(n, nbig; seed=7)
    rng = MersenneTwister(seed)
    sigma = 1.2 * (1.0 / n)^(1 / 3)
    pf = vpm.ParticleField(n, Float64; formulation=vpm.rVPM, kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(), SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, autotune_p=false, autotune_ncrit=false,
                    autotune_reg_error=false, default_rho_over_sigma=1.0))
    for k in 1:n
        X = rand(rng, 3); G = (2 .* rand(rng, 3) .- 1) ./ n
        vpm.add_particle(pf, X, G, k <= nbig ? 3sigma : sigma * (0.9 + 0.2rand(rng)))
    end
    return pf
end

function rel_err(P, Q, np, rows)
    e = 0.0; r = 0.0
    for i in 1:np, k in rows
        e += (P[k, i] - Q[k, i])^2; r += Q[k, i]^2
    end
    return sqrt(e / r)
end

@testset "oversize cores on the radix path" begin
    n, nbig = 20000, 6
    ref = oversize_field(n, nbig)
    vpm.UJ_direct(ref)
    R = copy(ref.particles)
    for K in (-1, 8)
        pf = oversize_field(n, nbig)
        vpm.radix_fmm_settings!(pf; oversize_count=K)
        before = copy(pf.particles)
        vpm.UJ_fmm_gpu!(pf; reset=true)
        P = pf.particles
        # strengths and cores restored exactly
        @test P[vpm.GAMMA_INDEX, 1:n] == before[vpm.GAMMA_INDEX, 1:n]
        @test P[vpm.SIGMA_INDEX, 1:n] == before[vpm.SIGMA_INDEX, 1:n]
        eu = rel_err(P, R, n, vpm.U_INDEX); ej = rel_err(P, R, n, vpm.J_INDEX)
        st = vpm._radix_fmm_couplings[pf]
        println("  K = $K: ell $(st.cache.ell), U rel err $(round(eu, sigdigits=3)), J rel err $(round(ej, sigdigits=3)), sigma_limit $(round(st.sigma_limit, sigdigits=3))")
        @test eu < 5e-3
        @test ej < 2e-2
        K == -1 && (global ell0 = st.cache.ell)
        K == 8 && @test st.cache.ell >= ell0
        # the masked evaluation saw the (K+1)-th largest core: the geometry limit
        # is below the big cores
        K == 8 && @test st.sigma_limit < 3 * 1.2 * (1.0 / n)^(1 / 3)
    end
end


@testset "adaptive oversize threshold" begin
    # a 1% tail of 3x cores: the adaptive rule must mask the tail (and only
    # about the tail) and build the grid the tail-free field gets
    n = 20000; nbig = 200
    ref = oversize_field(n, nbig); vpm.UJ_direct(ref); R = copy(ref.particles)
    clean = oversize_field(n, 0); vpm.radix_fmm_settings!(clean; oversize_count=-1)
    vpm.UJ_fmm_gpu!(clean; reset=true); ell_clean = vpm._radix_fmm_couplings[clean].cache.ell
    pf = oversize_field(n, nbig)
    vpm.radix_fmm_settings!(pf; oversize_count=0, oversize_fraction=0.02)
    before = copy(pf.particles)
    vpm.UJ_fmm_gpu!(pf; reset=true)
    P = pf.particles
    @test P[vpm.GAMMA_INDEX, 1:n] == before[vpm.GAMMA_INDEX, 1:n]
    @test P[vpm.SIGMA_INDEX, 1:n] == before[vpm.SIGMA_INDEX, 1:n]
    st = vpm._radix_fmm_couplings[pf]
    thr = vpm._radix_oversize_thr[pf].thr
    masked = count(>(thr), P[vpm.SIGMA_INDEX, 1:n])
    eu = rel_err(P, R, n, vpm.U_INDEX); ej = rel_err(P, R, n, vpm.J_INDEX)
    println("  adaptive: thr $(round(thr, sigdigits=3)) masks $masked of $n (tail $nbig), ell $(st.cache.ell) vs tail-free $ell_clean, U $(round(eu, sigdigits=3)) J $(round(ej, sigdigits=3))")
    @test nbig <= masked <= round(Int, 0.02n)
    @test st.cache.ell == ell_clean
    @test eu < 5e-3
    @test ej < 2e-2
end

@testset "adaptive threshold stays within the cached grid" begin
    # A continuous core tail (1.1-3x), as a convecting wake has. The adaptive
    # threshold is derived for a fresh padded box; once the field's extent grows
    # (still inside the cache's padded box, so no recenter), that box is larger
    # than the cache's and the threshold let cores through that the cached grid
    # cannot serve: the coupling rebuilt at the same (ell, q) on every refresh
    # (NREL 5MW, 2026-10-02). The threshold is now capped at the cache's limit where
    # it is used, so the coupling survives and the evaluation stays accurate.
    n = 20000; ntail = 400
    sigma = 1.2 * (1.0 / n)^(1 / 3)
    function mk()
        f = oversize_field(n, 0; seed=11)
        for k in 1:ntail; f.particles[vpm.SIGMA_INDEX, k] = sigma * (1.1 + 1.9 * (k - 1) / (ntail - 1)); end
        return f
    end
    pf = mk()
    vpm.radix_fmm_settings!(pf; oversize_count=0, oversize_fraction=0.02)
    vpm.UJ_fmm_gpu!(pf; reset=true)
    st1 = vpm._radix_fmm_couplings[pf]
    # grow the extent 5 % along x (padding is 10 % per face: still inside the box)
    i = argmax(view(pf.particles, vpm.X_INDEX[1], 1:n))
    pf.particles[vpm.X_INDEX[1], i] += 0.05
    ref = mk(); ref.particles[vpm.X_INDEX[1], i] += 0.05
    vpm.UJ_direct(ref); R = copy(ref.particles)
    delete!(vpm._radix_oversize_thr, pf)          # the refresh that saw the grown box
    vpm.UJ_fmm_gpu!(pf; reset=true)
    st2 = vpm._radix_fmm_couplings[pf]
    thr = vpm._radix_oversize_thr[pf].thr
    lim = vpm.fmm.radix_sigma_limit(vpm._radix_geometry_policy(st2.settings), st2.cache)
    eu = rel_err(pf.particles, R, n, vpm.U_INDEX); ej = rel_err(pf.particles, R, n, vpm.J_INDEX)
    println("  grown box: same coupling $(st2 === st1), thr $(round(thr, sigdigits=4)) vs cache limit $(round(lim, sigdigits=4)), U $(round(eu, sigdigits=3)) J $(round(ej, sigdigits=3))")
    @test st2 === st1
    @test eu < 5e-3
    @test ej < 2e-2
end
