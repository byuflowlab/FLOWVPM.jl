# RectangularGaussianErfVortex through FastMultipole.direct_rectangular!: the
# rectangular sum against an all-pairs reference built from the same g/dgdr, and
# against an independent erf (SpecialFunctions) with the cancellation-aware gate.

using Test
import FLOWVPM
import FLOWVPM.FastMultipole: direct_rectangular!
import Random

const RectangularGaussianErfVortex = FLOWVPM.RectangularGaussianErfVortex

function _point_reference(g_dgdr::F, tgt, src) where F
    n_tgt = size(tgt, 2)
    n_src = size(src, 2)
    ref = zeros(12, n_tgt)
    for i in 1:n_tgt, q in 1:n_src
        dx = tgt[1, i] - src[1, q]; dy = tgt[2, i] - src[2, q]; dz = tgt[3, i] - src[3, q]
        r2 = dx^2 + dy^2 + dz^2
        iszero(r2) && continue
        r = sqrt(r2)
        sigma = src[7, q]
        g, dgdr = g_dgdr(r / sigma)
        r3inv = 1 / (r2 * r)
        c4 = 1 / (4pi)
        gx, gy, gz = src[4, q], src[5, q], src[6, q]
        crss1 = -c4 * r3inv * (dy*gz - dz*gy)
        crss2 = -c4 * r3inv * (dz*gx - dx*gz)
        crss3 = -c4 * r3inv * (dx*gy - dy*gx)
        ref[1, i] += g * crss1; ref[2, i] += g * crss2; ref[3, i] += g * crss3
        aux1 = dgdr / (sigma*r) - 3g / r2
        aux2 = -c4 * g * r3inv
        ref[4, i] += aux1*crss1*dx
        ref[5, i] += aux1*crss2*dx - aux2*gz
        ref[6, i] += aux1*crss3*dx + aux2*gy
        ref[7, i] += aux1*crss1*dy + aux2*gz
        ref[8, i] += aux1*crss2*dy
        ref[9, i] += aux1*crss3*dy - aux2*gx
        ref[10, i] += aux1*crss1*dz - aux2*gy
        ref[11, i] += aux1*crss2*dz + aux2*gx
        ref[12, i] += aux1*crss3*dz
    end
    return ref
end

# g/dgdr from an erf function (the g_dgdr_gauserf form)
function _g_dgdr_from_erf(erf_fn::F) where F
    sqrt2opi = sqrt(2 / pi)
    return rho -> begin
        aux = sqrt2opi * rho * exp(-rho^2 / 2)
        (erf_fn(rho / sqrt(2)) - aux, rho * aux)
    end
end

@testset "RectangularGaussianErfVortex" begin
    Random.seed!(51051)
    T = Float64
    n_src = 300
    n_tgt = 140
    src = rand(T, 7, n_src) .- 0.5
    src[7, :] .= 0.02 .+ 0.08 .* rand(n_src)          # sigma > 0
    tgt = zeros(T, 3, n_tgt)
    tgt[:, 1:100] .= rand(T, 3, 100) .- 0.5           # inside the cloud
    tgt[:, 101:110] .= src[1:3, 1:10]                 # coincident with sources
    tgt[:, 111:140] .= 50.0 .* (rand(T, 3, 30) .- 0.5)  # far away
    # also make some targets EXTREMELY close (but not equal) to sources: the
    # CPU semantics keep these pairs (no absolute eps2 cutoff)
    tgt[:, 1:5] .= src[1:3, 11:15] .+ 1e-8 .* randn(3, 5)

    out = zeros(T, 12, n_tgt)
    direct_rectangular!(out, tgt, RectangularGaussianErfVortex(), src; gradient=true)
    @test all(isfinite, out)

    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

    # layer 1a (STRUCTURAL, always, tight): all-pairs reference using the
    # implementation's own g/dgdr, so the gate isolates the pair-sum
    # structure (cross products, aux1/aux2 assembly, accumulation,
    # threading) with zero erf-implementation confound.
    ref = _point_reference(FLOWVPM.g_dgdr_gauserf, tgt, src)
    @test relerr(out[1:3, :], ref[1:3, :]) < 1e-14
    @test relerr(out[4:12, :], ref[4:12, :]) < 1e-14

    # layer 1b (CROSS-ERF): the same sums
    # referenced with an INDEPENDENT erf (openlibm). The two erfs agree to
    # 1 ulp, but g = erf(rho/sqrt2) - aux cancels catastrophically at tiny
    # rho: g ~ sqrt(2/pi) rho^3/3 while the erf value is ~sqrt(2/pi) rho, so
    # a 1-ulp erf difference is amplified to a RELATIVE g (hence J) error of
    # up to ~3 eps/rho^2 — measured up to 5.3e-2 at rho = 1e-7. The
    # deliberately near-coincident columns 1:5 (offset 1e-8, rho ~ 2e-7 to
    # 1e-6) therefore CANNOT meet an ulp-level gate against a different erf;
    # they get a per-column amplification-scaled gate instead. Regular
    # columns keep a tight gate: their smallest rho is O(1e-2), where the
    # amplification 3 eps/rho^2 is ~1e-11 on the pair with the smallest g —
    # and that pair's J contribution is ~rho^3 of the column scale, keeping
    # the column-normalized error at the low end of 1e-13; gate 1e-12 with
    # margin.
    begin
        # erf gate: FLOWVPM's fdlibm custom_erf must match openlibm to 1 ulp
        # absolutely everywhere and 1 ulp RELATIVELY at small arguments (the
        # cancellation-critical regime)
        worst_abs = 0.0
        for x in vcat(0.0, 10.0 .^ (-12:0.05:0.8))
            for s in (x, -x)
                worst_abs = max(worst_abs,
                    abs(FLOWVPM.custom_erf(s) - FLOWVPM.erf(s)))
            end
        end
        @test worst_abs < 5e-16
        worst_rel = 0.0
        for x in 10.0 .^ (-10:0.01:-4)
            worst_rel = max(worst_rel,
                abs(FLOWVPM.custom_erf(x) - FLOWVPM.erf(x)) /
                    FLOWVPM.erf(x))
        end
        @test worst_rel < 1e-15

        ref_sf = _point_reference(_g_dgdr_from_erf(FLOWVPM.erf), tgt, src)
        @test relerr(out[1:3, 6:end], ref_sf[1:3, 6:end]) < 1e-12
        @test relerr(out[4:12, 6:end], ref_sf[4:12, 6:end]) < 1e-12
        # near-coincident columns: per-column gate at the derived
        # amplification bound 3 eps/rho_min^2 (x16 safety for multiple
        # contributing near pairs), floored at 1e-12
        for i in 1:5
            rho_min = Inf
            for q in 1:n_src
                r = sqrt((tgt[1, i] - src[1, q])^2 + (tgt[2, i] - src[2, q])^2 +
                         (tgt[3, i] - src[3, q])^2)
                r > 0 && (rho_min = min(rho_min, r / src[7, q]))
            end
            gate = max(1e-12, 16 * 3 * eps(1.0) / rho_min^2)
            @test relerr(out[1:3, i:i], ref_sf[1:3, i:i]) < gate
            @test relerr(out[4:12, i:i], ref_sf[4:12, i:i]) < gate
        end
    end

    # U-only call matches the U rows of the gradient call
    out_u = zeros(T, 3, n_tgt)
    direct_rectangular!(out_u, tgt, RectangularGaussianErfVortex(), src; gradient=false)
    @test out_u == out[1:3, :]

    # accumulation semantics: second call doubles
    out2 = copy(out)
    direct_rectangular!(out2, tgt, RectangularGaussianErfVortex(), src; gradient=true)
    @test out2 ≈ 2 .* out rtol=1e-14

    # Float32 path (F32 variant of the point kernel)
    src32 = Float32.(src); tgt32 = Float32.(tgt)
    out32 = zeros(Float32, 12, n_tgt)
    direct_rectangular!(out32, tgt32, RectangularGaussianErfVortex(), src32; gradient=true)
    @test all(isfinite, out32)
    # columns 1:5 are 1e-8 from a source — that separation rounds away in F32
    # (positions O(0.5), eps(F32) ~ 6e-8), so compare well-separated targets
    @test relerr(Float64.(out32[1:3, 6:end]), ref[1:3, 6:end]) < 1e-4
end

