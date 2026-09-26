# Device relaxation (both Pedrizzetti forms) against the host functions on the
# same random field, in the device's own float type; a zero-strength particle
# stays finite under the corrected form. Run from an environment with the
# device package loaded (LiftingLines test/metal_env includes al_backend.jl).
using Test, Random, StaticArrays
import FLOWVPM
const vpm = FLOWVPM
include(joinpath(@__DIR__, "..", "..", "LiftingLines", "test", "gpu", "al_backend.jl"))   # devmatrix, DEV_TF_DEFAULT, dev_functional
dev_functional() || (println("no functional device; skipping"); exit(0))
TF = DEV_TF_DEFAULT

function fields(TF, rlx)
    Random.seed!(7)
    n = 257
    mk(at) = vpm.ParticleField(n + 8, TF; formulation = vpm.formulation_rVPM, kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct,
                               relaxation = vpm.Relaxation(rlx, 1, TF(0.3)), arraytype = at)
    host = mk(Matrix)
    for i in 1:n
        vpm.add_particle(host, SVector{3,TF}(randn(TF, 3)), SVector{3,TF}(randn(TF, 3)), TF(0.1))
    end
    host.particles[vpm.J_INDEX, 1:n] .= randn(TF, 9, n)
    host.particles[vpm.GAMMA_INDEX, n] .= 0            # a zero-strength particle
    dev = mk(devmatrix)
    copyto!(dev.particles, host.particles); dev.np = host.np
    return host, dev, n
end

@testset "device relaxation matches the host" begin
    for rlx in (vpm.relax_pedrizzetti, vpm.relax_correctedpedrizzetti)
        host, dev, n = fields(TF, rlx)
        vpm._relax_broadcast!(rlx, TF(0.3), host)
        vpm._relax_broadcast!(rlx, TF(0.3), dev)
        Gh = host.particles[vpm.GAMMA_INDEX, 1:n]; Gd = Array(dev.particles)[vpm.GAMMA_INDEX, 1:n]
        @test all(isfinite, Gh) && all(isfinite, Gd)
        @test maximum(abs.(Gh .- Gd)) <= 4 * eps(TF) * maximum(abs.(Gh))
        # against the per-particle scalar reference on a second host copy
        ref, _, _ = fields(TF, rlx)
        for i in 1:n; rlx(TF(0.3), ref, i); end
        @test maximum(abs.(ref.particles[vpm.GAMMA_INDEX, 1:n] .- Gh)) <= 4 * eps(TF) * maximum(abs.(Gh))
    end
end
