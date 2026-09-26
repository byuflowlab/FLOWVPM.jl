# Classic-VPM Euler stretching uses the pre-step Gamma for all three components
# (a sequential read-after-write fed the new G[1] into G[2] and G[3], 2026-09-26),
# and the SFS controls are finite on a zero-strength particle.
using Test, StaticArrays
import FLOWVPM
const vpm = FLOWVPM

@testset "classic Euler stretching and control guards" begin
    for transposed in (false, true)
        pf = vpm.ParticleField(10; formulation = vpm.ClassicVPM{Float64}(), kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct, transposed)
        vpm.add_particle(pf, SVector(0.0, 0.0, 0.0), SVector(1.0, 2.0, 3.0), 0.1)
        J = collect(1.0:9.0) .* 0.01
        pf.particles[vpm.J_INDEX, 1] .= J; pf.particles[vpm.U_INDEX, 1] .= 0
        G0 = copy(pf.particles[vpm.GAMMA_INDEX, 1]); dt = 0.5
        vpm._euler_cpu_classic!(pf, dt, zeros(3), vpm.kernel_gaussianerf.zeta(0.0))
        Jm = reshape(J, 3, 3)
        expected = G0 .+ dt .* ((transposed ? Jm' : Jm) * G0)
        @test pf.particles[vpm.GAMMA_INDEX, 1] ≈ expected atol = 1e-14
    end
    pf = vpm.ParticleField(10; formulation = vpm.formulation_rVPM, kernel = vpm.kernel_gaussianerf, UJ = vpm.UJ_direct)
    for k in 1:2
        vpm.add_particle(pf, SVector(0.1k, 0.0, 0.0), SVector(1.0, 2.0, 3.0), 0.1)
        pf.particles[vpm.SFS_INDEX, k] .= [5.0, -1.0, 0.5]; pf.particles[vpm.C_INDEX[1], k] = 1e6
    end
    pf.t = 1.0; pf.nt = 10
    vpm.control_magnitude(pf, 1); vpm.control_magnitude(vpm.get_particle(pf, 2), pf)
    @test pf.particles[vpm.SFS_INDEX, 1] == pf.particles[vpm.SFS_INDEX, 2] != [5.0, -1.0, 0.5]
    vpm.add_particle(pf, SVector(0.5, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 0.1); pf.particles[vpm.SFS_INDEX, 3] .= 1
    vpm.control_directional(pf, 3)
    @test all(isfinite, pf.particles[vpm.SFS_INDEX, 3])
    @test_throws ErrorException (for _ in 1:20; vpm.add_particle(pf, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), 0.1); end)
end
