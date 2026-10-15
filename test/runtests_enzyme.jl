using Test
import Random
import LinearAlgebra
import FLOWVPM
import Enzyme
import ForwardDiff

# Enzyme reverse mode (default settings) through one UJ_direct time step:
# dot-product test of the gradient against central FD along random directions.
# The field is built inside the differentiated function (FLOWVPMEnzymeExt marks
# its filament workspaces and the ρ/σ tuning inactive). Objective: weights on
# [X; Gamma; sigma] after the step; inputs x = [X; Gamma; sigma] before it.
# Three configurations cover every option of each axis at least once.

const ENZ_N = 50

function enz_step(x::AbstractVector{T}, w, cfg) where T
    n = ENZ_N
    pf = FLOWVPM.ParticleField(n, T; formulation=cfg.formulation, kernel=cfg.kernel, UJ=FLOWVPM.UJ_direct,
                               SFS=cfg.SFS, relaxation=cfg.relaxation, integration=cfg.integration,
                               fmm=FLOWVPM.FMM(autotune_reg_error=false))
    for i in 1:n
        FLOWVPM.add_particle(pf, (x[3i-2], x[3i-1], x[3i]), (x[3n+3i-2], x[3n+3i-1], x[3n+3i]), x[6n+i])
    end
    FLOWVPM.nextstep(pf, 0.01; relax=cfg.relax)
    s = zero(T)
    for i in 1:n
        X = FLOWVPM.get_X(pf, i); G = FLOWVPM.get_Gamma(pf, i)
        s += w[3i-2]*X[1] + w[3i-1]*X[2] + w[3i]*X[3] +
             w[3n+3i-2]*G[1] + w[3n+3i-1]*G[2] + w[3n+3i]*G[3] + w[6n+i]*FLOWVPM.get_sigma(pf, i)[]
    end
    return s
end

@testset "Enzyme reverse: one UJ_direct step" begin
    pedr = FLOWVPM.Relaxation(FLOWVPM.relax_pedrizzetti, 1, 0.3)
    cpedr = FLOWVPM.Relaxation(FLOWVPM.relax_correctedpedrizzetti, 1, 0.3)
    constsfs = FLOWVPM.ConstantSFS(FLOWVPM.Estr_fmm; Cs=1.0, clippings=(FLOWVPM.clipping_backscatter,))
    cfgs = (
        (name="classic, gaussianerf, pedrizzetti, constant SFS, euler", formulation=FLOWVPM.formulation_classic,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=pedr, relax=true, SFS=constsfs, integration=FLOWVPM.euler),
        (name="rVPM, winckelmans, correctedpedrizzetti, dynamic SFS, rungekutta3", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_winckelmans, relaxation=cpedr, relax=true, SFS=FLOWVPM.SFS_Cd_twolevel_nobackscatter,
         integration=FLOWVPM.rungekutta3),
        (name="rVPM, gaussianerf, no relaxation, no SFS, rungekutta3", formulation=FLOWVPM.formulation_rVPM,
         kernel=FLOWVPM.kernel_gaussianerf, relaxation=cpedr, relax=false, SFS=FLOWVPM.SFS_none,
         integration=FLOWVPM.rungekutta3),
    )
    n = ENZ_N
    Random.seed!(1)
    x = vcat(rand(3n), 0.1 .* randn(3n), 0.15 .+ 0.02 .* rand(n))
    w = randn(7n)
    for cfg in cfgs
        @testset "$(cfg.name)" begin
            f = x -> enz_step(x, w, cfg)
            g = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, enz_step, Enzyme.Active, Enzyme.Duplicated(x, g), Enzyme.Const(w), Enzyme.Const(cfg))
            @test all(isfinite, g)
            for k in 1:3
                v = randn(7n); v ./= LinearAlgebra.norm(v)
                gv = LinearAlgebra.dot(g, v)
                # FD truncation error is O(h^2), roundoff O(1/h): best of three steps
                err = minimum(h -> abs((f(x .+ h .* v) - f(x .- h .* v))/(2h) - gv)/abs(gv), (1e-4, 1e-5, 1e-6))
                @test err < 1e-5
            end
            if cfg.SFS === FLOWVPM.SFS_none   # ForwardDiff through custom_erf (SFS self pair is 0/0 for Dual)
                v = randn(7n)
                @test isapprox(ForwardDiff.derivative(t -> f(x .+ t .* v), 0.0), LinearAlgebra.dot(g, v); rtol=1e-10)
            end
        end
    end
end
