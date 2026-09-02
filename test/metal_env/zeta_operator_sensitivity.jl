# Is the post-reset Gamma spread intrinsic to the RBF reset? Host-only: zeta_fmm vs exact zeta_direct
# through the same 6-step run with forced resets, Float64, plus single-evaluation zeta relerrs incl. the device.
include(joinpath(ENV["WT"], "test/metal_env/ka_backend.jl")); include(joinpath(ENV["WT"], "test/metal_env/pipeline_field.jl"))
using FLOWVPM, Printf; const V = FLOWVPM; import KernelAbstractions as KA
const NP = parse(Int, get(ENV, "PROF_NP", "4000")); const NSTEPS = 6
h0 = load_wake(36; np=NP, TF=Float64, P=5); np = V.get_np(h0); sgm0 = minimum(h0.particles[V.SIGMA_INDEX, 1:np]); nu = 1e-2
tol = parse(Float64, get(ENV, "CG_TOL", "1e-6")); itmax = parse(Int, get(ENV, "CG_ITMAX", "300"))
mk(zeta) = (f = V.ParticleField(h0.maxparticles, Float64; np, fmm=V.FMM(; p=6, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false, default_rho_over_sigma=1.0),
    viscous=V.CoreSpreading(nu, sgm0, zeta; beta=1.0001, itmax, tol, verbose=false, iterror=false), SFS=V.noSFS,
    relaxation=V.Relaxation(V.relax_pedrizzetti, 1, 0.3), integration=V.rungekutta3, transposed=true); f.particles .= h0.particles; f)
# single-evaluation zeta agreement
a = mk(V.zeta_fmm); b = mk(V.zeta_direct); V.zeta_fmm(a); V.zeta_direct(b)
Wf = copy(a.particles[V.VORTICITY_INDEX, 1:np]); Wd = copy(b.particles[V.VORTICITY_INDEX, 1:np])
rel(x, y) = maximum(abs.(x .- y)) / maximum(abs.(y))
if dev_functional()
    d = V.ParticleField(h0.maxparticles, Float32; arraytype=devmatrix, np, fmm=V.FMM(; p=6, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false, default_rho_over_sigma=1.0))
    d.particles .= devarray(Float32.(Array(h0.particles))); V.radix_fmm_settings!(d; m2l_strategy=:concat); V.zeta_fmm(d)
    Wdev = Float64.(Array(d.particles)[V.VORTICITY_INDEX, 1:np])
    @printf("zeta single eval: host_fmm vs direct %.2e | device vs direct %.2e | device vs host_fmm %.2e\n", rel(Wf, Wd), rel(Wdev, Wd), rel(Wdev, Wf))
else
    @printf("zeta single eval: host_fmm vs direct %.2e\n", rel(Wf, Wd))
end
# 6-step runs with resets
a = mk(V.zeta_fmm); b = mk(V.zeta_direct); dt = 0.02 * sgm0
ta = @elapsed V.run_vpm!(a, dt, NSTEPS; verbose=false, save_path=nothing, prompt=false)
tb = @elapsed V.run_vpm!(b, dt, NSTEPS; verbose=false, save_path=nothing, prompt=false)
A = a.particles; B = b.particles
rel2(rows) = sqrt(sum(abs2, A[rows, 1:np] .- B[rows, 1:np]) / sum(abs2, B[rows, 1:np]))
@printf("host zeta_fmm vs host zeta_direct, np=%d, %d steps, resets (t_sgm %.2f / %.2f), tol=%g itmax=%d: Gamma max %.2e L2 %.2e | sigma max %.2e | U max %.2e | %.0f s / %.0f s\n",
    np, NSTEPS, a.viscous.t_sgm, b.viscous.t_sgm, tol, itmax, rel(A[V.GAMMA_INDEX,1:np], B[V.GAMMA_INDEX,1:np]), rel2(V.GAMMA_INDEX),
    rel(A[V.SIGMA_INDEX,1:np], B[V.SIGMA_INDEX,1:np]), rel(A[V.U_INDEX,1:np], B[V.U_INDEX,1:np]), ta, tb)
