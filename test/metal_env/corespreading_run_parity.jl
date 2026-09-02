# Multi-step run_vpm! parity, host (Float64) vs device (Float32): RK3 + CoreSpreading (forced resets)
# + DynamicSFS + Pedrizzetti relaxation + save(). Real wake step 36, optionally subsampled (PROF_NP).
include(joinpath(ENV["WT"], "test/metal_env/ka_backend.jl")); include(joinpath(ENV["WT"], "test/metal_env/pipeline_field.jl"))
using FLOWVPM, Printf; const V = FLOWVPM; import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
const NP = (s = get(ENV, "PROF_NP", ""); isempty(s) ? nothing : parse(Int, s))
const NSTEPS = parse(Int, get(ENV, "NSTEPS", "3"))
h0 = load_wake(36; np=NP, TF=Float64, P=5)
np = V.get_np(h0); sgm0 = minimum(h0.particles[V.SIGMA_INDEX, 1:np]); nu = 1e-2
estr = isdefined(V, :Estr_fmm) ? V.Estr_fmm : V.Estr_direct
mkfield(R, arraytype) = V.ParticleField(h0.maxparticles, R; arraytype, np,
    fmm=V.FMM(; p=6, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false, default_rho_over_sigma=1.0),
    viscous=V.CoreSpreading(nu, sgm0, V.zeta_fmm; beta=parse(Float64, get(ENV, "CS_BETA", "1.0001")), itmax=parse(Int, get(ENV, "CG_ITMAX", "30")), tol=parse(Float64, get(ENV, "CG_TOL", "1e-3")), verbose=false, iterror=false),   # Float64 schemes on purpose: the constructor coerces
    SFS=(get(ENV, "SFS_MODE", "dynamic") == "none" ? V.noSFS : V.DynamicSFS(estr)), relaxation=V.Relaxation(V.relax_pedrizzetti, get(ENV, "RELAX", "on") == "none" ? 10^6 : 1, 0.3), integration=V.rungekutta3, transposed=true)
host = mkfield(Float64, Matrix); host.particles .= h0.particles
const DEV_TF = getfield(Base, Symbol(get(ENV, "DEV_TF", "Float32")))
dev = mkfield(DEV_TF, devmatrix); dev.particles .= devarray(DEV_TF.(Array(h0.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
dt = 0.02 * sgm0
outdir = joinpath(ENV["SCRATCH"], "cs_run_$(DEV_TF)_$(getpid())"); rm(outdir; force=true, recursive=true)
th = @elapsed V.run_vpm!(host, dt, NSTEPS; verbose=false, save_path=nothing, prompt=false)
td = @elapsed V.run_vpm!(dev, dt, NSTEPS; verbose=false, save_path=outdir, run_name="dev", prompt=false, create_savepath=true)
H = host.particles; D = Float64.(Array(dev.particles))
rel(rows) = maximum(abs.(D[rows, 1:np] .- H[rows, 1:np])) / maximum(abs.(H[rows, 1:np]))
rel2(rows) = sqrt(sum(abs2, D[rows, 1:np] .- H[rows, 1:np]) / sum(abs2, H[rows, 1:np]))
frac(rows; thr=1e-4) = count(>(thr * maximum(abs.(H[rows, 1:np]))), vec(maximum(abs.(D[rows, 1:np] .- H[rows, 1:np]); dims=1))) / np
@printf("dev=%s sfs=%s relax=%s beta=%.5f np=%d nsteps=%d  host %.1f s  device %.2f s | relerr X %.2e  Gamma %.2e  sigma %.2e  U %.2e | sigma range host [%.4g, %.4g] dev [%.4g, %.4g] | resets host t_sgm=%.3g dev t_sgm=%.3g | saved: %s\n",
    string(DEV_TF), get(ENV, "SFS_MODE", "dynamic"), get(ENV, "RELAX", "on"), host.viscous.beta, np, NSTEPS, th, td, rel(V.X_INDEX), rel(V.GAMMA_INDEX), rel(V.SIGMA_INDEX), rel(V.U_INDEX),
    extrema(H[V.SIGMA_INDEX,1:np])..., extrema(D[V.SIGMA_INDEX,1:np])..., host.viscous.t_sgm, dev.viscous.t_sgm, join(readdir(outdir), ","))
@printf("L2: Gamma %.2e  U %.2e | particles with |dGamma| > 1e-4 max: %.4f  > 1e-3 max: %.4f\n", rel2(V.GAMMA_INDEX), rel2(V.U_INDEX), frac(V.GAMMA_INDEX), frac(V.GAMMA_INDEX; thr=1e-3))
# warm per-step cost (everything compiled): a second run of the same length
th2 = @elapsed V.run_vpm!(host, dt, NSTEPS; verbose=false, save_path=nothing, prompt=false)
td2 = @elapsed V.run_vpm!(dev, dt, NSTEPS; verbose=false, save_path=nothing, prompt=false)
@printf("warm: host %.2f s/step   device %.3f s/step   resets host t_sgm=%.3g dev t_sgm=%.3g\n", th2/NSTEPS, td2/NSTEPS, host.viscous.t_sgm, dev.viscous.t_sgm)
