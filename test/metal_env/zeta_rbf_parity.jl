# Device zeta_fmm and rbf_conjugategradient vs the host implementations on the real wake.
include(joinpath(ENV["WT"], "test/metal_env/ka_backend.jl")); include(joinpath(ENV["WT"], "test/metal_env/pipeline_field.jl"))
using FLOWVPM, Printf, Statistics; const V = FLOWVPM; import KernelAbstractions as KA
dev_functional() || (println("skipping"); exit(0))
const NP = (s = get(ENV, "PROF_NP", ""); isempty(s) ? nothing : parse(Int, s))
host = load_wake(36; np=NP, TF=Float64, P=5)
np = V.get_np(host)
dev = V.ParticleField(host.maxparticles, Float32; arraytype=devmatrix, np=np,
    fmm=V.FMM(; p=6, autotune_p=false, autotune_ncrit=false, autotune_reg_error=false, default_rho_over_sigma=1.0))
dev.particles .= devarray(Float32.(Array(host.particles)))
V.radix_fmm_settings!(dev; m2l_strategy=:concat)
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
# --- zeta ---
V.zeta_fmm(host); Wh = copy(host.particles[V.VORTICITY_INDEX, 1:np])
V.zeta_fmm(dev);  Wd = Float64.(Array(dev.particles)[V.VORTICITY_INDEX, 1:np])
t_h = @elapsed V.zeta_fmm(host)
t_d = minimum(begin t0 = time_ns(); V.zeta_fmm(dev); KA.synchronize(KA.get_backend(dev.particles)); (time_ns() - t0) / 1e9 end for _ in 1:5)
@printf("zeta  np=%d  relerr(dev vs host) = %.3e   host %.3f s   device %.4f s\n", np, relerr(Wd, Wh), t_h, t_d)
# --- rbf conjugate gradient (target = the zeta field itself, as the reset does) ---
sgm0 = minimum(host.particles[V.SIGMA_INDEX, 1:np])
mk() = V.CoreSpreading(1e-3, sgm0, V.zeta_fmm; itmax=30, tol=1e-3, verbose=false, iterror=false)
host.particles[V.M_INDEX[7:9], 1:np] .= Wh
dev.particles[V.M_INDEX[7:9], 1:np] .= devarray(Float32.(Wh))
Gh0 = copy(host.particles[V.GAMMA_INDEX, 1:np])
csh = mk(); t_rh = @elapsed V.rbf_conjugategradient(host, csh)
csd = mk(); t_rd = @elapsed V.rbf_conjugategradient(dev, csd)
Gh = host.particles[V.GAMMA_INDEX, 1:np]; Gd = Float64.(Array(dev.particles)[V.GAMMA_INDEX, 1:np])
@printf("rbf   host: err %s  %.3f s | device: err %s  %.3f s | relerr(Gamma dev vs host) = %.3e | |dGamma|/|Gamma0| host %.3e\n",
    string(round.(sqrt.(csh.rrs ./ csh.rr0s); sigdigits=3)), t_rh, string(round.(sqrt.(csd.rrs ./ csd.rr0s); sigdigits=3)), t_rd,
    relerr(Gd, Gh), relerr(Gh, Gh0))
