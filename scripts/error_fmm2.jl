# activate test environment
using Pkg
if splitpath(Base.active_project())[end-1] !== "FLOWVPM.jl"
    this_dir = @__DIR__
    Pkg.activate(normpath(joinpath(this_dir,"..")))
end
import FLOWVPM
vpm = FLOWVPM
bson = vpm.BSON

using Profile
using PProf

function create_pfield(n_particles; circulation=1.0, Lx=1.0, Ly=1.0, Lz=7.0, overlap=1.3, theta=0.4, p=4, ncrit=50, nonzero_sigma=false, add_noise=true, relative_error=0.0)
    v_particle = Lx*Ly*Lz / n_particles
    d_particle = v_particle^(1/3)
    n_x = Int(div(Lx,d_particle))
    n_y = Int(div(Ly,d_particle))
    n_z = Int(div(Lz,d_particle))
    n_particles = n_x * n_y * n_z
    Lx = d_particle * n_x
    Ly = d_particle * n_y
    Lz = d_particle * n_z
    pfield = vpm.ParticleField(n_particles; formulation=vpm.formulation_rVPM, UJ=vpm.UJ_fmm, fmm=vpm.FMM(;theta, p, ncrit, nonzero_sigma, relative_error))
    Gamma_base = circulation / n_particles * [0,0,1.0]
    for x in range(d_particle/2,stop=Lx,step=d_particle)
        for y in range(d_particle/2,stop=Ly,step=d_particle)
            for z in range(d_particle/2,stop=Lz,step=d_particle)
                X = x,y,z
                Gamma = add_noise ? Gamma_base + (rand(vpm.SVector{3}) .- 0.5) ./ 10 : Gamma_base
                sigma = add_noise ? d_particle/2*overlap + (rand() - 0.5) * d_particle/2*overlap/10 : d_particle/2*overlap
                vpm.add_particle(pfield, X, Gamma, sigma)
            end
        end
    end
    return pfield, n_particles
end

function benchmark_fmm(n_particles; circulation=1.0, Lx=1.0, Ly=1.0, Lz=7.0, overlap=1.3, theta=0.4, p=4, ncrit=50, nonzero_sigma=false, relative_error=0.0)
    pfield, nparticles = create_pfield(n_particles; circulation, Lx, Ly, Lz, overlap, theta, p, ncrit, nonzero_sigma, relative_error)
    pfield.UJ(pfield)
    t = @elapsed pfield.UJ(pfield)
    return t, nparticles
end

println("===== BEGIN VPM+FMM BENCHMARK: $(Threads.nthreads()) THREADS")
i = 7
#n_particles = [4^i for i in 5:11]
n_particles = 4^i

println("Requested np:\t$n_particles")
circulation=1.0
Lx=1.0
Ly=1.0
Lz=7.0
overlap=1.3
theta=0.4
p=4
ncrit=7
nonzero_sigma=true
relative_error=0.25
pfield, nparticles = create_pfield(n_particles; circulation, Lx, Ly, Lz, overlap, theta, p, ncrit, nonzero_sigma, relative_error)

# fmm
FLOWVPM._reset_particles(pfield)
pfield.UJ(pfield)
U_fmm = zeros(3,pfield.np)
for i in 1:pfield.np
    U_fmm[:,i] .= pfield.particles[10:12,i]
end

# direct
FLOWVPM._reset_particles(pfield)
FLOWVPM.UJ_direct(pfield)
U_direct = zeros(3,pfield.np)
for i in 1:pfield.np
    U_direct[:,i] .= pfield.particles[10:12,i]
end

# error
err = U_direct .- U_fmm
ε = sum(sqrt.(sum(err .^2, dims=1))) / pfield.np
