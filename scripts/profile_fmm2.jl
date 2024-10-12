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
using Random
using Statistics

function create_pfield(n_particles, seed; circulation=1.0, Lx=1.0, Ly=1.0, Lz=7.0, overlap=1.3, theta=0.4, p=4, ncrit=50, nonzero_sigma=false, add_noise=true, relative_error=0.25)
    Random.seed!(seed)
    v_particle = Lx*Ly*Lz / n_particles
    d_particle = v_particle^(1/3)
    n_x = Int(div(Lx,d_particle))
    n_y = Int(div(Ly,d_particle))
    n_z = Int(div(Lz,d_particle))
    n_particles = n_x * n_y * n_z
    Lx = d_particle * n_x
    Ly = d_particle * n_y
    Lz = d_particle * n_z
    pfield = vpm.ParticleField(n_particles; formulation=vpm.formulation_rVPM, UJ=vpm.UJ_fmm, fmm=vpm.FMM(;theta=theta, p=p, ncrit=ncrit, nonzero_sigma=nonzero_sigma, relative_error))
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

function benchmark_fmm(n_particles; circulation=1.0, Lx=1.0, Ly=1.0, Lz=7.0, overlap=1.3, theta=0.4, p=4, ncrit=50, nonzero_sigma=false, relative_error=0.25)
    pfield, nparticles = create_pfield(n_particles; circulation, Lx, Ly, Lz, overlap, theta, p, ncrit, nonzero_sigma, relative_error)
    pfield.UJ(pfield)
    t = @elapsed pfield.UJ(pfield)
    return t, nparticles
end

println("===== BEGIN VPM+FMM BENCHMARK: $(Threads.nthreads()) THREADS")

function cost_error(relative_error, U_direct, i=7)
    seed = 123
    n_particles = 4^i
    println("\tRequested np:\t$n_particles")
    circulation=1.0
    Lx=1.0
    Ly=1.0
    Lz=7.0
    overlap=1.3
    theta=0.4
    p=20
    ncrit=7
    nonzero_sigma=true
    pfield, nparticles = create_pfield(n_particles, seed; circulation, Lx, Ly, Lz, overlap, theta, p, ncrit, nonzero_sigma, relative_error)

    # time cost
    @elapsed pfield.UJ(pfield)
    t = @elapsed pfield.UJ(pfield)
    #=

    # profile
    @profile pfield.UJ(pfield)
    Profile.clear()
    @profile pfield.UJ(pfield)

    pprof()
    =#

    # check error
    println("\tchecking error...")
    # fmm
    FLOWVPM._reset_particles(pfield)
    pfield.UJ(pfield)
    U_fmm = zeros(3,pfield.np)
    for i in 1:pfield.np
        U_fmm[:,i] .= pfield.particles[10:12,i]
    end

    if isnothing(U_direct)
        # direct
        FLOWVPM._reset_particles(pfield)
        FLOWVPM.UJ_direct(pfield)
        U_direct = zeros(3,pfield.np)
        for i in 1:pfield.np
            U_direct[:,i] .= pfield.particles[10:12,i]
        end
    end

    # error
    err = U_direct .- U_fmm
    norm_err = sqrt.(sum(err.^2,dims=1))
    norm_direct = sqrt.(sum(U_direct .^2, dims=1))
    rel_err = norm_err ./ norm_direct
    #ε = mean(rel_err)
    ε = maximum(rel_err)

    return t, ε, U_direct
end

function cost_error_2(rtols, U_direct=nothing)
    ts = zeros(length(rtols))
    εs = zeros(length(rtols))
    for (i,rtol) in enumerate(rtols)
        println("$i: requested rtol = $rtol")
        t, ε, U_direct = cost_error(rtol,nothing)
        ts[i] = t
        εs[i] = ε
        @show U_direct[:,1]
    end
    return ts, εs, U_direct
end

#rtols = [10.0^n for n in [-16,-8,-4,-2,-1,-0.5,0]]
rtols = [10.0^n for n in [-4,-2,-1,0]]
#ts, εs, U_direct = cost_error_2(rtols)
#rtols2 = [10.0^n for n in [1.0,2.0,3.0]]
#ts2, εs2 = cost_error(rtols2)


seed = 123
n_particles = 4^i
println("\tRequested np:\t$n_particles")
circulation=1.0
Lx=1.0
Ly=1.0
Lz=7.0
overlap=1.3
theta=0.4
p=4
ncrit=10
nonzero_sigma=false
relative_error = 0.1
pfield, nparticles = create_pfield(n_particles, seed; circulation, Lx, Ly, Lz, overlap, theta, p, ncrit, nonzero_sigma, relative_error)

# time cost
@time pfield.UJ(pfield)
@time pfield.UJ(pfield)

# profile
@profile pfield.UJ(pfield)
Profile.clear()
@profile pfield.UJ(pfield)

pprof()
