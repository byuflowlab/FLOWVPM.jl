using FLOWVPM

const NSTATES = 7

function vortex_ring(n_particles, radius, circulation; overlap=1.3)
    pfield = ParticleField(n_particles)
    vortex_ring!(pfield, n_particles, radius, circulation; overlap)
    return pfield
end

function vortex_ring!(pfield, n_particles::Int, radius::Float64, circulation::Float64; overlap=1.3)
    # get particle positions
    θ = LinRange(0, 2π, n_particles + 1)[1:end-1]
    x = radius * cos.(θ)
    y = radius * sin.(θ)
    z = zeros(n_particles)
    positions = hcat(x, y, z)'

    # Calculate particle radius for desired overlap
    arc_length = 2π * radius / n_particles
    particle_radius = overlap * arc_length / 2

    # add to particle field
    Gamma_mag = circulation / n_particles
    for i in 1:n_particles
        X = SVector{3}(positions[i, :])
        G = cross(SVector{3}(0.0,0,1), X)
        G *= Gamma_mag / norm(G)
        add_particle!(pfield, X, G, particle_radius)
    end

    return pfield
end

function pfield_2_vec!(y, pfield::ParticleField)
    y .= vec(view(pfield.particles, 1:NSTATES, 1:pfield.np))
end

function vec_2_pfield!(pfield::ParticleField, y)
    # get number of particles
    np, rem = divrem(length(y), NSTATES)
    @assert rem == 0 "y length is not a multiple of particle state length"
    @assert np <= pfield.nmax "number of particles exceeds pfield maximum"
    
    # copy vector to pfield
    view(pfield.particles, 1:NSTATES, 1:np) .= y
    pfield.np = np
end

function odestep!(pfield, y, yprev, t, tprev, p)
    # extract parameters
    _, _, save_switch, save_path, run_name = p

    # calculate time step
    dt = t - tprev

    # set tprev
    pfield.t = tprev

    # copy yprev to pfield
    vec_2_pfield!(pfield, yprev)

    # determine whether or not to relax particle strengths
    i = pfield.nt
    relax = pfield.relaxation != FLOWVPM.relaxation_none &&
            pfield.relaxation.nsteps_relax >= 1 &&
            i>0 && (i%pfield.relaxation.nsteps_relax == 0)

    # convect particle field
    FLOWVPM.nextstep(pfield, dt; relax)

    # extract updated particle positions
    pfield_2_vec!(y, pfield)

    # update timestep number
    pfield.nt += 1

    # save particle field
    save_switch && eltype(pfield) <: AbstractFloat && (save(pfield, run_name; path=save_path, add_num=true, overwrite_time=pfield.nt))
end

n_particles = 100
pfield = ParticleField(n_particles)
onestep!(y, yprev, t, tprev, xd, xci, p) = odestep!(pfield, y, yprev, t, tprev, p)



"""
assume xd contains vortex ring initial radius and circulation
"""
function initialize(t0, xd, xc0, p)
    # extract design variables
    r, gamma = xd

    # extract parameters
    nparticles, overlap, save_switch, save_path, run_name = p
    
    # initialize vortex ring
    vortex_ring!(pfield, nparticles, r, gamma; overlap)

    # save iniital particle state
    save_switch && eltype(pfield) <: AbstractFloat && (save(pfield, run_name; path=save_path, add_num=true, overwrite_time=pfield.nt))
end
