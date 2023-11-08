# activate test environment
if splitpath(Base.active_project())[end-1] == "FLOWVPM.jl"
    import TestEnv
    TestEnv.activate()
end
import FLOWVPM
vpm = FLOWVPM
bson = vpm.BSON

verbose1 = false
verbose2 = true
global this_is_a_test = true # we don't want to import PyPlot or anything else

examples_path = joinpath(dirname(pathof(FLOWVPM)), "..", "examples", "vortexrings")
include(joinpath(examples_path, "vortexrings.jl"))

function benchmark(; formulation=vpm.rVPM, nrings=1, Nphi=100, nc=1, overwrite_bson=true)
    # -------------- SIMULATION PARAMETERS -------------------------------------
    integration = vpm.euler                 # time integration scheme
    nsteps    = 1                           # Number of time steps
    dt        = 1e-2                        # size of a timestep in seconds
    dZ        = 0.1                         # (m) spacing between rings
    circulations = 1.0*ones(nrings)         # (m^2/s) circulation of each ring
    Rs        = 1.0*ones(nrings)            # (m) radius of each ring
    ARs       = 1.0*ones(nrings)            # Aspect ratio AR = a/r of each ring
    Rcrosss   = 0.15*Rs                     # (m) cross-sectional radii
    sigmas    = Rcrosss                     # Particle smoothing of each radius
    Nphis     = Nphi*ones(Int, nrings)       # Number of cross sections per ring
    ncs       = nc*ones(Int, nrings)        # Number layers per cross section
    extra_ncs = 0*ones(Int, nrings)         # Number of extra layers per cross section
    Os        = [[0, 0, dZ*(ri-1)] for ri in 1:nrings]  # Position of each ring
    Oaxiss    = [I for ri in 1:nrings]      # Orientation of each ring
    nref      = 1                           # Reference ring

    beta      = 0.5                         # Parameter for theoretical velocity
    faux      = 0.25                        # Shrinks the discretized core by this factor

    # -------------- TIMESTEPS -------------------------------------------------
    Uref = Uring(circulations[nref], Rs[nref], Rcrosss[nref], beta) # (m/s) reference velocity
    # dt = (Rtot/Uref) / nsteps
    Rtot = dt * nsteps * Uref               # (m) run simulation for equivalent
                                            #     time to this many radii

    # -------------- SOLVER SETTINGS -------------------------------------------
    solver_fmm = (
        formulation   = formulation,
        SFS           = vpm.noSFS,
        relaxation    = vpm.pedrizzetti,
        kernel        = vpm.winckelmans,
        viscous       = vpm.Inviscid(),
        transposed    = true,
        integration   = integration,
        UJ            = vpm.UJ_fmm,
        fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4)
    )

    # --------------- PREPARE ARCHIVES -----------------------------------------
    if !isfile("benchmark_fmm.bson") || overwrite_bson
        formulation_log = []
        nrings_log = []
        Nphis_log = []
        ncs_log = []
        nparticles_log = []
        t_log = []
        bson.@save "benchmark_fmm.bson" formulation_log nrings_log Nphis_log ncs_log nparticles_log t_log
    end

    # --------------- RUN SIMULATION -------------------------------------------
    pfield = run_vortexring_simulation(  nrings, circulations,
                                        Rs, ARs, Rcrosss,
                                        Nphis, ncs, extra_ncs, sigmas,
                                        Os, Oaxiss;
                                        # ------- SIMULATION OPTIONS -----------
                                        nref=nref,
                                        nsteps=nsteps,
                                        Rtot=Rtot,
                                        beta=beta,
                                        faux=faux,
                                        # ------- OUTPUT OPTIONS ---------------
                                        save_path=nothing,
                                        calc_monitors=false,
                                        verbose=verbose1, v_lvl=1,
                                        verbose_nsteps=ceil(Int, nsteps/4),
                                        pfieldargs=solver_fmm
                                        )

    println("===== BEGIN TEST =====")
    vpm.UJ_fmm(pfield)
    t = @elapsed vpm.UJ_fmm(pfield)
    println("\ttime:\t\t$t s")

    bson.@load "benchmark_fmm.bson" formulation_log nrings_log Nphis_log ncs_log nparticles_log t_log
    push!(formulation_log, solver_fmm.formulation)
    push!(nrings_log, nrings)
    push!(Nphis_log, Nphis)
    push!(ncs_log, ncs)
    push!(nparticles_log, pfield.np)
    println("\tnparticles:\t$(pfield.np)")
    push!(t_log, t)
    bson.@save "benchmark_fmm.bson" formulation_log nrings_log Nphis_log ncs_log nparticles_log t_log

end

# using ProfileView
# @profview benchmark(; formulation=vpm.rVPM, nrings=1, Nphi=10, nc=0, overwrite_bson=true) # 10 radii long column

benchmark(; formulation=vpm.rVPM, nrings=1, Nphi=10, nc=0, overwrite_bson=true) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=10, nc=1, overwrite_bson=true) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=20, nc=2, overwrite_bson=false) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=30, nc=3, overwrite_bson=false) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=40, nc=4, overwrite_bson=false) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=50, nc=5, overwrite_bson=false) # 10 radii long column
benchmark(; formulation=vpm.rVPM, nrings=100, Nphi=100, nc=10, overwrite_bson=false) # 10 radii long column
