# activate test environment
if splitpath(Base.active_project())[end-1] == "FLOWVPM.jl"
    import TestEnv
    TestEnv.activate()
end
using Test
import Printf: @printf
import FLOWVPM
vpm = FLOWVPM


for (description, integration, UJ, nc) in (
                                            ("Euler time-integration + direct UJ", vpm.euler, vpm.UJ_direct, 0),
                                            ("Runge-Kutta time-integration + direct UJ", vpm.rungekutta3, vpm.UJ_direct, 0),
                                            ("FMM UJ", vpm.euler, vpm.UJ_fmm, 0),
                                            ("Full inviscid scheme", vpm.rungekutta3, vpm.UJ_fmm, 1),
                                          )

    println("\n"^2*description*" test: Single vortex ring...")

    @test begin

        verbose1 = false
        verbose2 = true
        global this_is_a_test = true

        examples_path = joinpath(dirname(pathof(FLOWVPM)), "..", "examples", "vortexrings")
        include(joinpath(examples_path, "vortexrings.jl"))

        # -------------- SIMULATION PARAMETERS -------------------------------------
        nsteps    = 100                         # Number of time steps
        Rtot      = 2.0                         # (m) run simulation for equivalent
                                                #     time to this many radii
        nrings    = 1                           # Number of rings
        dZ        = 0.1                         # (m) spacing between rings
        circulations = 1.0*ones(nrings)         # (m^2/s) circulation of each ring
        Rs        = 1.0*ones(nrings)            # (m) radius of each ring
        ARs       = 1.0*ones(nrings)            # Aspect ratio AR = a/r of each ring
        Rcrosss   = 0.15*Rs                     # (m) cross-sectional radii
        sigmas    = Rcrosss                     # Particle smoothing of each radius
        Nphis     = 100*ones(Int, nrings)       # Number of cross sections per ring
        ncs       = nc*ones(Int, nrings)        # Number layers per cross section
        extra_ncs = 0*ones(Int, nrings)         # Number of extra layers per cross section
        Os        = [[0, 0, dZ*(ri-1)] for ri in 1:nrings]  # Position of each ring
        Oaxiss    = [I for ri in 1:nrings]      # Orientation of each ring
        nref      = 1                           # Reference ring

        beta      = 0.5                         # Parameter for theoretical velocity
        faux      = 0.25                        # Shrinks the discretized core by this factor

        # -------------- SOLVER SETTINGS -------------------------------------------
        solver = (
            formulation   = vpm.cVPM,
            SFS           = vpm.noSFS,
            relaxation    = vpm.pedrizzetti,
            kernel        = vpm.winckelmans,
            viscous       = vpm.Inviscid(),
            transposed    = true,
            integration   = integration,
            UJ            = UJ,
            fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true)
        )


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
                                            pfieldargs=solver
                                            )
        t_elapsed = @elapsed pfield = run_vortexring_simulation(  nrings, circulations,
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
                                            pfieldargs=solver
                                            )
        # --------------- COMPARE TO ANALYTIC SOLUTION -----------------------------

        # Calculate resulting ring velocity
        tend = pfield.t                               # (s) simulation end time
        Z_vpm = [zeros(3) for ri in 1:nrings]         # Centroid position
        R_vpm, sgm_vpm = zeros(nrings), zeros(nrings) # Ring and cross section radii
        intervals = calc_ring_invervals(nrings, Nphis, ncs, extra_ncs)
        calc_rings_weighted!(Z_vpm, R_vpm, sgm_vpm, pfield, nrings, intervals)

        U_vpm = norm(Z_vpm[1] - Os[1]) / tend

        # Calculate analytic ring velocity
        U_ana = Uring(circulations[nref], Rs[nref], Rcrosss[nref], beta)

        # Error
        err = (U_vpm - U_ana) / U_ana

        if verbose2
            @printf "%sVortex ring self-induced velocity verification\n"    "\n"*"\t"^1
            @printf "%sAnalytical velocity:\t\t%1.3f m/s\n"                 "\t"^2 U_ana
            @printf "%sResulting velocity:\t\t%1.3f m/s\n"                  "\t"^2 U_vpm
            @printf "%sError:\t\t\t\t%1.8f﹪\n"                             "\t"^2 err*100
            @printf "%sTime:\t\t\t\t%1.8f s\n"                             "\t"^2 t_elapsed
        end

        # Test result
        @test abs(err) < 0.01
    end
end
