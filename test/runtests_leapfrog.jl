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
                                            ("Vortex stretching + Classic VPM test: Thin Leapfrog...", vpm.rungekutta3, vpm.UJ_fmm, 0),
                                            ("Vortex stretching + Classic VPM test: Thick Leapfrog...", vpm.rungekutta3, vpm.UJ_fmm, 1),
                                          )

    println("\n"^2*description)

    @testset begin

        verbose1 = false
        verbose2 = true
        global this_is_a_test = true

        examples_path = joinpath(dirname(pathof(FLOWVPM)), "..", "examples", "vortexrings")
        include(joinpath(examples_path, "vortexrings.jl"))

        # -------------- SIMULATION PARAMETERS -------------------------------------
        nsteps    = 350                         # Number of time steps
        Rtot      = nsteps/100                  # (m) run simulation for equivalent
                                                #     time to this many radii
        nrings    = 2                           # Number of rings
        dZ        = 0.7906                      # (m) spacing between rings
        circulations = 1.0*ones(nrings)         # (m^2/s) circulation of each ring
        Rs        = 0.7906*ones(nrings)         # (m) radius of each ring
        ARs       = 1.0*ones(nrings)            # Aspect ratio AR = a/r of each ring
        Rcrosss   = 0.10*Rs                     # (m) cross-sectional radii
        sigmas    = Rcrosss                     # Particle smoothing of each radius
        Nphis     = 100*ones(Int, nrings)       # Number of cross sections per ring
        ncs       = nc*ones(Int, nrings)        # Number layers per cross section
        extra_ncs = 0*ones(Int, nrings)         # Number of extra layers per cross section
        Os        = [[0, 0, dZ*(ri-1)] for ri in 1:nrings]  # Position of each ring
        Oaxiss    = [I for ri in 1:nrings]      # Orientation of each ring
        nref      = 1                           # Reference ring

        beta      = 0.5                         # Parameter for theoretical velocity
        faux      = 1.0                         # Shrinks the discretized core by this factor

        Re        = 3000                        # Reynolds number Re = Gamma/nu

        # -------------- SOLVER SETTINGS -------------------------------------------
        solver = (
            formulation   = vpm.cVPM,
            SFS           = vpm.noSFS,
            relaxation    = vpm.correctedpedrizzetti,
            kernel        = vpm.winckelmans,
            viscous       = vpm.Inviscid(),
            transposed    = true,
            integration   = integration,
            UJ            = UJ,
            fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true)
        )


        # --------------- RUN SIMULATION -------------------------------------------
        println("\n"*"\t"^1*"Running simulation...")

        pfield = run_vortexring_simulation(  nrings, circulations,
                                            Rs, ARs, Rcrosss,
                                            Nphis, ncs, extra_ncs, sigmas,
                                            Os, Oaxiss;
                                            # ------- SIMULATION OPTIONS -----------
                                            Re=Re,
                                            nref=nref,
                                            nsteps=2,
                                            Rtot=Rtot,
                                            beta=beta,
                                            faux=faux,
                                            # ------- OUTPUT OPTIONS ---------------
                                            save_path=nothing,
                                            calc_monitors=false,
                                            verbose=verbose1, v_lvl=1,
                                            # verbose_nsteps=ceil(Int, nsteps/4),
                                            verbose_nsteps=100,
                                            pfieldargs=solver
                                            )
        t_elapsed = @elapsed pfield = run_vortexring_simulation(  nrings, circulations,
                                            Rs, ARs, Rcrosss,
                                            Nphis, ncs, extra_ncs, sigmas,
                                            Os, Oaxiss;
                                            # ------- SIMULATION OPTIONS -----------
                                            Re=Re,
                                            nref=nref,
                                            nsteps=nsteps,
                                            Rtot=Rtot,
                                            beta=beta,
                                            faux=faux,
                                            # ------- OUTPUT OPTIONS ---------------
                                            save_path="leapfrog/",
                                            calc_monitors=true,
                                            verbose=verbose1, v_lvl=1,
                                            # verbose_nsteps=ceil(Int, nsteps/4),
                                            verbose_nsteps=100,
                                            pfieldargs=solver
                                            )
        # --------------- COMPARE TO ANALYTIC SOLUTION -----------------------------



        # Calculate end state of simulated leapfrog
        tend = pfield.t                               # (s) simulation end time
        Z_vpm = [zeros(3) for ri in 1:nrings]         # Centroid position
        R_vpm, sgm_vpm = zeros(nrings), zeros(nrings) # Ring and cross section radii
        intervals = calc_ring_invervals(nrings, Nphis, ncs, extra_ncs)
        calc_rings_weighted!(Z_vpm, R_vpm, sgm_vpm, pfield, nrings, intervals)

        Z1_vpm, Z2_vpm = Z_vpm[1][3], Z_vpm[2][3]     # Centroid of rings
        R1_vpm, R2_vpm = R_vpm[1], R_vpm[2]           # Radius of rings


        # Solve analytic system of ODEs
        Zs = [Os[ri][3] for ri in 1:nrings]

        if solver[:kernel] == vpm.winckelmans
            Deltas = 0*ones(nrings)
        else
            error("Unknown kernel Delta!")
        end

        println("\n"*"\t"^1*"Computing analytic solution...")

        (ts_ana, Rs_ana,
          Zs_ana, as_ana) = analytic_coaxialrings(nrings, circulations, Rs, Zs,
                                                    Rcrosss, Deltas;
                                                    dynamica=false, tend=pfield.t)

        # Calculate analytic end state of leapfrog
        Z1_ana, Z2_ana = Zs_ana[1][end], Zs_ana[2][end]
        R1_ana, R2_ana = Rs_ana[1][end], Rs_ana[2][end]


        # Error
        Z1_err = (Z1_vpm - Z1_ana) / Z1_ana
        Z2_err = (Z2_vpm - Z2_ana) / Z2_ana
        R1_err = (R1_vpm - R1_ana) / R1_ana
        R2_err = (R2_vpm - R2_ana) / R2_ana

        if verbose2
            @printf "%sLeapfrog end state verification\n"           "\n"*"\t"^1
            @printf "%s%10.10s%10s%10s%10s%10s\n"                   "\t"^2 "" " Centroid 1" " Centroid 2" "Radius 1" "Radius 2"
            @printf "%s%10.10s%10.3f%10.3f%10.3f%10.3f\n"           "\t"^2 "Analytic" Z1_ana Z2_ana R1_ana R2_ana
            @printf "%s%10.10s%10.3f%10.3f%10.3f%10.3f\n"           "\t"^2 "VPM" Z1_vpm Z2_vpm R1_vpm R2_vpm
            @printf "%s%10.10s%9.3f﹪%9.3f﹪%8.3f﹪%8.3f﹪\n"         "\t"^2 "ERROR" Z1_err*100 Z2_err*100 R1_err*100 R2_err*100
            @printf "%sTime:\t\t\t\t%1.8f s\n"                       "\t"^2 t_elapsed
        end

        # Test result
        abs(Z1_err) < 0.03 && abs(Z2_err) < 0.03 && abs(R1_err) < 0.03 && abs(R2_err) < 0.03
    end
end
