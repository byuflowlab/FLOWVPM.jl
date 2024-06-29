#=##############################################################################
# DESCRIPTION
Run the simulation of a leapfrogging ring.

# AUTHORSHIP
* Author    : Eduardo J. Alvarez
* Email     : Edo.AlvarezR@gmail.com
* Created   : Jul 2021
* Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################

this_is_a_test = false

include("vortexrings.jl")

function run_leapfrog(x::Vector{T}) where T
    radius = x[1]
    z = x[2]

    save_path = "leapfrog_simulation00"     # Simulation gets saved in this folder

    verbose1  = true

    # -------------- SIMULATION PARAMETERS -------------------------------------
    nsteps    = 7                           # Number of time steps
    Rtot      = nsteps/100                  # (m) run simulation for equivalent
    #     time to this many radii
    nrings    = 2                           # Number of rings
    dZ        = z#0.7906                      # (m) spacing between rings
    circulations = 1.0*ones(nrings)         # (m^2/s) circulation of each ring
    Rs        = radius*ones(nrings)         # (m) radius of each ring
    ARs       = 1.0*ones(nrings)            # Aspect ratio AR = a/r of each ring
    Rcrosss   = 0.10*Rs                     # (m) cross-sectional radii
    sigmas    = Rcrosss                     # Particle smoothing of each radius
    Nphis     = 100*ones(Int, nrings)       # Number of cross sections per ring
    ncs       = 1*ones(Int, nrings)         # Number layers per cross section
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
              integration   = vpm.rungekutta3,
              UJ            = vpm.UJ_fmm,
              fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true),
              useGPU        = true
             )


    # --------------- RUN SIMULATION -------------------------------------------
    println("\nRunning simulation...")

    pfield = run_vortexring_simulation(  nrings, circulations,
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
                                       save_path=save_path,
                                       calc_monitors=true,
                                       verbose=verbose1, v_lvl=1,
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

    return [Z1_vpm, Z2_vpm, R1_vpm, R2_vpm]
end
