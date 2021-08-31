#=##############################################################################
# DESCRIPTION
    Run the simulation of a leapfrogging ring.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################

include("vortexrings.jl")

save_path = "leapfrog_simulation00"     # Simulation gets saved in this folder

verbose1  = true
verbose2  = true
display_plots = true

# -------------- SIMULATION PARAMETERS -------------------------------------
nsteps    = 700                         # Number of time steps
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
    sgsmodel      = vpm.sgs_none,
    relaxation    = vpm.correctedpedrizzetti,
    kernel        = vpm.winckelmans,
    viscous       = vpm.Inviscid(),
    transposed    = true,
    integration   = vpm.rungekutta3,
    UJ            = vpm.UJ_fmm,
    fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5)
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


# Solve analytic system of ODEs
Zs = [Os[ri][3] for ri in 1:nrings]

if solver[:kernel] == vpm.winckelmans
    Deltas = 0*ones(nrings)
else
    error("Unknown kernel Delta!")
end

println("\nComputing analytic solution...")

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
end



# --------------- PLOT RESULTING LEAPFROG DYNAMICS -------------------------
read_path     = save_path                     # Path of  to read

if display_plots
    fig, axs = plot_dynamics1n2(read_path;
                                filename="vortexring-dynamics2.csv",
                                to_plot=[#  ((label_x, index_x, scale_x), (label_y, index_y, scale_y))
                                            (("Time (s)", "t", 1.0), (L"Ring radius $R$ (m)", 4, 1.0))
                                            (("Time (s)", "t", 1.0), (L"Ring centroid $Z$ (m)", 3, 1.0))
                                            # (("Time (s)", "t", 1.0), (L"Cross-sectional radius $a$ (m)", 5, 1.0))
                                            ((L"Ring centroid $Z/R_0$", 3, 1/Rs[1]), (L"Ring radius $R/R_0$", 4, 1/Rs[1]))
                                        ],
                                plot_ana=true,
                                ana_args=[nrings, circulations, Rs, Zs, Rcrosss, Deltas],
                                ana_optargs=[(:dynamica, false), (:tend, tend*4/5)],
                                sidelegend=true
                                )
end
