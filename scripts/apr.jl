# activate test environment
if splitpath(Base.active_project())[end-1] == "FLOWVPM.jl"
    import TestEnv
    TestEnv.activate()
end
using Test
import Printf: @printf
import FLOWVPM
vpm = FLOWVPM

this_is_a_test = true
examples_path = joinpath(dirname(pathof(FLOWVPM)), "..", "examples", "vortexrings")
include(joinpath(examples_path, "vortexrings.jl"))

overlap = 0.3
R = 1.0
Nphi = 20
sgm0 = 2*pi*R/Nphi/2*(1+overlap)
nu = 1.48e-5
nc = 1

# -------------- SIMULATION PARAMETERS -------------------------------------
nsteps    = 200                         # Number of time steps
Rtot      = 5.0                         # (m) run simulation for equivalent
                                        #     time to this many radii
nrings    = 2                           # Number of rings
dZ        = 1.0                         # (m) spacing between rings
circulations = 10.0*ones(nrings)         # (m^2/s) circulation of each ring
circulations[2] *= -1.0
Rs        = R*ones(nrings)            # (m) radius of each ring
ARs       = 1.0*ones(nrings)            # Aspect ratio AR = a/r of each ring
Rcrosss   = 0.15*Rs                     # (m) cross-sectional radii
sigmas    = Rcrosss
# sigmas    .= sgm0                     # Particle smoothing of each radius
Nphis     = Nphi*ones(Int, nrings)       # Number of cross sections per ring
ncs       = nc*ones(Int, nrings)        # Number layers per cross section
extra_ncs = 0*ones(Int, nrings)         # Number of extra layers per cross section
Os        = [[0, 0, dZ*(ri-1)] for ri in 1:nrings]  # Position of each ring
Oaxiss    = [I for ri in 1:nrings]      # Orientation of each ring
nref      = 1                           # Reference ring

beta      = 0.5                         # Parameter for theoretical velocity
faux      = 0.25                        # Shrinks the discretized core by this factor

# -------------- SOLVER SETTINGS -------------------------------------------
solver = (
    formulation   = vpm.rVPM,
    SFS           = vpm.SFS_Cd_twolevel_nobackscatter,
    relaxation    = vpm.correctedpedrizzetti,
    kernel        = vpm.gaussianerf,
    viscous       = vpm.Inviscid(),
    transposed    = true,
    integration   = vpm.rungekutta3,
    UJ            = vpm.UJ_fmm,
    fmm           = vpm.FMM(; p=4, ncrit=50, theta=0.4, nonzero_sigma=true),
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
                                    save_path=joinpath(@__DIR__,"apr_tests"),
                                    calc_monitors=false,
                                    verbose=false, v_lvl=1,
                                    verbose_nsteps=ceil(Int, nsteps/4),
                                    pfieldargs=solver
                                    )