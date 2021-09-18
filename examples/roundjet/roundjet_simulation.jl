#=##############################################################################
# DESCRIPTION
    Driver of round jet simulations.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################


function run_roundjet_simulation(pfield::vpm.ParticleField,
                                        # ------- SIMULATION PARAMETERS --------
                                        d::Real,        # (m) jet diameter
                                        U1::Real,       # (m/s) Inflow centerline velocity
                                        Vfreestream::Array{<:Real, 1}; # (m/s) freestream vector
                                        AR=1.0,         # Aspect ratio a/b of jet NOTE: DON'T USE ANYTHING BUT 1.0 WITH rbf==true, AS IT HAS BEEN HARDCODED TO BE ROUNDED
                                        O=zeros(3),     # Origin of jet
                                        Oaxis=Float64[i==j for i in 1:3, j in 1:3], # Orientation of jet
                                        thetaod=0.025,  # Ratio of inflow momentum thickness of shear layer to diameter, θ/d
                                        Vprofile=(r, d, theta) -> abs(r) < d/2 ? tanh( (d/2 - abs(r)) / (theta) ) : 0, # Velocity profile of at boundary condition
                                        Re=nothing,     # Reynolds number
                                        # ------- SOLVER OPTIONS ---------------
                                        steps_per_d=50, # Number of time steps for the centerline at U1 to travel one diameter
                                        d_travel_tot=10,# Run simulation for an equivalent of this many diameters
                                        dxotheta=1/4,   # Distance Δx between particles over momentum thickness θ
                                        overlap=2.4,    # Overlap between particles
                                        minWfraction=1e-2,  # Threshold at which not to add particles
                                        maxRoR=1.0,     # (m) maximum radial distance to discretize
                                        max_zsigma=12.0,# Maximum sigmas in z-direction to create annulis for defining BC
                                        rbf=true,       # If true, it runs an RBF interpolation to match the analytic vorticity with more accuracy
                                        rbf_optargs=[(:itmax,200), (:tol,2.5e-2), (:iterror,true), (:verbose,true), (:debug,false)],
                                        restart_file=nothing,
                                        restart_sigma=nothing,
                                        # ------- OUTPUT OPTIONS ---------------
                                        verbose=true,           # Enable verbose
                                        v_lvl=0,
                                        runtime_functions=[],   # Monitors and runtime functions
                                        run_name="roundjet",    # Prefix of outputfiles
                                        optargs...
                                        )

    # -------------- SIMULATION PARAMETERS -------------------------------------
    # Jet geometry
    R         = d/2                         # (m) jet radius
    Cline     = Oaxis[:, 3]                 # Centerline direction

    # Temporal discretization
    dt         = d/steps_per_d / U1         # (s) time step
    nsteps     = ceil(Int, d_travel_tot*d / U1 / dt) # Number of time steps

    # Spatial discretization
    maxR       = maxRoR*R
    dx         = dxotheta * thetaod * d     # (m) approximate distance between particles
    sigma      = overlap*dx                 # particle smoothing

    # Inflow jet velocity (not including freestream)
    Vjet(r) = U1*Vprofile(r, d, thetaod*d)

    # -------------- SIMULATION SETUP ------------------------------------------

    # Convert velocity profile into vorticity profile
    Vjet_wrap(X) = Vjet(X[1])
    g(X) = ForwardDiff.gradient(Vjet_wrap, X)
    dVdr(r) = g([r])[1]
    W(r) = -dVdr(r)

    # Brute-force find maximum vorticity in the region to discretize
    Wpeak = maximum([abs.(W.(r)) for r in range(-maxR, maxR, length=1000)])

    # Pre-calculations for discretization
    Nphi = ceil(Int, 2*pi*R/dx)             # Number of cross sections
    NR = ceil(Int, maxR/dx)                 # Number of radial sections (annuli)
    dr = maxR/NR                            # (m) actual radial distance between particles

    if verbose
        println("\t"^(v_lvl+1)*"Number of cross-sections Nphi:\t$(Nphi)")
        println("\t"^(v_lvl+1)*"Number of annuli NR:\t\t$(NR)")
    end

    V2 = dot(Vfreestream, Cline)     # Axial component of the freestream
    BCi = Int[]                            # Indices of particles that make up the boundary condition

    # Spatial discretization of the boundary condition
    for ri in 1:NR         # Iterate over annuli (NOTE: Here we assume that the profile is symmetric from -R to R)

        # Lower and upper bounds, and center of annulus
        rlo = dr*(ri-1)                     # Annulus lower bound
        rup = dr*ri                         # Annulus uper bound
        rc = (rlo+rup)/2                    # Annulus center

        Vc = V2 + Vjet(rc)                  # Velocity at center of annulus
        dz = Vc*dt                          # Distance traveled in one time step

        # Integrate vorticity radially over annulus segment
        # NOTE: This line implicitely assumes AR=1.0
        Wint, err = Cubature.hquadrature(W, rlo, rup; reltol=1e-8, abstol=0, maxevals=1000)

        circulation = Wint*dz + 1e-12       # Annulus circulation
        Wmean = Wint / (rup-rlo)            # Mean vorticity

        # Longitudinal area (in z-r plane) of annulus swept by the velocity
        # NOTE: This line implicitely assumes AR=1.0
        area = dz*(rup-rlo)

        Nz = ceil(Int, max_zsigma*sigma/dz)

        if abs(Wmean) / Wpeak >= minWfraction
            for zi in 0:Nz                  # Iterate over Z layers (time steps)

                org_np = vpm.get_np(pfield)

                this_O = O + zi*dz*Cline  # Center of this layer

                # Discretize the annulus into Nphi sections as particles
                addannulus(pfield, circulation,
                                        rc, AR,
                                        Nphi, sigma, area;
                                        O=this_O, Oaxis=Oaxis,
                                        verbose=verbose, v_lvl=v_lvl+1)

                # Save indices of particles at the boundary
                if zi==0
                    for pi in org_np+1:vpm.get_np(pfield)
                        push!(BCi, pi)
                    end
                end
            end
        end
    end

    if verbose
        println("\t"^v_lvl*"Initial boundary-condition particles:\t$(vpm.get_np(pfield))")
    end

    # Determine the vortex strength of the boundary condition through RBF on the target vorticity
    if rbf

        if verbose
            println("\t"^v_lvl*"RBF fit to the analytical profile...")
        end

        Wp = zeros(3)

        # Use analytic vorticity as target vorticity (stored under P.M[7:9])
        for P in vpm.iterator(pfield)

            # Radial position of this particle
            aux = dot(P.X, Cline)
            r = 0
            for i in 1:3; r += ( P.X[i] - aux*Cline[i] )^2; end;
            r = sqrt(r)

            # Vorticity at this particle
            Wp .= P.Gamma
            Wp .*= W(r) / norm(P.Gamma)

            for i in 1:3
                P.M[i+6] = Wp[i]

                # NOTE: If I don't take only half of the target vorticity
                #       the simulation ends up having double the velocity
                #       and vorticity once the jet is fully developed
                P.M[i+6] /= 2

            end
        end

        # RBF interpolation of the analytic distribution
        rbf_scheme = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; v_lvl=v_lvl+1, rbf_optargs...)
        vpm.rbf_conjugategradient(pfield, rbf_scheme)
    end

    # Save a copy of boundary-condition particles
    BC = [deepcopy(vpm.get_particle(pfield, pi)) for pi in BCi]

    # Remove all particles
    for pi in vpm.get_np(pfield):-1:1
        vpm.remove_particle(pfield, pi)
    end

    # Runtime function that adds particles imposing the boundary condition
    function boundary_condition(pfield, t, dt; optargs...)

        for P in BC
            vpm.add_particle(pfield, P)
        end

        return false
    end

    if verbose
        println("\t"^v_lvl*"Final boundary-condition particles:\t$(length(BC))")
    end

    # Simulation restart
    if restart_file != nothing
        # Read restart file, overwritting the particle field
        vpm.read!(pfield, restart_file; overwrite=true, load_time=true)

        if restart_sigma != nothing

            # Evaluate current vorticity field (gets stored under P.Jexa[1:3])
            vpm.zeta_fmm(pfield)

            # Resize particle cores and store target vorticity under P.M[7:9]
            for P in vpm.iterate(pfield)

                P.sigma[1] = restart_sigma

                for i in 1:3
                    P.M[i+6] = P.Jexa[i]
                end
            end

            # Calculate the new vortex strenghts through RBF
            rbf_scheme = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; v_lvl=v_lvl+1, rbf_optargs...)
            vpm.rbf_conjugategradient(pfield, rbf_scheme)

        end
    end

    # Overwrite kinematic viscosity with the requested Reynolds number
    if Re != nothing && vpm.isinviscid(pfield.viscous) == false

        # (m^2/s) kinematic viscosity. Note: Is this correct?
        pfield.viscous.nu = (U1 - V2) * d / Re

        if vpm.iscorespreading(pfield.viscous)
            # Set core size for reset after overgrowing
            pfield.viscous.sgm0 = sigma
        end
    end

    # --------------- DEFINE RUNTIME FUNCTION ----------------------------------
    this_runtime_functions = vcat(boundary_condition, runtime_functions)
    runtime_function(args...; optargs...) = !prod(!f(args...; optargs...) for f in this_runtime_functions)


    # -------------- SIMULATION ------------------------------------------------
    vpm.run_vpm!(pfield, dt, nsteps;    runtime_function=runtime_function,
                                        save_path=save_path,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        optargs...
                                        )
end



function run_roundjet_simulation(d::Real, U1::Real,
                                        U2::Real,           # (m/s) Coflow velocity
                                        args...;
                                        U2angle=[0,0,0],    # (deg) Coflow angle from centerline
                                        Oaxis=Float64[i==j for i in 1:3, j in 1:3],
                                        maxparticles::Int=Int(3e6), pfieldargs=(),
                                        optargs...) where {R<:Real}

    # Define freestream (coflow) velocity
    Vfreestream = U2 * gt.rotation_matrix2(U2angle...) * Oaxis[:, 3]
    Vinf(t)     = Vfreestream

    # Start particle field with the target maximum number of particles
    pfield = vpm.ParticleField(maxparticles; Uinf=Vinf, pfieldargs...)

    return run_roundjet_simulation(pfield, d, U1, Vfreestream, args...;
                                            Oaxis=Oaxis,
                                            optargs...
                                            )
end
