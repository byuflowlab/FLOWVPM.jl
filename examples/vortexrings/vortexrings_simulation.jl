#=##############################################################################
# DESCRIPTION
    Driver of vortex ring simulations.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################


"""
Runs the simulation for a total amount of time equivalent to how long the
reference ring `nref` would take to travel a distance of `Rtot` radii in
isolation and inviscid flow (calculated through the function `Uring(...)`).
The time step `dt` is then calculated as `dt = (Rtot/Uring) / nsteps`
"""
function run_vortexring_simulation(
        pfield::vpm.ParticleField{R, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any},
                                        nsteps::Int,
                                        dt,
                                        nrings::Int,
                                        Nphis, ncs, extra_ncs;
                                        # ------- SIMULATION OPTIONS -----------
                                        runtime_function=(args...; optargs...)->false,
                                        # ------- OUTPUT OPTIONS ---------------
                                        save_path=nothing,      # Where to save the simulation
                                        run_name="vortexring",  # File prefix
                                        prompt=true,            # Whether to prompt the user
                                        verbose=true,           # Enable verbose
                                        v_lvl=0,
                                        verbose_nsteps=100,
                                        calc_monitors=true,
                                        mon_enstrophy=vpm.monitor_enstrophy,
                                        monitor_others=(args...; optargs...) -> false,
                                        ringmon_optargs=[],
                                        optargs...
                                        ) where R


    # -------------- SETUP -----------------------------------------------------
    if save_path != nothing
        vpm.create_path(save_path, prompt)
    end

    # Generate monitors
    if calc_monitors
        monitor_enstrophy_this(args...; optargs...) = mon_enstrophy(args...; save_path=save_path, optargs...)
        monitor_vortexring = generate_monitor_vortexring(nrings, Nphis, ncs, extra_ncs; TF=R, save_path=save_path,
                                                                        fname_pref=run_name, ringmon_optargs...)
    end

    function monitors(args...; optargs...)
        if calc_monitors
            return monitor_enstrophy_this(args...; optargs...) || monitor_vortexring(args...; optargs...) || monitor_others(args...; optargs...)
        else
            return false
        end
    end

    # Define runtime function
    this_runtime_function(args...; optargs...) = monitors(args...; optargs...) || runtime_function(args...; optargs...)

    # -------------- SIMULATION ------------------------------------------------
    vpm.run_vpm!(pfield, dt, nsteps;    runtime_function=this_runtime_function,
                                        save_path=save_path,
                                        create_savepath=false,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        verbose_nsteps=verbose_nsteps,
                                        optargs...
                                        )


    return pfield
end



function run_vortexring_simulation(
        pfield::vpm.ParticleField{TF,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
                                        nrings, circulations,
                                        Rs, ARs, Rcrosss,
                                        Nphis, ncs, extra_ncs, sigmas,
                                        Os, Oaxiss;
                                        # ------- SIMULATION OPTIONS -----------
                                        nref=1,         # Reference ring
                                        nsteps=1000,    # Number of time steps in simulation
                                        Rtot=10.0,      # Runs the simulation for this long (in radii distances)
                                        beta=0.5,       # Parameter for theoretical velocity
                                        faux=1.0,       # Shrinks the discretized core by this factor
                                        rbf=false,      # If true, it runs an RBF interpolation to match the analytic vorticity
                                        rbf_optargs=[(:itmax,200), (:tol,1e-2), (:iterror,true), (:verbose,true), (:debug,false)],
                                        zeta=(r,Rcross) -> 1/(pi*Rcross^2) * exp(-r^2/Rcross^2), # Analytic vorticity distribution (used in RBF)
                                        minWfraction=0, # Removes any particles with less vorticity than this fraction of the peak vorticity
                                        restart_file=nothing,
                                        restart_sigma=nothing,
                                        # ------- OUTPUT OPTIONS ---------------
                                        verbose=true,           # Enable verbose
                                        v_lvl=0,
                                        use_monitor_ringvorticity=false,
                                        monvort_optargs=[(:nprobes, 1000)],
                                        monitor_others=(args...; optargs...) -> false,
                                        optargs...
                                        ) where TF


    # -------------- SETUP -----------------------------------------------------

    Uref = Uring(circulations[nref], Rs[nref], Rcrosss[nref], beta) # (m/s) reference velocity
    dt = (Rtot/Uref) / nsteps         # (s) time step



    # Add vortex rings to particle field
    for ri in 1:nrings
        addvortexring(pfield, circulations[ri],
                        Rs[ri], ARs[ri], faux*Rcrosss[ri],
                        Nphis[ri], ncs[ri], sigmas[ri]; extra_nc=extra_ncs[ri],
                        O=Os[ri], Oaxis=Oaxiss[ri],
                        verbose=verbose, v_lvl=v_lvl
                      )
    end

    if restart_file != nothing
        # Read restart file, overwritting the particle field
        vpm.read!(pfield, restart_file; overwrite=true, load_time=true)

        if restart_sigma != nothing

            # Evaluate current vorticity field (gets stored under get_J(P)[1:3])
            vpm.zeta_fmm(pfield)

            # Resize particle cores and store target vorticity under P.M[7:9]
            for P in vpm.iterate(pfield)

                P.sigma[1] = restart_sigma

                for i in 1:3
                    P.M[i+6] = get_J(P)[i]
                end
            end

            # Calculate the new vortex strenghts through RBF
            viscous = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; v_lvl=v_lvl+1, rbf_optargs...)
            vpm.rbf_conjugategradient(pfield, viscous)

        end

    elseif rbf
        # Generate analytic vorticity field
        W_fun! = generate_Wfun(nrings, circulations,
                                    Rs, ARs, Rcrosss, Os, Oaxiss; zeta=zeta)
        W = zeros(3)

        # Remove particles at positions where the vorticity is negligible
        if minWfraction > 0

            peakW = maximum(circulations[ri]*zeta(0, Rcrosss[ri]) for ri in 1:nrings)

            for Pi in vpm.get_np(pfield):-1:1
                P = vpm.get_particle(pfield, Pi)

                W .= 0
                W_fun!(W, P.X)
                magW = norm(W)

                if magW/peakW <= minWfraction
                    vpm.remove_particle(pfield, Pi)
                end
            end
        end

        if verbose
            println("\t"^(v_lvl)*"Total number of particles: $(vpm.get_np(pfield))")
        end

        # Use analytic vorticity as target vorticity (stored under P.M[7:9])
        for P in vpm.iterator(pfield)
            W .= 0
            W_fun!(W, P.X)
            for i in 1:3
                P.M[i+6] = W[i]
            end
        end

        # RBF interpolation of the analytic distribution
        viscous = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; v_lvl=v_lvl+1, rbf_optargs...)
        vpm.rbf_conjugategradient(pfield, viscous)
    end

    if verbose
        @printf "%sReference ring: %i\n"                              "\t"^v_lvl nref
        @printf "%sGeometric Thickness Rcross/R:\t\t%1.3f\n"          "\t"^(v_lvl+1) Rcrosss[nref]/Rs[nref]
        @printf "%sSmoothing thickness sigma/R:\t\t%1.3f\n"           "\t"^(v_lvl+1) sigmas[nref]/Rs[nref]
        @printf "%sSmoothing overlap sigma/(2*pi*R/Nphi):\t%1.3f\n"   "\t"^(v_lvl+1) sigmas[nref]/(2*pi*Rs[nref]/Nphis[nref])
        @printf "%sRing angle covered by sigma:\t\t%1.3f°\n"          "\t"^(v_lvl+1) 180/pi*(2*atan(sigmas[nref]/2,Rs[nref]))
        @printf "%sTime step:\t\t\t\t%1.5e s\n"                       "\t"^(v_lvl+1) dt
    end

    monitor_ringvorticity = !use_monitor_ringvorticity ? (args...; optargs...) -> false :
                                generate_monitor_ringvorticity(nrings, Nphis,
                                                            ncs, extra_ncs, TF=TF;
                                                            save_path=save_path,
                                                            monvort_optargs...)

    this_monitor_others(args...; optargs...) = monitor_others(args...; optargs...) || monitor_ringvorticity(args...; optargs...)

    return run_vortexring_simulation(pfield, nsteps, dt,
                                            nrings, Nphis, ncs, extra_ncs;
                                            verbose=verbose,
                                            v_lvl=v_lvl,
                                            monitor_others=this_monitor_others,
                                            optargs...
                                            )
end



function run_vortexring_simulation(nrings::Int, circulations,
                                        Rs, ARs, Rcrosss,
                                        Nphis, ncs, extra_ncs, args...;
                                        maxparticles="automatic", pfieldargs=(),
                                        nref=1, Re=nothing, R=Float64, optargs...)

    if maxparticles == "automatic"
        maxp = sum( ri -> number_particles(Nphis[ri], ncs[ri]; extra_nc=extra_ncs[ri]), 1:nrings)
    else
        maxp = maxparticles
    end

    # Start particle field with the target maximum number of particles
    pfield = vpm.ParticleField(maxp, R; pfieldargs...)

    # Overwrite kinematic viscosity with the requested Reynolds number
    if Re != nothing && vpm.isinviscid(pfield.viscous) == false
        nu = circulations[nref]/Re
        pfield.viscous.nu = nu
    end

    return run_vortexring_simulation(pfield, nrings, circulations,
                                            Rs, ARs, Rcrosss,
                                            Nphis, ncs, extra_ncs, args...;
                                            nref=nref, optargs...
                                            )
end
