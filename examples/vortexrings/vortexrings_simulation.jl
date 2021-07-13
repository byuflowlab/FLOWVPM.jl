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
function run_vortexring_simulation(pfield::vpm.ParticleField,
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
                                        ringmon_optargs=[],
                                        optargs...
                                        )


    # -------------- SETUP -----------------------------------------------------

    Uref = Uring(circulations[nref], Rs[nref], Rcrosss[nref], beta) # (m/s) reference velocity
    dt = (Rtot/Uref) / nsteps         # (s) time step



    # Add vortex rings to particle field
    for ri in 1:nrings
        addvortexring(pfield, circulations[ri],
                        Rs[ri], ARs[ri], faux*Rcrosss[ri],
                        Nphis[ri], ncs[ri], sigmas[ri]; extra_nc=extra_ncs[ri],
                        O=Os[ri],
                        Oaxis=Oaxiss[ri]
                      )
    end

    if verbose
        @printf "%sReference ring: %i\n"                              "\t"^v_lvl nref
        @printf "%sGeometric Thickness Rcross/R:\t\t%1.3f\n"          "\t"^(v_lvl+1) Rcrosss[nref]/Rs[nref]
        @printf "%sSmoothing thickness sigma/R:\t\t%1.3f\n"           "\t"^(v_lvl+1) sigmas[nref]/Rs[nref]
        @printf "%sSmoothing overlap sigma/(2*pi*R/Nphi):\t%1.3f\n"   "\t"^(v_lvl+1) sigmas[nref]/(2*pi*Rs[nref]/Nphis[nref])
        @printf "%sRing angle covered by sigma:\t\t%1.3f°\n"          "\t"^(v_lvl+1) 180/pi*(2*atan(sigmas[nref]/2,Rs[nref]))
        @printf "%sTime step:\t\t\t\t%1.5e s\n"                       "\t"^(v_lvl+1) dt
    end

    if save_path != nothing
        vpm.create_path(save_path, prompt)
    end


    # Generate monitors
    if calc_monitors
        monitor_enstrophy_this(args...; optargs...) = mon_enstrophy(args...; save_path=save_path, optargs...)
        monitor_vortexring = generate_monitor_vortexring(nrings, Nphis, ncs, extra_ncs; save_path=save_path,
                                                                        fname_pref=run_name, ringmon_optargs...)
    end

    function monitors(args...; optargs...)
        if calc_monitors
            return monitor_enstrophy_this(args...; optargs...) || monitor_vortexring(args...; optargs...)
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


function run_vortexring_simulation(nrings::Int, circulations,
                                        Rs, ARs, Rcrosss,
                                        Nphis, ncs, extra_ncs, args...;
                                        maxparticles="automatic", pfieldargs=(),
                                        nref=1, Re=nothing, optargs...)

    if maxparticles == "automatic"
        maxp = sum( ri -> number_particles(Nphis[ri], ncs[ri]; extra_nc=extra_ncs[ri]), 1:nrings)
    else
        maxp = maxparticles
    end

    # Start particle field with the target maximum number of particles
    pfield = vpm.ParticleField(maxp; pfieldargs...)

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
