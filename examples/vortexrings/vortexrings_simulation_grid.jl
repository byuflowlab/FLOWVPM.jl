#=##############################################################################
# DESCRIPTION
    Driver of vortex ring simulations initiated from a grid.

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
function run_vortexring_grid_simulation(pfield::vpm.ParticleField,
                                        nrings, circulations,
                                        Rs, ARs, Rcrosss,
                                        dxoRcrosss, sigmas, minWfraction::Real,
                                        Os, Oaxiss;
                                        # ------- SIMULATION OPTIONS -----------
                                        nref=1,         # Reference ring
                                        nsteps=1000,    # Number of time steps in simulation
                                        Rtot=10.0,      # Runs the simulation for this long (in radii distances)
                                        beta=0.5,       # Parameter for theoretical velocity
                                        faux=1.0,       # Shrinks the discretized core by this factor
                                        # ------- OUTPUT OPTIONS ---------------
                                        save_path=save_path,
                                        verbose=true,           # Enable verbose
                                        v_lvl=0,
                                        addringoptargs=[],
                                        use_monitor_ringvorticity=false,
                                        monvort_optargs=[(:nprobes, 1000)],
                                        monitor_others=(args...; optargs...) -> false,
                                        optargs...
                                        )


    # -------------- SETUP -----------------------------------------------------

    Uref = Uring(circulations[nref], Rs[nref], Rcrosss[nref], beta) # (m/s) reference velocity
    dt = (Rtot/Uref) / nsteps         # (s) time step

    Nphis = []

    # Add vortex rings to particle field
    for ri in 1:nrings

        dx = dxoRcrosss[ri]*Rcrosss[ri]
        minmagGamma = minWfraction * circulations[ri]/(pi*Rcrosss[ri]^2)*dx^3

        Nphi = addvortexring(pfield, circulations[ri],
                                Rs[ri], ARs[ri], faux*Rcrosss[ri],
                                dxoRcrosss[ri], sigmas[ri], minmagGamma,
                                O=Os[ri],
                                Oaxis=Oaxiss[ri];
                                addringoptargs...
                              )
        push!(Nphis, Nphi)
    end

    if verbose
        @printf "%sReference ring: %i\n"                              "\t"^v_lvl nref
        @printf "%sGeometric Thickness Rcross/R:\t\t%1.3f\n"          "\t"^(v_lvl+1) Rcrosss[nref]/Rs[nref]
        @printf "%sSmoothing thickness sigma/R:\t\t%1.3f\n"           "\t"^(v_lvl+1) sigmas[nref]/Rs[nref]
        @printf "%sSmoothing overlap sigma/dx:\t\t%1.3f\n"            "\t"^(v_lvl+1) sigmas[nref]/(dxoRcrosss[nref]*Rcrosss[nref])
        @printf "%sRing angle covered by sigma:\t\t%1.3f°\n"          "\t"^(v_lvl+1) 180/pi*(2*atan(sigmas[nref]/2,Rs[nref]))
        @printf "%sTime step:\t\t\t\t%1.5e s\n"                       "\t"^(v_lvl+1) dt
    end


    monitor_ringvorticity = !use_monitor_ringvorticity ? (args...; optargs...) -> false :
                                generate_monitor_ringvorticity(nrings, Nphis,
                                                            zeros(nrings), zeros(nrings);
                                                            save_path=save_path,
                                                            monvort_optargs...)

    this_monitor_others(args...; optargs...) = monitor_others(args...; optargs...) || monitor_ringvorticity(args...; optargs...)

    return run_vortexring_simulation(pfield, dt,
                                            nrings, circulations,
                                            Rs, ARs, Rcrosss,
                                            Nphis, zeros(nrings), zeros(nrings), sigmas,
                                            Os, Oaxiss;
                                            save_path=save_path,
                                            verbose=verbose,
                                            v_lvl=v_lvl,
                                            monitor_others=this_monitor_others,
                                            optargs...
                                            )
end


function run_vortexring_grid_simulation(nrings::Int, circulations, args...;
                                        maxparticles=Int(2e6), pfieldargs=(),
                                        nref=1, Re=nothing, optargs...)
    maxp = maxparticles

    # Start particle field with the target maximum number of particles
    pfield = vpm.ParticleField(maxp; pfieldargs...)

    # Overwrite kinematic viscosity with the requested Reynolds number
    if Re != nothing && vpm.isinviscid(pfield.viscous) == false
        nu = circulations[nref]/Re
        pfield.viscous.nu = nu
    end

    return run_vortexring_grid_simulation(pfield, nrings, circulations, args...;
                                                           nref=nref, optargs...
                                         )
end
