#=##############################################################################
# DESCRIPTION
    Simulation of a single vortex ring.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

try
    # If this variable exist, we know we are running this as a unit test
    this_is_a_test
catch e
    using PyPlot
end

import FLOWVPM
vpm = FLOWVPM

include("vortexrings_functions.jl")



"""
Validation on a vortex ring compared with the analytical self-induced velocity
in Sullivan's *Dynamics of thin vortex rings*. Beta factor was extracted from
Berdowski's thesis "3D Lagrangian VPM-FMM for Modeling the Near-wake of a HAWT".
"""
function validation_singlevortexring(;
                                        kernel=vpm.winckelmans, UJ=vpm.UJ_fmm,
                                        integration=vpm.rungekutta3,
                                        fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5),
                                        Re=400, viscous=vpm.Inviscid(),
                                        save_path="temps/val_vortexring00/",
                                        tol=1e-2,
                                        nc=1, Nphi=200, extra_nc=0,
                                        nsteps=1000, coR=0.15, nR=5,
                                        R=1.0,
                                        overwrite_faux1=nothing,
                                        # kernel=vpm.gaussianerf, UJ=vpm.UJ_fmm,
                                        # integration=vpm.euler,
                                        # fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.3),
                                        # maxparticles=30000, nc=2,  extra_nc=3, nsteps=200, nR=1,
                                        # Re=400, viscous=vpm.CoreSpreading(0, 0, vpm.zeta_fmm; beta=1.01, itmax=20, tol=1e-1,
                                        #                                     iterror=false, verbose=true, debug=false),
                                        optargs...)

    # -------------- PARAMETERS ----------------------------------------------

    Gamma = 1.0
    # faux1 = nc==0 ? extra_nc==0 ? 1 : 0.1 : nc==1 ? 0.25 : 0.5
    faux1 = nc==0 ? extra_nc==0 ? 1 : 0.25 : nc==1 ? 0.25 : 0.5

    faux1 = overwrite_faux1 != nothing ? overwrite_faux1 : faux1

    res_Ucore, err = run_singlevortexring(R, Gamma, coR, Nphi, nc, Re, nsteps, nR;
                                extra_nc=extra_nc,
                                faux1=faux1,
                                # SIMULATION SETUP
                                kernel=kernel,
                                UJ=UJ,
                                fmm=fmm,
                                # NUMERICAL SCHEMES
                                transposed=true,
                                integration=integration,
                                nsteps_relax=1,
                                viscous=viscous,
                                # SIMULATION OPTIONS
                                save_path=save_path,
                                optargs...
                                )


    res = err <= tol         # Result
    return res
end



"""
**ARGUMENTS**
* `R::Real`                 : Ring radius.
* `Gamma::Real`             : Ring circulation.
* `coR::Real`               : (core/R) ring cross section core radius.
* `Nphi::Int64`             : Ring discretization (number of particles).
* `Nc::Int64`               : Number of layers of particles discretizing the
                            cross section.
* `Re::Real`                : Reynolds number Re=Gamma/nu.
* `nsteps::Int64`           : Number of time steps in simulation.
* `nR::Real`          : Number of radius distances to solve for.

**OPTIONAL ARGUMENTS**
* `extra_nc::Int64`         : Extra layer of zero-strength particles
                            surrounding the cross section.
"""
function run_singlevortexring(R::Real, Gamma::Real, coR::Real,
                              Nphi::Int64, nc::Int64, Re::Real,
                              nsteps::Int64, nR::Real;
                              extra_nc::Int64=0,
                              faux1=1.0,
                              override_nu::Union{Nothing, Real}=nothing,
                              maxparticles=10000,
                              # SIMULATION SETUP
                              kernel::vpm.Kernel=vpm.winckelmans,
                              UJ::Function=vpm.UJ_direct,
                              fmm=vpm.FMM(; p=4, ncrit=10, theta=0.4, phi=0.5),
                              # NUMERICAL SCHEMES
                              formulation=vpm.formulation_default,
                              transposed=true,
                              relax=true,
                              rlxf=0.3,
                              integration=vpm.rungekutta3,
                              nsteps_relax=1,
                              viscous=vpm.Inviscid(),
                              # SIMULATION OPTIONS
                              save_path="temps/vortexring00",
                              run_name="vortexring",
                              verbose_nsteps=10,
                              paraview=true, prompt=true,
                              verbose=true, verbose2=true, v_lvl=0,
                              disp_plot=true, plot_ana=true,
                              outs=[]
                              )

    # Additional parameters
    sigma = R*coR                 # Smoothing radius
    rmax = faux1*sigma            # Cross section's radius.
    lambda = sigma/(2*pi*R/Nphi)  # Smoothing radius overlap
    smoothdeg = 180/pi*(2*atan(sigma/2,R)) # (deg) Ring's angle covered by sigma
    sgmoR = sigma/R

    optargs_ring = []

    # Set up viscous scheme
    if vpm.isinviscid(viscous) == false
        viscous.nu = override_nu != nothing ? override_nu : Gamma/Re
        if vpm.iscorespreading(viscous)
            viscous.sgm0 = sigma

            # Tilt the ring slightly to not be aligned with the z-axis,
            # otherwise the RBF will fail trying to divide by zero
            tilt = 1*pi/180
            coord_sys = [cos(tilt) 0 sin(tilt); 0 1 0; -sin(tilt) 0 cos(tilt)]
            push!(optargs_ring, (:ring_coord_system, coord_sys))
        end
    end


    beta = 0.5
    Ucore = Gamma/(4*pi*R)*(log(8/coR)-beta)  # Analytical self-induced velocity

    dt = nR*R/Ucore/nsteps        # Time step size

    if verbose2
        println("\t"^v_lvl*"Ring's core / R:\t\t$(round(typeof(coR)!=Float64 ? coR.value : coR, digits=3))")
        println("\t"^v_lvl*"Geometric core / R:\t\t$(round(typeof(rmax/R)!=Float64 ? (rmax/R).value : rmax/R, digits=3))")
        println("\t"^v_lvl*"Smoothing radius / R:\t\t$(round(typeof(sigma/R)!=Float64 ? (sigma/R).value : sigma/R, digits=3))")
        println("\t"^v_lvl*"Smoothing overlap sigma/h:\t$(round(typeof(lambda)!=Float64 ? (lambda).value : lambda, digits=3))")
        println("\t"^v_lvl*"Smoothing ring angle:\t\t$(round(typeof(smoothdeg)!=Float64 ? (smoothdeg).value : smoothdeg, digits=3)) deg")
        println("\t"^v_lvl*"dt:\t\t\t\t$(typeof(dt)!=Float64 ? (dt).value : dt)")
    end


    # -------------- PARTICLE FIELD-----------------------------------------------
    # Creates the field
    pfield = vpm.ParticleField(maxparticles; viscous=viscous, kernel=kernel, UJ=UJ,
                                transposed=transposed,
                                relax=relax, rlxf=rlxf,
                                integration=integration,
                                fmm=fmm,
                                formulation=formulation
                                )

    # Adds the ring to the field
    addvortexring(pfield, Gamma, R, Nphi, nc, rmax;
                  extra_nc=extra_nc, lambda=lambda, optargs_ring...)


    # -------------- RUNTIME FUNCTION --------------------------------------------
    # Function for calculating center's position at each time step
    Xs, ts = [], []
    function center_position(pfield::vpm.ParticleField, t, dt; optargs...)
        Np = vpm.get_np(pfield)
        X = sum([P.X for P in vpm.iterator(pfield)])
        X = X/Np
        push!(Xs, X)
        push!(ts, t)
        return false
    end

    # -------------- SIMULATION --------------------------------------------------
    # Runs the simulation
    vpm.run_vpm!(pfield, dt, nsteps; runtime_function=center_position,
                                        nsteps_relax=nsteps_relax,
                                        save_path=save_path,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        verbose_nsteps=verbose_nsteps
                                        )



    # -------------- POST-PROCESSING ---------------------------------------------
    # Resulting self-induced velocity
    res_Ucore = norm(Xs[end] - Xs[1]) / (ts[end] - ts[1])
    err = abs( (res_Ucore - Ucore) / Ucore )          # Error to analytical

    # Comparison to analytical solution
    if verbose2
        println("\t"^v_lvl*"Vortex ring self-induced velocity verification")
        println("\t"^v_lvl*"\tAnalytical velocity: \t$(round(typeof(Ucore)!=Float64 ? Ucore.value : Ucore, digits=10))")
        println("\t"^v_lvl*"\tResulting velocity: \t$(round(typeof(res_Ucore)!=Float64 ? res_Ucore.value : res_Ucore, digits=10))")
        println("\t"^v_lvl*"\tError: \t\t\t$(round(typeof(err)!=Float64 ? (100*err).value : (100*err), digits=10)) %\n")
    end

    # Plots velocity
    if disp_plot

        Xzs = [!(typeof(X[3]) in [Float64, Float32, Int64]) ? X[3].value : X[3] for X in Xs]
        plt_Xzs = [!(typeof(Xzs[i]) in [Float64, Float32, Int64]) ? Xzs[i].value : Xzs[i] for i in 1:10:size(ts)[1]]
        plt_ts = [!(typeof(ts[i]) in [Float64, Float32, Int64]) ? ts[i].value : ts[i] for i in 1:10:size(ts)[1]]
        if plot_ana; plot(ts, Ucore*ts, "k", label="Analytical Inviscid", alpha=0.9); end;
        plot(plt_ts, plt_Xzs, "or", label="FLOWVPM", alpha=0.5)
        legend(loc="best", frameon=false)
        xlabel("Time (s)")
        ylabel("Centroid position (m)")
        grid(true, color="0.9", linestyle=":")
        title("Single vortex ring validation")

        if save_path!=nothing
            savefig(joinpath(save_path, "vring.png"), dpi=300)
        end
    end

    # --------------- VISUALIZATION --------------------------------------------
    if save_path!=nothing
        if paraview
            println("\t"^v_lvl*"Calling Paraview...")
            strn = ""
            strn = strn * run_name * "...xmf;"

            run(`paraview --data="$(joinpath(save_path,strn))"`)
        end
    end

    push!(outs, Xs)
    push!(outs, ts)

    res_Ucore, err
end
