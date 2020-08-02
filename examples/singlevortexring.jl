#=##############################################################################
# DESCRIPTION
    Simulation of a single vortex ring.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

import FLOWVPM
vpm = FLOWVPM

include("vortexrings_functions.jl")



"""
Validation on a vortex ring compared with the analytical self-induced velocity
in Sullivan's *Dynamics of thin vortex rings*. Beta factor was extracted from
Berdowski's thesis "3D Lagrangian VPM-FMM for Modeling the Near-wake of a HAWT".
"""
function validation_singlevortexring(;
                                        # kernel=kernel_gaus, UJ=UJ_fmm,
                                        integration="rk",
                                        # fmm = FMM(; p=4, ncrit=10, theta=0.4, phi=0.5),
                                        Re=400,
                                        save_path="temps/val_vortexring00/",
                                        run_name="vortexring",
                                        paraview=true, prompt=true,
                                        verbose=true, verbose2=true,
                                        tol=1e-2, disp_plot=true,
                                        nc=1, Nphi=200, extra_nc=0,
                                        nsteps=200, coR=0.15, nR=5, faux1=1.0,
                                        R=1.0)

    # -------------- PARAMETERS ----------------------------------------------

    Gamma = 1.0               # Circulation
    # R = 1.0                   # Ring radius
    # coR = 0.15                # (core/R) ring cross section core size (radius)

    # nR = 5                  # Number of R distances to solve for
    # nsteps = 200            # Number of time steps

    # fmm = FMM(; p=4, ncrit=10, theta=0.4, phi=0.5)

    # res_Ucore, err = run_singlevortexring(R, Gamma, coR, Nphi, nc, Re, nsteps, nR;
    run_singlevortexring(R, Gamma, coR, Nphi, nc, Re, nsteps, nR;
                                extra_nc=extra_nc,
                                faux1=faux1,
                                # SIMULATION SETUP
                                # kernel=kernel,
                                # UJ=UJ,
                                # fmm=fmm,
                                # NUMERICAL SCHEMES
                                transposed=true,
                                relax=true,
                                rlxf=0.3,
                                integration=integration,
                                nsteps_relax=1,
                                beta_cs=1.25,
                                # SIMULATION OPTIONS
                                save_path=save_path,
                                run_name=run_name,
                                paraview=paraview, prompt=prompt,
                                verbose=verbose, verbose2=verbose2,
                                disp_plot=disp_plot
                                )


    # res = err <= tol         # Result
    # return res
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
                              # SIMULATION SETUP
                              # kernel::Kernel=kernel_gaus,
                              # UJ::Function=UJ_fmm,
                              # fmm::FMM=FMM(; p=4, ncrit=10, theta=0.4, phi=0.5),
                              # NUMERICAL SCHEMES
                              transposed=true,
                              relax=true,
                              rlxf=0.3,
                              integration="euler",
                              nsteps_relax=1,
                              beta_cs=1.25,
                              # SIMULATION OPTIONS
                              save_path="temps/vortexring00",
                              run_name="vortexring",
                              verbose_nsteps=10,
                              paraview=true, prompt=true,
                              verbose=true, verbose2=true,
                              disp_plot=true, plot_ana=true,
                              outs=[]
                              )

    # Additional parameters
    sigma = R*coR                 # Smoothing radius
    rmax = faux1*sigma            # Cross section's radius.
    lambda = sigma/(2*pi*R/Nphi)  # Smoothing radius overlap
    smoothdeg = 180/pi*(2*atan(sigma/2,R)) # (deg) Ring's angle covered by sigma
    sgmoR = sigma/R
    nu = override_nu==nothing ? Gamma/Re : override_nu

    beta = 0.5
    Ucore = Gamma/(4*pi*R)*(log(8/coR)-beta)  # Analytical self-induced velocity

    dt = nR*R/Ucore/nsteps        # Time step size

    if verbose2
        println("Ring's core / R:\t\t$(round(typeof(coR)!=Float64 ? coR.value : coR, digits=3))")
        println("Geometric core / R:\t\t$(round(typeof(rmax/R)!=Float64 ? (rmax/R).value : rmax/R, digits=3))")
        println("Smoothing radius / R:\t\t$(round(typeof(sigma/R)!=Float64 ? (sigma/R).value : sigma/R, digits=3))")
        println("Smoothing overlap sigma/h:\t$(round(typeof(lambda)!=Float64 ? (lambda).value : lambda, digits=3))")
        println("Smoothing ring angle:\t\t$(round(typeof(smoothdeg)!=Float64 ? (smoothdeg).value : smoothdeg, digits=3)) deg")
        println("dt:\t\t\t\t$(typeof(dt)!=Float64 ? (dt).value : dt)")
    end


    # -------------- PARTICLE FIELD-----------------------------------------------
    # Creates the field
    maxparticles = 10000
    pfield = vpm.ParticleField(maxparticles)
    # pfield = ParticleField(nu, kernel, UJ;
    #                         transposed=transposed,
    #                         relax=relax,
    #                         rlxf=rlxf,
    #                         integration=integration,
    #                         fmm=fmm
    #                         )
    sgm0 = sigma                            # Default core size
    # beta_cs = 1.25                          # Maximum core size growth

    # Adds the ring to the field
    addvortexring(pfield, Gamma, R, Nphi, nc, rmax;
                  extra_nc=extra_nc, lambda=lambda)


    vpm.save(pfield, run_name; path=save_path, createpath=true)


    # -------------- RUNTIME FUNCTION --------------------------------------------
    # Function for calculating center's position at each time step
    Xs, ts = [], []
    function center_position(pfield::vpm.ParticleField, t, dt)
        Np = vpm.get_np(pfield)
        X = sum([vpm.get_X(pfield, pi) for pi in 1:Np])
        X = X/Np
        push!(Xs, X)
        push!(ts, t)
        return false
    end

  # # -------------- SIMULATION --------------------------------------------------
  # # Runs the simulation
  # run_vpm!(pfield, dt, nsteps;  runtime_function=center_position,
  #                               nsteps_relax=nsteps_relax,
  #                               beta=beta_cs, sgm0=sgm0,
  #                               save_path=save_path,
  #                               run_name=run_name,
  #                               prompt=prompt,
  #                               verbose=verbose,
  #                               verbose_nsteps=verbose_nsteps
  #                               )
  #
  #
  #
  # # -------------- POST-PROCESSING ---------------------------------------------
  # # Resulting self-induced velocity
  # res_Ucore = norm(Xs[end] - Xs[1]) / (ts[end] - ts[1])
  # err = abs( (res_Ucore - Ucore) / Ucore )          # Error to analytical
  #
  # # Comparison to analytical solution
  # if verbose2
  #   println("Vortex ring self-induced velocity verification")
  #   println("\tAnalytical velocity: \t$(round(typeof(Ucore)!=Float64 ? Ucore.value : Ucore, digits=10))")
  #   println("\tResulting velocity: \t$(round(typeof(res_Ucore)!=Float64 ? res_Ucore.value : res_Ucore, digits=10))")
  #   println("\tError: \t\t\t$(round(typeof(err)!=Float64 ? (100*err).value : (100*err), digits=10)) %\n")
  # end
  #
  # # Plots velocity
  # if disp_plot
  #
  #   Xzs = [!(typeof(X[3]) in [Float64, Int64]) ? X[3].value : X[3] for X in Xs]
  #   plt_Xzs = [!(typeof(Xzs[i]) in [Float64, Int64]) ? Xzs[i].value : Xzs[i] for i in 1:10:size(ts)[1]]
  #   plt_ts = [!(typeof(ts[i]) in [Float64, Int64]) ? ts[i].value : ts[i] for i in 1:10:size(ts)[1]]
  #   plot(plt_ts, plt_Xzs, "or", label="ADVPM", alpha=0.5)
  #   if plot_ana; plot(ts, Ucore*ts, "k", label="Analytical Inviscid", alpha=0.9); end;
  #   legend(loc="best")
  #   xlabel("Time (s)")
  #   ylabel("Ring's core position (m)")
  #   grid(true, color="0.8", linestyle="--")
  #   title("Single vortex ring validation")
  #
  #   if save_path!=nothing
  #     savefig(joinpath(save_path, "vring.png"))
  #   end
  # end
  #
  # # --------------- VISUALIZATION --------------------------------------------
  # if save_path!=nothing
  #   if paraview
  #     println("Calling Paraview...")
  #     strn = ""
  #     strn = strn * run_name * "...xmf;"
  #
  #     run(`paraview --data="$(joinpath(save_path,strn))"`)
  #   end
  # end
  #
  # push!(outs, Xs)
  # push!(outs, ts)
  #
  # res_Ucore, err
end
