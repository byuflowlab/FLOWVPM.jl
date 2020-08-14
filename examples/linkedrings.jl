#=##############################################################################
# DESCRIPTION
    Simulation of two linked vortex rings (a.k.a., knot case).

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

import FLOWVPM
vpm = FLOWVPM

include("vortexrings_functions.jl")

data_path = joinpath(splitdir(@__FILE__)[1], "data")*"/"


function validation_linkedrings(;   kernel=vpm.winckelmans, UJ=vpm.UJ_fmm,
                                    integration=vpm.rungekutta3,
                                    fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5),
                                    Re=31e3, viscous=vpm.Inviscid(),
                                    save_path="temps/val_linkedrings00/",
                                    nc=1, Nphi=100, extra_nc=0,
                                    nsteps=480, nR=2.13,
                                    optargs...)

    nu = 1.004e-6
    # faux1 = nc==0 ? extra_nc==0 ? 1.0 : 0.1 : nc==1 ? 0.25 : 0.5
    faux1 = nc==0 ? extra_nc==0 ? 1.0 : 0.1 : nc==1 ? 1.0 : 0.5

    R1, R2 = 26.7/1000, 26.7/1000
    Gamma1, Gamma2 = Re*nu, Re*nu
    coR1, coR2 = 0.075, 0.075

    sep = R1                    # (m) separation between rings (radius to radius)
    angle = 20*pi/180           # (rad) inclination of rings

    coord_sys1 = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)]
    coord_sys2 = [cos(-angle) 0 sin(-angle); 0 1 0; -sin(-angle) 0 cos(-angle)]

    C1 = [0.0, 0.0, 0.0]
    C2 = [0.0, sep, 0.0]

    run_linkedrings(R1, R2, Gamma1, Gamma2, coR1, coR2,
                                  Nphi, nc, Re,
                                  nsteps, nR;
                                  extra_nc=extra_nc,
                                  faux1=faux1,
                                  kernel=kernel,
                                  UJ=UJ,
                                  fmm=fmm,
                                  integration=integration,
                                  viscous=viscous,
                                  override_nu=nu,
                                  save_path=save_path,
                                  optargs_ring1=[(:C, C1), (:ring_coord_system, coord_sys1)],
                                  optargs_ring2=[(:C, C2), (:ring_coord_system, coord_sys2)],
                                  optargs...
                                  )

    return nothing
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
function run_linkedrings(R1::Real, R2::Real,
                              Gamma1::Real, Gamma2::Real,
                              coR1::Real, coR2::Real,
                              Nphi::Int64, nc::Int64, Re::Real,
                              nsteps::Int64, nR::Real;
                              extra_nc::Int64=0,
                              faux1=1.0,
                              override_nu::Union{Nothing, Real}=nothing,
                              optargs_ring1=(),
                              optargs_ring2=(),
                              # SIMULATION SETUP
                              kernel::vpm.Kernel=vpm.winckelmans,
                              UJ::Function=vpm.UJ_direct,
                              fmm=vpm.FMM(; p=4, ncrit=10, theta=0.4, phi=0.5),
                              # NUMERICAL SCHEMES
                              transposed=true,
                              relax=true,
                              rlxf=0.1,
                              integration=vpm.rungekutta3,
                              nsteps_relax=1,
                              viscous=vpm.Inviscid(),
                              # SIMULATION OPTIONS
                              save_path="temps/linkedrings00",
                              run_name="vortexring",
                              verbose_nsteps=10,
                              paraview=true, prompt=true,
                              verbose=true, verbose2=true, v_lvl=0,
                              disp_plot=true,
                              )

    # Additional parameters
    sigma1 = R1*coR1                 # Smoothing radius
    sigma2 = R2*coR2                 # Smoothing radius
    rmax1 = faux1*sigma1             # Cross section's radius.
    rmax2 = faux1*sigma2             # Cross section's radius.
    lambda1 = sigma1/(2*pi*R1/Nphi)  # Smoothing radius overlap
    lambda2 = sigma2/(2*pi*R2/Nphi)  # Smoothing radius overlap
    smoothdeg1 = 180/pi*(2*atan(sigma1/2,R1)) # (deg) Ring's angle covered by sigma
    smoothdeg2 = 180/pi*(2*atan(sigma2/2,R2)) # (deg) Ring's angle covered by sigma
    sgmoR1 = sigma1/R1
    sgmoR2 = sigma2/R2

    # Set up viscous scheme
    if vpm.isinviscid(viscous) == false
        viscous.nu = override_nu != nothing ? override_nu : Gamma1/Re
        if vpm.iscorespreading(viscous)
            viscous.sgm0 = sigma1
        end
    end

    beta = 0.5
    Ucore = Gamma1/(4*pi*R1)*(log(8/coR1)-beta)  # Analytical self-induced velocity
    dt = nR*R1/Ucore/nsteps        # Time step size

    if verbose2
        println("\t"^v_lvl*"Ring's core #1 / R:\t\t$(round(typeof(coR1)!=Float64 ? coR1.value : coR1, digits=3))")
        println("\t"^v_lvl*"Geometric core #1 / R1:\t\t$(round(typeof(rmax1/R1)!=Float64 ? (rmax1/R1).value : rmax1/R1, digits=3))")
        println("\t"^v_lvl*"Smoothing radius #1 / R1:\t$(round(typeof(sigma1/R1)!=Float64 ? (sigma1/R1).value : sigma1/R1, digits=3))")
        println("\t"^v_lvl*"Smoothing overlap sigma #1/h1:\t$(round(typeof(lambda1)!=Float64 ? (lambda1).value : lambda1, digits=3))")
        println("\t"^v_lvl*"Smoothing ring angle #1:\t$(round(typeof(smoothdeg1)!=Float64 ? (smoothdeg1).value : smoothdeg1, digits=3)) deg")
        println("\t"^v_lvl*"dt:\t\t\t\t$(typeof(dt)!=Float64 ? (dt).value : dt)")
    end


    # -------------- PARTICLE FIELD-----------------------------------------------
    # Creates the field
    maxparticles = 20000
    pfield = vpm.ParticleField(maxparticles; viscous=viscous,
                                kernel=kernel, UJ=UJ,
                                transposed=transposed,
                                relax=relax, rlxf=rlxf,
                                integration=integration,
                                fmm=fmm
                                )

    # Adds the rings to the field
    addvortexring(pfield, Gamma1, R1, Nphi, nc, rmax1;
                        extra_nc=extra_nc, lambda=lambda1, optargs_ring1...)
    addvortexring(pfield, Gamma2, R2, Nphi, nc, rmax2;
                        extra_nc=extra_nc, lambda=lambda2, optargs_ring2...)


    # -------------- RUNTIME FUNCTION --------------------------------------------
    runtime_function(args...) = false

    # -------------- SIMULATION --------------------------------------------------
    # Runs the simulation
    vpm.run_vpm!(pfield, dt, nsteps; runtime_function=runtime_function,
                                        nsteps_relax=nsteps_relax,
                                        save_path=save_path,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        verbose_nsteps=verbose_nsteps
                                        )



    # -------------- POST-PROCESSING ---------------------------------------------
    nothing

    # --------------- VISUALIZATION --------------------------------------------
    if save_path!=nothing
        if paraview
            println("\t"^v_lvl*"Calling Paraview...")
            strn = ""
            strn = strn * run_name * "...xmf;"

            run(`paraview --data="$(joinpath(save_path,strn))"`)
        end
    end

    return nothing
end
