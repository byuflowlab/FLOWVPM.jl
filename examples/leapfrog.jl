#=##############################################################################
# DESCRIPTION
    Simulation of two leapfrogging vortex rings.

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
    import CSV
    using PyPlot
end

import FLOWVPM
vpm = FLOWVPM

include("vortexrings_functions.jl")

data_path = joinpath(splitdir(@__FILE__)[1], "data")*"/"

"""
Validation on vortex strenching by simulating the interaction between coaxial
vortex rings (leapfrog). See Berdowski's *3D Lagrangian VPM-FMM for Modeling the
 Near-wake of a HAWT*, Sec. 6.2.
"""
function validation_leapfrog(;  kernel=vpm.kernel_wnklmns, UJ=vpm.UJ_fmm,
                                integration=vpm.rungekutta3,
                                fmm=vpm.FMM(; p=4, ncrit=50, theta=0.4, phi=0.5),
                                Re=400, viscous=false,
                                save_path="temps/val_leapfrog03/",
                                tol=1e-2,
                                nc=1, Nphi=100, extra_nc=0,
                                nsteps=200*6, nR=30, faux1=1.0,
                                optargs...)

    R1, R2 = 0.5, 1.0
    Gamma1, Gamma2 = 1.0, 1.0
    coR1, coR2 = 0.2, 0.1
    run_leapfrog(R1, R2, Gamma1, Gamma2, coR1, coR2,
                                  Nphi, nc, Re,
                                  nsteps, nR;
                                  extra_nc=extra_nc,
                                  faux1=faux1,
                                  kernel=kernel,
                                  UJ=UJ,
                                  fmm=fmm,
                                  integration=integration,
                                  viscous=viscous,
                                  save_path=save_path,
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
function run_leapfrog(R1::Real, R2::Real,
                              Gamma1::Real, Gamma2::Real,
                              coR1::Real, coR2::Real,
                              Nphi::Int64, nc::Int64, Re::Real,
                              nsteps::Int64, nR::Real;
                              extra_nc::Int64=0,
                              faux1=1.0,
                              override_nu::Union{Nothing, Real}=nothing,
                              # SIMULATION SETUP
                              kernel::vpm.Kernel=vpm.kernel_wnklmns,
                              UJ::Function=vpm.UJ_direct,
                              fmm=vpm.FMM(; p=4, ncrit=10, theta=0.4, phi=0.5),
                              # NUMERICAL SCHEMES
                              transposed=true,
                              relax=true,
                              rlxf=0.3,
                              integration=vpm.rungekutta3,
                              nsteps_relax=1,
                              beta_cs=1.25,
                              viscous=true,
                              # SIMULATION OPTIONS
                              save_path="temps/leapfrog00",
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
    nu = viscous ? override_nu==nothing ? Gamma1/Re : override_nu : 0.0

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
    pfield = vpm.ParticleField(maxparticles; nu=nu, kernel=kernel, UJ=UJ,
                                transposed=transposed,
                                relax=relax, rlxf=rlxf,
                                integration=integration,
                                fmm=fmm
                                )
    sgm0 = sigma1                           # Default core size
    # beta_cs = 1.25                        # Maximum core size growth

    # Adds the rings to the field
    addvortexring(pfield, Gamma1, R1, Nphi, nc, rmax1;
                        extra_nc=extra_nc, lambda=lambda1)
    addvortexring(pfield, Gamma2, R2, Nphi, nc, rmax2;
                        extra_nc=extra_nc, lambda=lambda2)


    # -------------- RUNTIME FUNCTION --------------------------------------------
    Np=(1+4*nc*(nc+1))*Nphi   # Number of particles per ring
    C1s, C2s = [], []         # Center of the ring
    R1s, R2s = [], []         # Radius of the ring
    ts = []                   # Time stamp

    # Function for tracking the radius of each ring
    function rings_radii(pfield::vpm.ParticleField, t, dt)
        if 2*Np!=vpm.get_np(pfield);
            error("Logic error!\n2Np:$(2*Np)\nnp:$(vpm.get_np(pfield))")
        end

        Cs = [C1s, C2s]
        Rs = [R1s, R2s]
        intervals = [0, Np, 2Np]

        # Iterates over each ring
        for r_i in 1:2

            # Calculates center of the ring
            C = zeros(3)
            for i in (intervals[r_i]+1):(intervals[r_i+1])
                C .+= vpm.get_X(pfield, i)
            end
            C = C/Np

            # Calculates average radius
            ave_r = 0
            for i in (intervals[r_i]+1):(intervals[r_i+1])
                this_X = vpm.get_X(pfield, i)
                this_r = norm(this_X - C)
                ave_r += this_r
            end
            ave_r = ave_r/Np

            push!(Cs[r_i], C)
            push!(Rs[r_i], ave_r)
        end
        push!(ts, t)

        return false
    end

    # -------------- SIMULATION --------------------------------------------------
    # Runs the simulation
    vpm.run_vpm!(pfield, dt, nsteps; runtime_function=rings_radii,
                                        nsteps_relax=nsteps_relax,
                                        beta=beta_cs, sgm0=sgm0,
                                        save_path=save_path,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        verbose_nsteps=verbose_nsteps
                                        )



    # -------------- POST-PROCESSING ---------------------------------------------
    # Plot radii in time
    if disp_plot
        # Loads the data extracted from Berdowski's Fig 6.6
        data_r1 = CSV.read(data_path*"leapfrogring2.csv"; datarow=1)
        data_r2 = CSV.read(data_path*"leapfrogring1.csv"; datarow=1)

        plot( data_r1[!, 1], data_r1[!, 2], "-r", label="Ring 1 Analytical")
        plot( data_r2[!, 1], data_r2[!, 2], "-b", label="Ring 2 Analytical")

        plot(ts, R1s, ".r", label="FLOWVPM Ring 1", alpha=0.1)
        plot(ts, R2s, ".b", label="FLOWVPM Ring 2", alpha=0.1)

        legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=false)
        xlabel("Time (s)")
        ylabel("Ring radius (m)")
        grid(true, color="0.8", linestyle="--")
        title("Leapfrogging vortex rings")
        tight_layout()

        if save_path!=nothing
            savefig(joinpath(save_path, "leapfrog.png"))
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

    return nothing
end
