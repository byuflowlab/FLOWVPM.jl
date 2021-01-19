#=##############################################################################
# DESCRIPTION
    Viscous schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

################################################################################
# ABSTRACT VISCOUS SCHEME TYPE
################################################################################
"""
    `ViscousScheme{R}`

Type declaring viscous scheme.

Implementations must have the following properties:
    * `nu::R`                   : Kinematic viscosity.
"""
abstract type ViscousScheme{R} end

"""
Implementation of viscous diffusion scheme that gets called in the inner loop
of the time integration scheme at each time step.
"""
function viscousdiffusion(pfield, scheme::ViscousScheme, dt; optargs...)
    error("Viscous diffusion scheme has not been implemented yet!")
end

viscousdiffusion(pfield, dt; optargs...
                    ) = viscousdiffusion(pfield, pfield.viscous, dt; optargs...)
##### END OF ABSTRACT VISCOUS SCHEME ###########################################

################################################################################
# INVISCID SCHEME TYPE
################################################################################
struct Inviscid{R} <: ViscousScheme{R}
    nu::R                                 # Kinematic viscosity
    Inviscid{R}(; nu=zero(R)) where {R} = new(nu)
end

Inviscid() = Inviscid{RealFMM}()

"""
    `isinviscid(scheme::ViscousScheme)`

Returns true if viscous scheme is inviscid.
"""
isinviscid(scheme::ViscousScheme) = typeof(scheme).name == Inviscid.body.name

viscousdiffusion(pfield, scheme::Inviscid, dt; optargs...) = nothing
##### END OF INVISCID SCHEME ###################################################


################################################################################
# CORE SPEADING SCHEME TYPE
################################################################################
mutable struct CoreSpreading{R} <: ViscousScheme{R}
    # User inputs
    nu::R                                 # Kinematic viscosity
    sgm0::R                               # Core size after reset
    zeta::Function                        # Basis function evaluation method

    # Optional inputs
    beta::R                               # Maximum core size growth σ/σ_0
    itmax::Int                            # Maximum number of RBF iterations
    tol::R                                # RBF interpolation tolerance
    iterror::Bool                         # Throw error if RBF didn't converge
    verbose::Bool                         # Verbose on RBF interpolation
    v_lvl::Int                            # Verbose printing tab level
    debug::Bool                           # Print verbose for debugging

    # Internal properties
    t_sgm::R                              # Time since last core size reset
    rbf::Function                         # RBF function
    rr0s::Array{R, 1}                     # Initial field residuals
    rrs::Array{R, 1}                      # Current field residuals
    prev_rrs::Array{R, 1}                 # Previous field residuals
    pAps::Array{R, 1}                     # pAp product
    alphas::Array{R, 1}                   # Alpha coefficients
    betas::Array{R, 1}                    # Beta coefficients
    flags::Array{Bool, 1}                 # Convergence flags

    CoreSpreading{R}(
                        nu, sgm0, zeta;
                        beta=R(1.5),
                        itmax=R(15), tol=R(1e-3),
                        iterror=true, verbose=false, v_lvl=2, debug=false,
                        t_sgm=R(0.0),
                        rbf=rbf_conjugategradient,
                        rr0s=zeros(R, 3), rrs=zeros(R, 3), prev_rrs=zeros(R, 3),
                        pAps=zeros(R, 3), alphas=zeros(R, 3), betas=zeros(R, 3),
                        flags=zeros(Bool, 3)
                    ) where {R} = new(
                        nu, sgm0, zeta,
                        beta,
                        itmax, tol,
                        iterror, verbose, v_lvl, debug,
                        t_sgm,
                        rbf,
                        rr0s, rrs, prev_rrs,
                        pAps, alphas, betas,
                        flags
                    )
end

CoreSpreading(nu, sgm0, args...; optargs...
                    ) = CoreSpreading{RealFMM}(RealFMM(nu), RealFMM(sgm0), args...; optargs...)

"""
   `iscorespreading(scheme::ViscousScheme)`

Returns true if viscous scheme is core spreading.
"""
iscorespreading(scheme::ViscousScheme
                            ) = typeof(scheme).name == CoreSpreading.body.name



function viscousdiffusion(pfield, scheme::CoreSpreading, dt; aux1=0, aux2=0)

    proceed = false

    # ------------------ EULER SCHEME ------------------------------------------
    if pfield.integration == euler

        # Core spreading
        for p in iterator(pfield)
            p.sigma[1] = sqrt(p.sigma[1]^2 + 2*scheme.nu*dt)
        end

        proceed = true

    # ------------------ RUNGE-KUTTA SCHEME ------------------------------------
    elseif pfield.integration == rungekutta3

        # Core spreading
        for p in iterator(pfield)
            p.M[1, 3] = aux1*p.M[1, 3] + dt*2*scheme.nu
            p.sigma[1] = sqrt(p.sigma[1]^2 + aux2*p.M[1, 3])
        end

        # Update things in the last RK inner iteration
        if isapprox(aux2, 8/15, atol=1e-7)
            proceed = true
        end

    # ------------------ DEFAULT -----------------------------------------------
    else
        error("Time integration scheme $(pfield.integration) not"*
                        " implemented in core spreading viscous scheme yet!")
    end

    if proceed

        t_sgm_crit = scheme.sgm0^2*(scheme.beta^2-1)/(2*scheme.nu)

        # Update core growth timer
        scheme.t_sgm += dt

        if scheme.verbose
            println("\t"^scheme.v_lvl*
                    "Current sigma growth: $(round(scheme.t_sgm, digits=7))"*
                    "\tCritical:$(round(t_sgm_crit, digits=7))")
        end

        # Reset core sizes if cores have overgrown
        if scheme.t_sgm >= t_sgm_crit
            # Calculate approximated vorticity (stored under P.Jexa[1:3])
            scheme.zeta(pfield)

            for p in iterator(pfield)
                # Use approximated vorticity as target vorticity (stored under P.Jexa[7:9])
                for i in 1:3
                    p.M[i+6] = p.Jexa[i]
                end
                # Reset core sizes
                p.sigma[1] = scheme.sgm0
            end

            # Calculate new strengths through RBF to preserve original vorticity
            scheme.rbf(pfield, scheme)

            # Reset core growth timer
            scheme.t_sgm = 0
        end

    end
end




"""
Radial basis function interpolation of Gamma using the conjugate gradient
method. This method only works on a particle field with uniform smoothing
radius sigma.

See 20180818 notebook and https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
"""
function rbf_conjugategradient(pfield, cs::CoreSpreading)

    #= NOTES
    * The target vorticity (`omega_targ`) is expected to be stored in P.M[7:9]
        (give it the basis-approximated vorticity instead of the UJ-calculated
        one or the method will diverge).
    * The basis function evaluation (`omega_cur`) is stored in Jexa[1:3] (it
        used to be p).
    * The solution is built under P.M[1:3] (it used to be x).
    * The current residual is stored under P.M[4:6] (it used to be r).
    =#

    # Initialize memory
    cs.rr0s .= 0
    cs.rrs .= 0
    cs.flags .= false

    for P in iterator(pfield)
        for i in 1:3
            # Initial guess: Γ_i ≈ ω_i⋅vol_i
            P.M[i] = P.M[i+6]*P.vol[1]
            # Sets initial guess as Gamma for vorticity evaluation
            P.Gamma[i] = P.M[i]
        end
    end

    # Current vorticity: Evaluate basis function storing results under P.Jexa[1:3]
    cs.zeta(pfield)

    for P in iterator(pfield)
        for i in 1:3
            # Residual of initial guess (r0=b-Ax0)
            P.M[i+3] = P.M[i+6] - P.Jexa[i]    # r = omega_targ - omega_cur

            # Update coefficients
            P.Gamma[i] = P.M[i+3]             # p0 = r0

            # Initial field residual
            cs.rr0s[i] += P.M[i+3]^2
        end
    end

    cs.rrs .= cs.rr0s                         # Current field residuals
    for i in 1:3                              # Iteration flag of each dimension
        cs.flags[i] = sqrt(cs.rr0s[i]) > cs.tol && sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
    end

    # Run Conjugate Gradient algorithm
    for it in 1:cs.itmax
        if !(true in cs.flags)
            break
        end

        # Evaluate current vorticity
        cs.zeta(pfield)

        # Calculate pAp product on each dimension
        cs.pAps .= 0
        for P in iterator(pfield)
            for i in 1:3
                cs.pAps[i] += P.Gamma[i] * P.Jexa[i]
            end
        end

        for i in 1:3                          # alpha = rr./pAp
            # cs.alphas[i] = cs.rrs[i]/cs.pAps[i] * cs.flags[i]
            cs.alphas[i] = cs.rrs[i]/cs.pAps[i]
        end

        cs.prev_rrs .= cs.rrs
        cs.rrs .= 0

        for P in iterator(pfield)
            for i in 1:3
                P.M[i] += cs.alphas[i]*P.Gamma[i]   # x = x + alpha*p
                P.M[i+3] -= cs.alphas[i].*P.Jexa[i] # r = r - alpha*Ap
                cs.rrs[i] += P.M[i+3]^2             # Update field residual
            end
        end

        cs.betas .= cs.rrs
        cs.betas ./= cs.prev_rrs

        for P in iterator(pfield)
            for i in 1:3
                P.Gamma[i] = P.M[i+3] + cs.betas[i]*P.Gamma[i]
            end
        end

        for i in 1:3
            cs.flags[i] *= sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
        end

        # Non-convergenced case
        if it==cs.itmax && true in cs.flags
            if cs.iterror
                error("Maximum number of iterations $(cs.itmax) reached before"*
                        " convergence."*
                        " Errors: $(sqrt.(cs.rrs ./ cs.rr0s)), tolerance:$(cs.tol)")
            elseif cs.verbose
                @warn("Maximum number of iterations $(cs.itmax) reached before"*
                        " convergence."*
                        " Errors: $(sqrt.(cs.rrs ./ cs.rr0s)), tolerance:$(cs.tol)")
            else
                nothing
            end
        end

        if cs.debug
            println("\t"^(cs.v_lvl+1)*
                        "Iteration $(it)\tError: $(sqrt.(cs.rrs ./ cs.rr0s))")
        end

    end

    # Save final solution
    for P in iterator(pfield)
        for i in 1:3
            P.Gamma[i] = P.M[i]
        end
    end

    return nothing
end



"""
  `zeta_direct(pfield)`

Evaluates the basis function that the field exerts on itself through direct
particle-to-particle interactions, saving the results under P.Jexa[1:3].
"""
function zeta_direct(pfield)
    for P in iterator(pfield); P.Jexa[1:3] .= 0; end;
    return zeta_direct(iterator(pfield), iterator(pfield), pfield.kernel.zeta)
end

function zeta_direct(sources, targets, zeta::Function)

    for Pi in targets
        for Pj in sources

            dX1 = Pi.X[1] - Pj.X[1]
            dX2 = Pi.X[2] - Pj.X[2]
            dX3 = Pi.X[3] - Pj.X[3]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            zeta_sgm = 1/Pj.sigma[1]*zeta(r/Pj.sigma[1])

            Pi.Jexa[1] += Pj.Gamma[1]*zeta_sgm
            Pi.Jexa[2] += Pj.Gamma[2]*zeta_sgm
            Pi.Jexa[3] += Pj.Gamma[3]*zeta_sgm

        end
    end
end

"""
  `zeta_fmm(pfield)`

Evaluates the basis function that the field exerts on itself through
the FMM neglecting the far field, saving the results under P.Jexa[1:3].
"""
function zeta_fmm(pfield)
    call_FLOWExaFMM(pfield; rbf=true)
end
##### END OF CORE SPREADING SCHEME #############################################


##### COMMON FUNCTIONS #########################################################
################################################################################
