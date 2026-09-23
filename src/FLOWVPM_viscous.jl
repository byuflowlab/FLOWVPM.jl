#=##############################################################################
# DESCRIPTION
    Viscous schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
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

"""
    Inviscid()

Creates an inviscid scheme with zero kinematic viscosity.
"""
Inviscid() = Inviscid{FLOAT_TYPE}()

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
mutable struct CoreSpreading{R,Tzeta,Trbf} <: ViscousScheme{R}
    # User inputs
    nu::R                                 # Kinematic viscosity
    sgm0::R                               # Core size after reset
    zeta::Tzeta                        # Basis function evaluation method

    # Optional inputs
    beta::R                               # Maximum core size growth σ/σ_0
    growth_beta::R                        # Reset also when max σ/σ_0 over the field reaches this (0: off)
    itmax::Int                            # Maximum number of RBF iterations
    tol::R                                # RBF interpolation tolerance
    iterror::Bool                         # Throw error if RBF didn't converge
    verbose::Bool                         # Verbose on RBF interpolation
    v_lvl::Int                            # Verbose printing tab level
    debug::Bool                           # Print verbose for debugging

    # Internal properties
    t_sgm::R                              # Time since last core size reset
    rbf::Trbf                         # RBF function
    rr0s::Array{R, 1}                     # Initial field residuals
    rrs::Array{R, 1}                      # Current field residuals
    prev_rrs::Array{R, 1}                 # Previous field residuals
    pAps::Array{R, 1}                     # pAp product
    alphas::Array{R, 1}                   # Alpha coefficients
    betas::Array{R, 1}                    # Beta coefficients
    flags::Array{Bool, 1}                 # Convergence flags

    CoreSpreading{R,Tzeta,Trbf}(
                        nu, sgm0, zeta::Tzeta=zeta_fmm;
                        beta=R(1.5), growth_beta=R(0),
                        itmax=R(15), tol=R(1e-3),
                        iterror=true, verbose=false, v_lvl=2, debug=false,
                        t_sgm=R(0.0),
                        rbf::Trbf=rbf_conjugategradient,
                        rr0s=zeros(R, 3), rrs=zeros(R, 3), prev_rrs=zeros(R, 3),
                        pAps=zeros(R, 3), alphas=zeros(R, 3), betas=zeros(R, 3),
                        flags=zeros(Bool, 3)
                    ) where {R,Tzeta,Trbf} = new(
                        nu, sgm0, zeta,
                        beta, growth_beta,
                        itmax, tol,
                        iterror, verbose, v_lvl, debug,
                        t_sgm,
                        rbf,
                        rr0s, rrs, prev_rrs,
                        pAps, alphas, betas,
                        flags
                    )
end

"""
    CoreSpreading(nu, sgm0, zeta::Tzeta=zeta_fmm; <keyword arguments>)

Creates a core spreading viscous scheme with the given parameters.

# Arguments
- `nu`::Real Kinematic viscosity.
- `sgm0`::Real Core size after reset.
- `zeta::Function = zeta_fmm` Basis function evaluation method.
- `beta::Real = 1.5` Maximum core size growth σ/σ_0.
- `itmax::Int = 15` Maximum number of RBF iterations.
- `tol::Real = 1e-3` RBF interpolation tolerance.
- `iterror::Bool = true` Throw error if RBF didn't converge.
- `verbose::Bool = false` Verbose on RBF interpolation.
- `v_lvl::Int = 2` Verbose printing tab level.
- `debug::Bool = false` Print verbose for debugging.
- `rbf::Function = rbf_conjugategradient` RBF function.
- `rr0s::Array{R, 1} = zeros(R, 3)` Initial field residuals.
- `rrs::Array{R, 1} = zeros(R, 3)` Current field residuals.
- `prev_rrs::Array{R, 1} = zeros(R, 3)` Previous field residuals.
- `pAps::Array{R, 1} = zeros(R, 3)` pAp product.
- `alphas::Array{R, 1} = zeros(R, 3)` Alpha coefficients.
- `betas::Array{R, 1} = zeros(R, 3)`
- `flags::Array{Bool, 1} = zeros(Bool, 3)`
"""
CoreSpreading(nu, sgm0, zeta::Tzeta=zeta_fmm; rbf::Trbf=rbf_conjugategradient, optargs...
                    ) where {Tzeta,Trbf} = CoreSpreading{FLOAT_TYPE,Tzeta,Trbf}(FLOAT_TYPE(nu), FLOAT_TYPE(sgm0), zeta; rbf, optargs...)

"""
   `iscorespreading(scheme::ViscousScheme)`

Returns true if viscous scheme is core spreading.
"""
iscorespreading(scheme::ViscousScheme
                            ) = typeof(scheme) <: CoreSpreading



function viscousdiffusion(pfield, scheme::CoreSpreading, dt; aux1=0, aux2=0)

    proceed = false

    # ------------------ EULER SCHEME ------------------------------------------
    if pfield.integration == euler

        # Core spreading
        if pfield.particles isa Array
            for p in iterator(pfield)
                get_sigma(p)[] = sqrt(get_sigma(p)[]^2 + 2*scheme.nu*dt)
            end
        else
            _corespreading_euler_broadcast!(pfield, scheme.nu, dt)
        end

        proceed = true

    # ------------------ RUNGE-KUTTA SCHEME ------------------------------------
    elseif pfield.integration == rungekutta3

        # Core spreading
        # NOTE: Here we're solving dsigmadt as dsigma^2/dt = 2*nu.
        # Should I be solving dsigmadt = nu/sigma instead?
        if pfield.particles isa Array
            for p in iterator(pfield)
                get_M(p)[7] = aux1*get_M(p)[7] + dt*2*scheme.nu
                get_sigma(p)[] = sqrt(get_sigma(p)[]^2 + aux2*get_M(p)[7])
            end
        else
            _corespreading_rk3_broadcast!(pfield, scheme.nu, dt, aux1, aux2)
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

        # Update core growth timer
        scheme.t_sgm += dt

        beta_cur = sqrt(2*scheme.nu*scheme.t_sgm/scheme.sgm0^2 + 1)

        if scheme.verbose
            println("\t"^scheme.v_lvl*
                    "Current sigma growth: $(round(beta_cur, digits=7))"*
                    "\tCritical:$(round(scheme.beta, digits=7))")
        end

        # Reset core sizes if cores have overgrown: by viscous spreading
        # (the time criterion) or, with `growth_beta` set, by the field's
        # actual largest core (the rVPM stretching grows cores too, and a
        # tail of runaway cores sizes the FMM grid; Alvarez 2022 §4.6 CS-RBF
        # applied to the measured growth, 2026-09-21)
        growth = scheme.growth_beta > 0 ? _corespreading_sigma_max(pfield) / scheme.sgm0 : zero(scheme.sgm0)
        growth_reset = scheme.growth_beta > 0 && growth >= scheme.growth_beta
        if beta_cur >= scheme.beta || growth_reset
            growth_reset &&
                println("  core reset: max sigma/sigma0 = $(round(growth; digits=2)) >= " *
                        "$(scheme.growth_beta) at t = $(pfield.t), np = $(pfield.np)")
            _corespreading_reset!(pfield, scheme)
        end

    end
end

"GPU-compatible broadcast path for `CoreSpreading`'s Euler core-growth update."
function _corespreading_euler_broadcast!(pfield, nu, dt)
    R = eltype(pfield.particles)
    nu = R(nu); dt = R(dt)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active))
    sigma = view(P, SIGMA_INDEX, :)

    sigma .= ifelse.(active .> 0, sqrt.(sigma.^2 .+ 2*nu*dt), sigma)

    return nothing
end

"GPU-compatible broadcast path for `CoreSpreading`'s RK3 core-growth update."
function _corespreading_rk3_broadcast!(pfield, nu, dt, aux1, aux2)
    R = eltype(pfield.particles)
    nu = R(nu); dt = R(dt); aux1 = R(aux1); aux2 = R(aux2)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active))
    M7 = view(P, M_INDEX[7], :)
    sigma = view(P, SIGMA_INDEX, :)

    M7 .= ifelse.(active .> 0, aux1 .* M7 .+ dt*2*nu, M7)
    sigma .= ifelse.(active .> 0, sqrt.(sigma.^2 .+ aux2 .* M7), sigma)

    return nothing
end

"GPU-compatible broadcast path for `CoreSpreading`'s core-size reset."
function _corespreading_reset_broadcast!(pfield, sgm0)
    R = eltype(pfield.particles)
    sgm0 = R(sgm0)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active))

    for i in 1:3
        M = view(P, M_INDEX[6+i], :)
        W = view(P, VORTICITY_INDEX[i], :)
        M .= ifelse.(active .> 0, W, M)
    end

    sigma = view(P, SIGMA_INDEX, :)
    sigma .= ifelse.(active .> 0, sgm0, sigma)

    return nothing
end
##### END OF CORE SPREADING SCHEME #############################################



################################################################################
# NOTE 2026-09-22: the ParticleStrengthExchange scheme was removed. PSE approximates
# the Laplacian by exchanging strength between overlapping neighbours and is only
# consistent on a regular (remeshed) particle distribution; this VPM is meshless and
# never remeshes, so PSE was never usable here. CoreSpreading is the viscous scheme.
################################################################################
##### END OF PARTICLE STRENGTH EXCHANGE SCHEME ###################################





# The CS-RBF reset: ζ-reconstructed vorticity as the target, every core back
# to `sgm0`, strengths re-solved by conjugate gradient (host and device
# `rbf_conjugategradient`, `zeta_fmm` methods), growth timer restarted.
function _corespreading_reset!(pfield, scheme::CoreSpreading)
    # Calculate approximated vorticity into the dedicated vorticity field.
    scheme.zeta(pfield)

    if pfield.particles isa Array
        for p in iterator(pfield)
            # Use approximated vorticity as target vorticity (stored under P.M[7:9]).
            for i in 1:3
                get_M(p)[6+i] = get_vorticity(p)[i]
            end
            # Reset core sizes
            get_sigma(p)[] = scheme.sgm0
        end
    else
        _corespreading_reset_broadcast!(pfield, scheme.sgm0)
    end

    # Calculate new strengths through RBF to preserve original vorticity
    scheme.rbf(pfield, scheme)
    scheme.growth_beta > 0 && println("  core reset: RBF residual ",
        join((Printf.@sprintf("%.2e", sqrt(scheme.rrs[i] / max(scheme.rr0s[i], eps()))) for i in 1:3), " "),
        " (tol $(scheme.tol), itmax $(scheme.itmax))")

    # The radix geometry was derived for the old cores: with every core back
    # at sgm0 the sigma-limited depth is far too shallow (HVAB 2026-09-21:
    # 40 s/step on the depth-4 grid after the reset). Drop the coupling so the
    # next evaluation re-derives it; the rebuild costs one build, the reset is rare.
    clear_radix_fmm_cache!(pfield)

    # Reset core growth timer
    scheme.t_sgm = 0
    return nothing
end

"""
    corespreading_reset_subset!(pfield, scheme, idx) -> nothing

The CS-RBF reset restricted to the particles `idx` (global indices): their cores
go to `scheme.sgm0` and their strengths are re-solved so that the ζ-vorticity
at their positions is preserved, with every other particle held fixed. The
target is ω_total(x_S) − ζ_rest(x_S) (two ζ evaluations), the conjugate
gradient runs over the subset only (the rest is flagged static and carries
zero strength during the solve, so A·p is linear in p), then the rest is
restored. Meant for the OVERGROWN tail: a reset of contracted cores cannot be
represented at sgm0 (2026-09-22, see notes) and this never touches them.
Host and device fields (the gather/scatter helpers of the oversize mask).
"""
function corespreading_reset_subset!(pfield, scheme::CoreSpreading, idx::Vector{Int})
    isempty(idx) && return nothing
    P = pfield.particles; np = pfield.np
    R = eltype(P)
    grows = first(GAMMA_INDEX):last(GAMMA_INDEX)
    # ω_total at the subset
    scheme.zeta(pfield)
    Wtot = _radix_oversize_gather(P, idx, VORTICITY_INDEX)
    # ζ of the rest at the subset: zero the subset's strength, evaluate again
    Gsub = _radix_oversize_gather(P, idx, grows)
    _radix_oversize_mask_rows!(P, idx, grows)
    scheme.zeta(pfield)
    Wrest = _radix_oversize_gather(P, idx, VORTICITY_INDEX)
    # freeze the rest: zero strength (contributes nothing to A·p) and out of the CG
    # through the explicit active mask (the static flag was removed 2026-09-22)
    Grest = copy(view(P, grows, 1:np))
    view(P, grows, 1:np) .= zero(R)
    K = length(idx)
    _radix_oversize_scatter!(P, idx, SIGMA_INDEX:SIGMA_INDEX, fill(R(scheme.sgm0), 1, K))
    _radix_oversize_scatter!(P, idx, M_INDEX[7]:M_INDEX[9], Matrix{R}(Wtot .- Wrest))
    _radix_oversize_scatter!(P, idx, grows, Matrix{R}(Gsub))      # initial search direction: old strengths
    active = falses(np); active[idx] .= true
    rbf_conjugategradient(pfield, scheme; active)
    Gnew = _radix_oversize_gather(P, idx, grows)
    # restore the rest, keep the subset's new strengths
    copyto!(view(P, grows, 1:np), Grest)
    println("  core reset (subset of $K): RBF residual ",
        join((Printf.@sprintf("%.2e", sqrt(scheme.rrs[i] / max(scheme.rr0s[i], eps()))) for i in 1:3), " "),
        " (tol $(scheme.tol), itmax $(scheme.itmax))")
    return nothing
end

_corespreading_sigma_max(pfield) = pfield.np == 0 ? zero(eltype(pfield.particles)) :
    (pfield.particles isa Array ? maximum(view(pfield.particles, SIGMA_INDEX, 1:pfield.np)) :
                                  _radix_sigma_max(pfield))

##### COMMON FUNCTIONS #########################################################

"""
Radial basis function interpolation of Gamma using the conjugate gradient
method. This method only works on a particle field with uniform smoothing
radius sigma.

See 20180818 notebook and https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
"""
# `active`: optional Bool vector (length np); particles with `false` keep their rows
# untouched and drop out of every dot product (used by corespreading_reset_subset!).
function rbf_conjugategradient(pfield, cs::CoreSpreading; active=nothing)

    #= NOTES
    * The target vorticity (`omega_targ`) is expected to be stored in P.M[7:9]
    (give it the basis-approximated vorticity instead of the UJ-calculated
    one or the method will diverge).
    * The basis function evaluation (`omega_cur`) is stored in the vorticity
    field.
    * The solution is built under P.M[1:3] (it used to be x).
    * The current residual is stored under P.M[4:6] (it used to be r).
    =#

    if cs.debug
        println("\t"^(cs.v_lvl+1)*"***** Probe Particle 1 ******\n"*
                "\t"^(cs.v_lvl+2)*"Init Gamma:\t$(round.(get_particle(pfield, 1)[4:6], digits=8))\n"*
                "\t"^(cs.v_lvl+2)*"Target w:\t$(round.(get_particle(pfield, 1)[M_INDEX[7:9]], digits=8))\n")
    end

    # Initialize memory
    cs.rr0s .= 0
    cs.rrs .= 0
    cs.flags .= false

    for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
        for i in 1:3
            # Initial guess: Γ_i ≈ ω_i⋅vol_i
            get_M(P)[i] = get_M(P)[6+i]*get_vol(P)[]
            # Sets initial guess as Gamma for vorticity evaluation
            get_Gamma(P)[i] = get_M(P)[i]
        end
    end

    # Current vorticity: evaluate basis functions into the vorticity field.
    cs.zeta(pfield)

    for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
        for i in 1:3
            # Residual of initial guess (r0=b-Ax0)
            get_M(P)[3+i] = get_M(P)[6+i] - get_vorticity(P)[i]    # r = omega_targ - omega_cur

            # Update coefficients
            get_Gamma(P)[i] = get_M(P)[3+i]             # p0 = r0

            # Initial field residual
            cs.rr0s[i] += (get_M(P)[3+i])^2
        end
    end

    cs.rrs .= cs.rr0s                         # Current field residuals
    for i in 1:3                              # Iteration flag of each dimension
        cs.flags[i] = sqrt(cs.rr0s[i]) > cs.tol || sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
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
        for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
            for i in 1:3
                cs.pAps[i] += get_Gamma(P)[i] .* get_vorticity(P)[i]
            end
        end

        for i in 1:3
            cs.alphas[i] = cs.rrs[i] / cs.pAps[i] * cs.flags[i]
            # cs.alphas[i] = cs.rrs[i]/cs.pAps[i]
        end

        cs.prev_rrs .= cs.rrs
        cs.rrs .= 0

        for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
            for i in 1:3
                get_M(P)[i] += cs.alphas[i]*get_Gamma(P)[i]   # x = x + alpha*p
                get_M(P)[i+3] -= cs.alphas[i].*get_vorticity(P)[i] # r = r - alpha*Ap
                cs.rrs[i] += get_M(P)[i+3]^2             # Update field residual
            end
        end

        cs.betas .= cs.rrs
        cs.betas ./= cs.prev_rrs

        # Avoid dividing by zero
        for i in 1:3
            if abs(cs.prev_rrs[i]) <= 2*eps()
                cs.betas[i] = 1
            end
        end

        for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
            for i in 1:3
                get_Gamma(P)[i] = get_M(P)[i+3] + cs.betas[i]*get_Gamma(P)[i]
            end
        end

        for i in 1:3
            cs.flags[i] *= abs(cs.rr0s[i]) <= 2*eps() ? false : sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
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
            println(
                    "\t"^(cs.v_lvl+1)*"Iteration $(it) / $(cs.itmax) max\n"*
                    "\t"^(cs.v_lvl+2)*"Error: $(sqrt.(cs.rrs ./ cs.rr0s))\n"*
                    "\t"^(cs.v_lvl+2)*"Flags: $(cs.flags)\n"*
                    "\t"^(cs.v_lvl+2)*"Sol Particle 1: $(round.(get_particle(pfield, 1)[28:30], digits=8))"
                   )
        end

    end

    # Save final solution
    for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
        for i in 1:3
            get_Gamma(P)[i] = get_M(P)[i]
        end
    end

    if cs.debug
        # Evaluate current vorticity
        cs.zeta(pfield)
        println("\t"^(cs.v_lvl+1)*"***** Probe Particle 1 ******\n"*
                "\t"^(cs.v_lvl+2)*"Final Gamma:\t$(round.(get_Gamma(pfield, 1), digits=8))\n"*
                "\t"^(cs.v_lvl+2)*"Final w:\t$(round.(get_vorticity(pfield, 1), digits=8))")
        println("\t"^(cs.v_lvl+1)*"***** COMPLETED RBF ******\n")

        rms_ini, rms_resend = zeros(3), zeros(3)

        for (ip, P) in enumerate(iterator(pfield))
        (active === nothing || active[ip]) || continue
            for i in 1:3
                rms_ini[i] += get_M(P)[i+6]^2
                rms_resend[i] += (get_vorticity(P)[i] - get_M(P)[i+6])^2
            end
        end
        for i in 1:3
            rms_ini[i] = sqrt(rms_ini[i])
            rms_resend[i] = sqrt(rms_resend[i])
        end

        println("\t"^(cs.v_lvl+1)*"RMS residual / RMS Wtarg: $(rms_resend./rms_ini)\n")
    end

    return nothing
end



"""
    gpu_zeta_direct!(pfield::ParticleField)

GPU implementation of the O(N²) direct-sum basis-function evaluation used by
`zeta_direct`, dispatched to when `pfield.particles` is not a plain `Array`.
Real implementation is a method of this function defined in
`ext/FLOWVPMGPUExt.jl` (loaded automatically alongside KernelAbstractions and a GPU package). This stub
is only reached if a non-`Array` particle field is used without CUDA.jl
loaded.
"""
gpu_zeta_direct!(pfield) = error(
    "No GPU zeta_direct implementation available for particle arrays of type " *
    "$(typeof(pfield.particles)). Load `using CUDA` alongside FLOWVPM to enable " *
    "the GPU direct basis-function evaluation for `CuArray`-backed particle fields.")

"""
  `zeta_direct(pfield)`

Evaluates the basis function that the field exerts on itself through direct
particle-to-particle interactions, saving the results under `VORTICITY_INDEX`.
"""
function zeta_direct(pfield)
    if pfield.particles isa Array
        if Threads.nthreads() > 1
            return zeta_direct_multithreaded(pfield)
        else
            return zeta_direct_singlethreaded(pfield)
        end
    else
        return gpu_zeta_direct!(pfield)
    end
end

function zeta_direct_singlethreaded(pfield)
    for P in iterator(pfield; include_static=true)
        get_vorticity(P) .= 0
    end
    return zeta_direct( iterator(pfield; include_static=true),
                        iterator(pfield; include_static=true),
                        pfield.kernel.zeta)
end

"""
    `zeta_direct_multithreaded(pfield)`

`Threads.@threads`-parallel version of `zeta_direct`'s CPU path, chunking
target particles across threads (mirrors `Estr_direct_multithreaded` in
`FLOWVPM_subfilterscale_models.jl`). Untyped `pfield` argument: this file is
`include`d before `FLOWVPM_particlefield.jl` defines `ParticleField`, so it
can't be annotated `::ParticleField` here (same constraint noted on the
`gpu_zeta_direct!` stub above).
"""
function zeta_direct_multithreaded(pfield)
    np = pfield.np
    for i in 1:np
        get_vorticity(pfield, i) .= 0
    end

    n_per_thread, rem = divrem(np, Threads.nthreads())
    n = n_per_thread + (rem > 0)
    assignments = 1:n:np
    zeta = pfield.kernel.zeta

    Threads.@threads for i_assignment in eachindex(assignments)
        start_idx = assignments[i_assignment]
        end_idx = min(start_idx + n - 1, np)

        for i_target in start_idx:end_idx
            Pi = get_particle(pfield, i_target)

            for i_source in 1:np
                Pj = get_particle(pfield, i_source)

                dX1 = get_X(Pi)[1] - get_X(Pj)[1]
                dX2 = get_X(Pi)[2] - get_X(Pj)[2]
                dX3 = get_X(Pi)[3] - get_X(Pj)[3]
                r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

                zeta_sgm = 1/get_sigma(Pj)[]^3*zeta(r/get_sigma(Pj)[])

                get_vorticity(Pi)[1] += get_Gamma(Pj)[1]*zeta_sgm
                get_vorticity(Pi)[2] += get_Gamma(Pj)[2]*zeta_sgm
                get_vorticity(Pi)[3] += get_Gamma(Pj)[3]*zeta_sgm
            end
        end
    end
    return nothing
end

function zeta_direct(sources, targets, zeta::Function)

    for Pi in targets
        for Pj in sources

            dX1 = get_X(Pi)[1] - get_X(Pj)[1]
            dX2 = get_X(Pi)[2] - get_X(Pj)[2]
            dX3 = get_X(Pi)[3] - get_X(Pj)[3]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            zeta_sgm = 1/get_sigma(Pj)[]^3*zeta(r/get_sigma(Pj)[])

            get_vorticity(Pi)[1] += get_Gamma(Pj)[1]*zeta_sgm
            get_vorticity(Pi)[2] += get_Gamma(Pj)[2]*zeta_sgm
            get_vorticity(Pi)[3] += get_Gamma(Pj)[3]*zeta_sgm

        end
    end
end

"""
  `zeta_fmm(pfield)`

Evaluates the basis function that the field exerts on itself through
the FMM neglecting the far field, saving the results under `VORTICITY_INDEX`.
"""
function zeta_fmm(pfield)
    fmm_options = pfield.fmm
    leaf_size=fmm_options.ncrit
    shrink=fmm_options.shrink_recenter
    recenter=fmm_options.shrink_recenter
    multipole_acceptance = fmm_options.theta
    zeta = pfield.kernel.zeta

    # create tree
    switches = FastMultipole.DerivativesSwitch(false, true, false, (pfield,))
    leaf_sizes = SVector(leaf_size)
    tree = FastMultipole.Tree((pfield,), false, switches;
                              leaf_size=leaf_sizes,
                              shrink=shrink,
                              recenter=recenter,
                              interaction_list_method=FastMultipole.SelfTuningTargetStop())
    _, direct_list = FastMultipole.build_interaction_lists(tree.branches, tree.branches, leaf_sizes, multipole_acceptance, false, true, true)
    sort_index = tree.sort_index_list[1]

    for P in iterator(pfield; include_static=true)
        get_vorticity(P) .= 0
    end

    # loop over direct list
    for (i_target, i_source) in direct_list
        target_branch = tree.branches[i_target]
        source_branch = tree.branches[i_source]
        for i_source_particle in source_branch.bodies_index[1]
            Pi = get_particle(pfield, sort_index[i_source_particle])

            for i_target_particle in target_branch.bodies_index[1]
                Pj = get_particle(pfield, sort_index[i_target_particle])

                dX1 = get_X(Pi)[1] - get_X(Pj)[1]
                dX2 = get_X(Pi)[2] - get_X(Pj)[2]
                dX3 = get_X(Pi)[3] - get_X(Pj)[3]
                r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

                zeta_sgm = 1/get_sigma(Pj)[]^3*zeta(r/get_sigma(Pj)[])

                get_vorticity(Pi)[1] += get_Gamma(Pj)[1]*zeta_sgm
                get_vorticity(Pi)[2] += get_Gamma(Pj)[2]*zeta_sgm
                get_vorticity(Pi)[3] += get_Gamma(Pj)[3]*zeta_sgm
            end
        end
    end
end

################################################################################
