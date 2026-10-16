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
of the time integration scheme at each time step. The calling integrator passes
itself as `integration` (`euler` or `rungekutta3`), so the scheme follows the
integrator that runs, not the field's `integration` setting (a driver may step a
field with either).
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
# CORE SPREADING SCHEME TYPE
################################################################################
mutable struct CoreSpreading{R,TG<:GalerkinReset{R}} <: ViscousScheme{R}
    nu::R                                 # Kinematic viscosity
    sgm0::R                               # Core size after reset
    beta::R                               # Maximum core size growth sigma/sigma_0
    growth_beta::R                        # Reset also when max sigma/sigma_0 over the field reaches this (0: off)
    iterror::Bool                         # Throw (else warn) when the reset's solve does not converge
    verbose::Bool                         # Print a summary line per reset
    v_lvl::Int                            # Verbose printing tab level
    galerkin::TG                          # Reset parameters
    t_sgm::R                              # Time since last core size reset
end

CoreSpreading{R}(nu, sgm0; beta=R(1.25), growth_beta=R(0), iterror=false, verbose=false, v_lvl=2,
                 galerkin=GalerkinReset{R}(), t_sgm=R(0)) where R =
    CoreSpreading{R,GalerkinReset{R}}(nu, sgm0, beta, growth_beta, iterror, verbose, v_lvl,
                                      convert(GalerkinReset{R}, galerkin), t_sgm)

"""
    CoreSpreading(nu, sgm0; <keyword arguments>)

Core spreading viscous scheme: every active core grows as sigma^2 += 2 nu dt, and
once the cores have grown by `beta` (sigma/sgm0 at the growth timer) they are reset
to `sgm0` by the Galerkin L2 projection of the old Gaussian field (`galerkin_shrink!`,
parameters `GalerkinReset`): the reset particles' neighbours (the halo) keep their
cores and take strength increments, new `sgm0` points are added where the old tails
reach past the last particle, and close reset particles are thinned. The particle
count and order change, so the field needs `maxparticles` headroom. Static particles
are never reset nor in the halo. Gaussian (gaussianerf) kernel only.

# Arguments
- `nu::Real` Kinematic viscosity.
- `sgm0::Real` Core size after reset.
- `beta::Real = 1.25` Maximum core size growth sigma/sgm0.
- `growth_beta::Real = 0` Also reset when the largest core of the field reaches
  `growth_beta * sgm0` (0: off).
- `iterror::Bool = false` The reset's conjugate gradient has not converged when it
  stops at `galerkin.itmax` (or breaks down) before its stopping rule; if the relative
  field error is then above `galerkin.tol`, throw (`true`) or warn (`false`).
- `verbose::Bool = false` Print a summary line per reset (field error, iterations,
  halo, edge points, thinned, time).
- `v_lvl::Int = 2` Verbose printing tab level.
- `galerkin::GalerkinReset = GalerkinReset()` Reset parameters (solver `tol`/`itmax`,
  halo, edge points, thinning, preconditioner, memory cap).
"""
CoreSpreading(nu, sgm0; optargs...) = CoreSpreading{FLOAT_TYPE}(FLOAT_TYPE(nu), FLOAT_TYPE(sgm0); optargs...)

"""
   `iscorespreading(scheme::ViscousScheme)`

Returns true if viscous scheme is core spreading.
"""
iscorespreading(scheme::ViscousScheme
                            ) = typeof(scheme) <: CoreSpreading



function viscousdiffusion(pfield, scheme::CoreSpreading, dt; aux1=0, aux2=0, integration=pfield.integration)

    proceed = false

    # ------------------ EULER SCHEME ------------------------------------------
    if integration == euler

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
    elseif integration == rungekutta3

        # Integrating d(sigma^2)/dt = 2nu is equivalent to d(sigma)/dt = nu/sigma.
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
        error("Time integration scheme $(integration) not"*
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
        # applied to the measured growth)
        growth = scheme.growth_beta > 0 ? _corespreading_sigma_max(pfield) / scheme.sgm0 : zero(scheme.sgm0)
        growth_reset = scheme.growth_beta > 0 && growth >= scheme.growth_beta
        if beta_cur >= scheme.beta || growth_reset
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

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)
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

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)
    M7 = view(P, M_INDEX[7], :)
    sigma = view(P, SIGMA_INDEX, :)

    M7 .= ifelse.(active .> 0, aux1 .* M7 .+ dt*2*nu, M7)
    sigma .= ifelse.(active .> 0, sqrt.(sigma.^2 .+ aux2 .* M7), sigma)

    return nothing
end

##### END OF CORE SPREADING SCHEME #############################################



################################################################################
# PARTICLE STRENGTH EXCHANGE SCHEME TYPE
################################################################################
mutable struct ParticleStrengthExchange{R} <: ViscousScheme{R}
    # User inputs
    nu::R                                 # Kinematic viscosity

    # Optional inputs
    recalculate_vols::Bool                # Whether to recalculate volumes

    ParticleStrengthExchange{R}(
                                    nu; recalculate_vols=true
                                ) where {R} = new(
                                    nu, recalculate_vols
                                )
end

"""
    ParticleStrengthExchange(nu; <keyword arguments>)

Creates a particle strength exchange viscous scheme with the given parameters.

# Arguments
- `nu`::Real Kinematic viscosity.
- `recalculate_vols::Bool = true` Whether to recalculate particle volumes.
"""
ParticleStrengthExchange(nu, args...; optargs...
                        ) = ParticleStrengthExchange{FLOAT_TYPE}(FLOAT_TYPE(nu), args...; optargs...)

function viscousdiffusion(pfield, scheme::ParticleStrengthExchange, dt; aux1=0, aux2=0, integration=pfield.integration)

    if pfield.UJ != UJ_fmm
        # NOTE: PSE has only been implemented with FMM so far
        error("PSE with UJ function $(pfield.UJ) has not been implemented yet!")
    end

    # Recalculate particle volume from current particle smoothing
    if scheme.recalculate_vols
        if pfield.particles isa Array
            for p in iterator(pfield)
                get_vol(p)[] = 4/3*pi*get_sigma(p)[]^3
            end
        else
            _pse_recalcvols_broadcast!(pfield)
        end
    end

    # ------------------ EULER SCHEME ------------------------------------------
    if integration == euler

        # Update Gamma
        if pfield.particles isa Array
            for p in iterator(pfield)
                for i in 1:3
                    get_Gamma(p)[i] += dt * scheme.nu*get_PSE(p)[i]
                end
            end
        else
            _pse_euler_broadcast!(pfield, scheme.nu, dt)
        end

        # ------------------ RUNGE-KUTTA SCHEME ------------------------------------
    elseif integration == rungekutta3

        # Update Gamma
        if pfield.particles isa Array
            for p in iterator(pfield)
                for i in 1:3
                    get_M(p)[3+i] += dt * scheme.nu*get_PSE(p)[i]
                    get_Gamma(p)[i] += aux2 * dt * scheme.nu*get_PSE(p)[i]
                end
            end
        else
            _pse_rk3_broadcast!(pfield, scheme.nu, dt, aux2)
        end

        # ------------------ DEFAULT -----------------------------------------------
    else
        error("Time integration scheme $(integration) not"*
              " implemented in PSE viscous scheme yet!")
    end

end

"GPU-compatible broadcast path for `ParticleStrengthExchange`'s volume recalculation."
function _pse_recalcvols_broadcast!(pfield)
    R = eltype(pfield.particles)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)
    vol = view(P, VOL_INDEX, :)
    sigma = view(P, SIGMA_INDEX, :)

    vol .= ifelse.(active .> 0, R(4/3*pi) .* sigma.^3, vol)

    return nothing
end

"GPU-compatible broadcast path for `ParticleStrengthExchange`'s Euler Gamma update."
function _pse_euler_broadcast!(pfield, nu, dt)
    R = eltype(pfield.particles)
    nu = R(nu); dt = R(dt)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)

    for i in 1:3
        G = view(P, GAMMA_INDEX[i], :)
        PSE = view(P, PSE_INDEX[i], :)
        G .= ifelse.(active .> 0, G .+ dt*nu .* PSE, G)
    end

    return nothing
end

"GPU-compatible broadcast path for `ParticleStrengthExchange`'s RK3 Gamma update."
function _pse_rk3_broadcast!(pfield, nu, dt, aux2)
    R = eltype(pfield.particles)
    nu = R(nu); dt = R(dt); aux2 = R(aux2)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)

    for i in 1:3
        M = view(P, M_INDEX[3+i], :)
        G = view(P, GAMMA_INDEX[i], :)
        PSE = view(P, PSE_INDEX[i], :)
        M .= ifelse.(active .> 0, M .+ dt*nu .* PSE, M)
        G .= ifelse.(active .> 0, G .+ aux2*dt*nu .* PSE, G)
    end

    return nothing
end
##### END OF PARTICLE STRENGTH EXCHANGE SCHEME ###################################





# The reset: every active (non-static) particle back to `sgm0` by the Galerkin
# projection (`galerkin_shrink!`), growth timer restarted.
function _corespreading_reset!(pfield, scheme::CoreSpreading)
    _galerkin_reset!(pfield, scheme, findall(iszero, Array(view(pfield.particles, STATIC_INDEX, 1:get_np(pfield)))))
    scheme.t_sgm = 0
    return nothing
end

"""
    corespreading_reset_subset!(pfield, scheme, idx) -> nothing

The core-spreading reset restricted to the particles `idx` (global indices): their
cores go to `scheme.sgm0` by the Galerkin projection (`galerkin_shrink!` with
`scheme.galerkin`); the growth timer is not touched. The particle count changes
(edge points appended at the end, thinned subset particles removed by
swap-with-last), so indices other than the untouched leading ones are not stable
across the call.
"""
function corespreading_reset_subset!(pfield, scheme::CoreSpreading, idx::Vector{Int})
    isempty(idx) || _galerkin_reset!(pfield, scheme, idx)
    return nothing
end

_corespreading_sigma_max(pfield) = pfield.np == 0 ? zero(eltype(pfield.particles)) :
    (pfield.particles isa Array ? maximum(view(pfield.particles, SIGMA_INDEX, 1:pfield.np)) :
                                  _radix_sigma_max(pfield))

##### COMMON FUNCTIONS #########################################################

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

# Integer loops with `(pfield, i)` accessors: Enzyme cannot type the filtered
# particle generator (EnzymeNoTypeError).
function zeta_direct_singlethreaded(pfield)
    np = pfield.np
    for i in 1:np
        get_vorticity(pfield, i) .= 0
    end
    zeta = pfield.kernel.zeta
    for i in 1:np
        for j in 1:np
            dX1 = get_X(pfield, i)[1] - get_X(pfield, j)[1]
            dX2 = get_X(pfield, i)[2] - get_X(pfield, j)[2]
            dX3 = get_X(pfield, i)[3] - get_X(pfield, j)[3]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            zeta_sgm = 1/get_sigma(pfield, j)[]^3*zeta(r/get_sigma(pfield, j)[])

            get_vorticity(pfield, i)[1] += get_Gamma(pfield, j)[1]*zeta_sgm
            get_vorticity(pfield, i)[2] += get_Gamma(pfield, j)[2]*zeta_sgm
            get_vorticity(pfield, i)[3] += get_Gamma(pfield, j)[3]*zeta_sgm
        end
    end
    return nothing
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
