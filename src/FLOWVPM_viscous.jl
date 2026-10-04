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
    precondition::Bool                    # Block-Jacobi preconditioned CG (false: plain CG)
    block_cell::R                         # Preconditioner block cell, in units of sgm0
    block_cap::Int                        # Most particles in one preconditioner block

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
                        itmax=100, tol=R(1e-3),
                        iterror=true, verbose=false, v_lvl=2, debug=false,
                        precondition=true, block_cell=R(8), block_cap=64,
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
                        precondition, block_cell, block_cap,
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

Core spreading assumes a single core size: every particle grows from `sgm0` and a
reset returns every reset particle to `sgm0`. Use it with fields whose particles
are all created at `sgm0`. A field with a range of cores (variable shedding rules,
a hub-to-tip core scaling) has its large cores shrunk and its small ones grown at
each reset, and the RBF problem behind the reset is then poorly posed (on a real
rotor wake reset to its median core, the conjugate gradient stalled near 0.2-0.5
residual).

# Arguments
- `nu`::Real Kinematic viscosity.
- `sgm0`::Real Core size after reset.
- `zeta::Function = zeta_fmm` Basis function evaluation method.
- `beta::Real = 1.5` Maximum core size growth σ/σ_0.
- `itmax::Int = 100` Maximum number of RBF iterations.
- `tol::Real = 1e-3` RBF interpolation tolerance.
- `iterror::Bool = true` Throw error if RBF didn't converge.
- `verbose::Bool = false` Verbose on RBF interpolation.
- `v_lvl::Int = 2` Verbose printing tab level.
- `debug::Bool = false` Print verbose for debugging.
- `precondition::Bool = true` Block-Jacobi preconditioned conjugate gradient:
  the active particles are grouped on a grid of cell `block_cell * sgm0`, cells
  holding more than `block_cap` particles split into octants, and each block's
  dense basis-function matrix is inverted once per reset. On a 20k-particle chunk
  of a rotor wake (subset reset of 1062 grown cores) the plain CG had not reached
  1e-3 after 300 iterations; preconditioned it took 25 in Float64, and in Float32
  reached the ~1e-3 floor of single precision in about 100 (plain: 4-6e-3).
  `false` runs the plain CG.
- `block_cell::Real = 8` Preconditioner block cell, in units of `sgm0`.
- `block_cap::Int = 64` Most particles in one preconditioner block.
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

"GPU-compatible broadcast path for `CoreSpreading`'s core-size reset."
function _corespreading_reset_broadcast!(pfield, sgm0)
    R = eltype(pfield.particles)
    sgm0 = R(sgm0)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= one(eltype(active)) .- view(P, STATIC_INDEX, :)

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

function viscousdiffusion(pfield, scheme::ParticleStrengthExchange, dt; aux1=0, aux2=0)

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
    if pfield.integration == euler

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
    elseif pfield.integration == rungekutta3

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
        error("Time integration scheme $(pfield.integration) not"*
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
    # freeze the rest: zero strength (contributes nothing to A·p), static (out of the CG)
    Grest = copy(view(P, grows, 1:np))
    Srest = copy(view(P, STATIC_INDEX:STATIC_INDEX, 1:np))
    view(P, grows, 1:np) .= zero(R)
    view(P, STATIC_INDEX:STATIC_INDEX, 1:np) .= one(R)
    K = length(idx)
    _radix_oversize_scatter!(P, idx, STATIC_INDEX:STATIC_INDEX, zeros(R, 1, K))
    _radix_oversize_scatter!(P, idx, SIGMA_INDEX:SIGMA_INDEX, fill(R(scheme.sgm0), 1, K))
    _radix_oversize_scatter!(P, idx, M_INDEX[7]:M_INDEX[9], Matrix{R}(Wtot .- Wrest))
    _radix_oversize_scatter!(P, idx, grows, Matrix{R}(Gsub))      # initial search direction: old strengths
    scheme.rbf(pfield, scheme)
    Gnew = _radix_oversize_gather(P, idx, grows)
    # restore the rest, keep the subset's new strengths
    copyto!(view(P, grows, 1:np), Grest)
    copyto!(view(P, STATIC_INDEX:STATIC_INDEX, 1:np), Srest)
    _radix_oversize_scatter!(P, idx, grows, Matrix{R}(Gnew))
    println("  core reset (subset of $K): RBF residual ",
        join((Printf.@sprintf("%.2e", sqrt(scheme.rrs[i] / max(scheme.rr0s[i], eps()))) for i in 1:3), " "),
        " (tol $(scheme.tol), itmax $(scheme.itmax))")
    return nothing
end

_corespreading_sigma_max(pfield) = pfield.np == 0 ? zero(eltype(pfield.particles)) :
    (pfield.particles isa Array ? maximum(view(pfield.particles, SIGMA_INDEX, 1:pfield.np)) :
                                  _radix_sigma_max(pfield))

##### COMMON FUNCTIONS #########################################################

#------- block-Jacobi preconditioner for the CS-RBF conjugate gradient -------#
#
# The RBF matrix A_ij = zeta(|x_i - x_j| / sgm0) / sgm0^3 of a reset is symmetric
# (every reset particle is at sgm0) and badly conditioned where cores overlap
# strongly: on a 5MW wake chunk the plain CG needed more than 300 iterations for
# 1e-3, so the default itmax stopped it near 3e-2 (2026-10-04). The active particles
# are grouped on a grid of cell `block_cell * sgm0`, crowded cells split into
# octants down to `block_cap` particles, and each block's dense matrix inverted
# once per reset (positions and cores do not change during the solve). Applying
# it is one small dense matrix-vector product per particle.

const _CS_BLOCK_SHIFT = Ref(1e-3)

# split `ids` (local indices into the columns of X) into octants of the cell at
# `lo` with edge `h` until each piece holds at most `cap`; coincident points (a
# piece that never splits) are cut into `cap`-sized runs
function _cs_split!(blocks, X, ids, lo, h, cap, depth=0)
    if length(ids) <= cap
        push!(blocks, ids); return blocks
    end
    if depth > 40
        for c in Iterators.partition(ids, cap); push!(blocks, collect(c)); end
        return blocks
    end
    hh = h / 2
    oct = [Int(X[1, k] - lo[1] >= hh) + 2 * Int(X[2, k] - lo[2] >= hh) + 4 * Int(X[3, k] - lo[3] >= hh) for k in ids]
    for o in 0:7
        sub = ids[oct .== o]
        isempty(sub) || _cs_split!(blocks, X, sub, lo .+ hh .* (o & 1, (o >> 1) & 1, (o >> 2) & 1), hh, cap, depth + 1)
    end
    return blocks
end

"""
    _cs_blocks(X, idx, sgm0, zeta, cell, cap, TF) -> NamedTuple

Block-Jacobi preconditioner of the CS-RBF system over the particles `idx` (global
indices; `X` their 3 x K positions, Float64): `perm` the particles block by block,
`start` each block's first position in `perm` (and one past the end), `block_of`
each position's block, `moff` each block's first entry in `minv`, the block
inverses column-major and stored in `TF`.
"""
function _cs_blocks(X::AbstractMatrix{Float64}, idx::Vector{Int}, sgm0, zeta, cell, cap, ::Type{TF}) where TF
    K = length(idx)
    h = Float64(cell) * Float64(sgm0)
    keys = [(floor(Int, X[1, k] / h), floor(Int, X[2, k] / h), floor(Int, X[3, k] / h)) for k in 1:K]
    order = sortperm(keys)
    blocks = Vector{Vector{Int}}()
    i = 1
    while i <= K
        j = i
        while j < K && keys[order[j + 1]] == keys[order[i]]; j += 1; end
        _cs_split!(blocks, X, order[i:j], Float64.(keys[order[i]]) .* h, h, cap)
        i = j + 1
    end
    nb = length(blocks)
    start = Vector{Int32}(undef, nb + 1); moff = Vector{Int}(undef, nb + 1)
    perm = Vector{Int32}(undef, K); block_of = Vector{Int32}(undef, K)
    start[1] = 1; moff[1] = 1
    for (b, blk) in enumerate(blocks)
        n = length(blk)
        start[b + 1] = start[b] + n; moff[b + 1] = moff[b] + n * n
        for (q, k) in enumerate(blk)
            perm[start[b] + q - 1] = idx[k]; block_of[start[b] + q - 1] = b
        end
    end
    minv = Vector{TF}(undef, moff[end] - 1)
    s0 = Float64(sgm0)
    Threads.@threads for b in 1:nb
        blk = blocks[b]; n = length(blk)
        A = [zeta(sqrt((X[1, a] - X[1, c])^2 + (X[2, a] - X[2, c])^2 + (X[3, a] - X[3, c])^2) / s0) / s0^3
             for a in blk, c in blk]
        # a small relative diagonal shift keeps the stored inverse positive definite in
        # Float32 (unshifted blocks reach condition numbers ~1e5 and the Float32 PCG
        # diverged); the preconditioner only approximates A^-1, the solution is unchanged
        for a in 1:n; A[a, a] *= 1 + _CS_BLOCK_SHIFT[]; end
        F = cholesky(Symmetric(A); check = false)
        Ai = issuccess(F) ? inv(F) : [a == c ? 1 / A[a, a] : 0.0 for a in 1:n, c in 1:n]
        copyto!(view(minv, moff[b]:(moff[b + 1] - 1)), vec(Ai))
    end
    return (; perm, start, block_of, moff, minv, nblocks = nb)
end

# the blocks of `pfield`'s active (non-static) particles, for the reset at hand
function _cs_blocks(pfield, cs::CoreSpreading)
    P = pfield.particles; np = pfield.np
    S = Array(view(P, STATIC_INDEX, 1:np)); idx = findall(iszero, S)
    X = Float64.(Array(view(P, X_INDEX, 1:np))[:, idx])
    return _cs_blocks(X, idx, cs.sgm0, pfield.kernel.zeta, cs.block_cell, cs.block_cap, eltype(P))
end

# Z = M^-1 R on the active particles (host); 3 x np matrices
function _cs_block_apply!(Z::AbstractMatrix, R::AbstractMatrix, blk)
    Threads.@threads for b in 1:blk.nblocks
        s = Int(blk.start[b]); n = Int(blk.start[b + 1]) - s; m0 = blk.moff[b]
        for q in 1:n
            z1 = zero(eltype(Z)); z2 = zero(eltype(Z)); z3 = zero(eltype(Z))
            for k in 1:n
                m = blk.minv[m0 + (q - 1) + (k - 1) * n]; j = blk.perm[s + k - 1]
                z1 += m * R[1, j]; z2 += m * R[2, j]; z3 += m * R[3, j]
            end
            i = blk.perm[s + q - 1]
            Z[1, i] = z1; Z[2, i] = z2; Z[3, i] = z3
        end
    end
    return Z
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
    * The basis function evaluation (`omega_cur`) is stored in the vorticity
    field.
    * The solution is built under P.M[1:3] (it used to be x).
    * The current residual is stored under P.M[4:6] (it used to be r).
    =#

    if cs.debug
        println("\t"^(cs.v_lvl+1)*"***** Probe Particle 1 ******\n"*
                "\t"^(cs.v_lvl+2)*"Init Gamma:\t$(round.(get_particle(pfield, 1)[4:6], digits=8))\n"*
                "\t"^(cs.v_lvl+2)*"Target w:\t$(round.(get_particle(pfield, 1)[34:36], digits=8))\n")
    end

    # Initialize memory
    cs.rr0s .= 0
    cs.rrs .= 0
    acc = zeros(Float64, 3)
    cs.flags .= false

    for P in iterator(pfield)
        for i in 1:3
            # Initial guess: Γ_i ≈ ω_i⋅vol_i
            get_M(P)[i] = get_M(P)[6+i]*get_vol(P)[]
            # Sets initial guess as Gamma for vorticity evaluation
            get_Gamma(P)[i] = get_M(P)[i]
        end
    end

    # Current vorticity: evaluate basis functions into the vorticity field.
    cs.zeta(pfield)

    for P in iterator(pfield)
        for i in 1:3
            # Residual of initial guess (r0=b-Ax0)
            get_M(P)[3+i] = get_M(P)[6+i] - get_vorticity(P)[i]    # r = omega_targ - omega_cur

            # Update coefficients
            get_Gamma(P)[i] = get_M(P)[3+i]             # p0 = r0

            # Initial field residual (summed in Float64: the step lengths of a Float32
            # field otherwise lose the digits the preconditioned CG needs)
            acc[i] += Float64(get_M(P)[3+i])^2
        end
    end

    cs.rr0s .= acc; rr0 = copy(acc)
    cs.rrs .= cs.rr0s                         # Current field residuals

    # Block-Jacobi preconditioner (see `_cs_blocks`): z0 = M^-1 r0, p0 = z0, and the
    # step lengths use r.z; without it z = r and this is the plain CG unchanged
    blk = cs.precondition ? _cs_blocks(pfield, cs) : nothing
    rrs = copy(rr0)
    Pm = pfield.particles; npart = pfield.np
    Rv = view(Pm, M_INDEX[4]:M_INDEX[6], 1:npart)
    Z = blk === nothing ? nothing : zeros(eltype(Pm), 3, npart)
    rzs = copy(rr0)
    if blk !== nothing
        _cs_block_apply!(Z, Rv, blk)
        rzs .= 0
        for j in 1:npart
            iszero(Pm[STATIC_INDEX, j]) || continue
            for i in 1:3
                Pm[GAMMA_INDEX[i], j] = Z[i, j]                 # p0 = z0
                rzs[i] += Float64(Rv[i, j]) * Z[i, j]
            end
        end
    end
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
        acc .= 0
        for P in iterator(pfield)
            for i in 1:3
                acc[i] += Float64(get_Gamma(P)[i]) * get_vorticity(P)[i]
            end
        end
        cs.pAps .= acc

        for i in 1:3
            # an inactive component may have pAps = 0: 0/0 * false is NaN (2026-09-26)
            cs.alphas[i] = cs.flags[i] ? rzs[i] / acc[i] : zero(eltype(cs.alphas))
            # cs.alphas[i] = cs.rrs[i]/cs.pAps[i]
        end

        cs.prev_rrs .= cs.rrs; prev_rrs = copy(rrs)
        acc .= 0

        for P in iterator(pfield)
            for i in 1:3
                get_M(P)[i] += cs.alphas[i]*get_Gamma(P)[i]   # x = x + alpha*p
                get_M(P)[i+3] -= cs.alphas[i].*get_vorticity(P)[i] # r = r - alpha*Ap
                acc[i] += Float64(get_M(P)[i+3])^2       # Update field residual
            end
        end

        rrs .= acc; cs.rrs .= acc
        if blk === nothing
            for i in 1:3
                # Avoid dividing by zero
                cs.betas[i] = abs(prev_rrs[i]) <= 2*eps() ? 1 : rrs[i] / prev_rrs[i]
            end
            rzs .= rrs

            for P in iterator(pfield)
                for i in 1:3
                    get_Gamma(P)[i] = get_M(P)[i+3] + cs.betas[i]*get_Gamma(P)[i]
                end
            end
        else
            _cs_block_apply!(Z, Rv, blk)                   # z = M^-1 r
            prev_rzs = copy(rzs); rzs .= 0
            for j in 1:npart
                iszero(Pm[STATIC_INDEX, j]) || continue
                for i in 1:3
                    rzs[i] += Float64(Rv[i, j]) * Z[i, j]
                end
            end
            for i in 1:3
                cs.betas[i] = abs(prev_rzs[i]) <= 2*eps() ? one(eltype(cs.betas)) : rzs[i] / prev_rzs[i]
            end
            for j in 1:npart
                iszero(Pm[STATIC_INDEX, j]) || continue
                for i in 1:3
                    Pm[GAMMA_INDEX[i], j] = Z[i, j] + cs.betas[i] * Pm[GAMMA_INDEX[i], j]   # p = z + beta p
                end
            end
        end

        for i in 1:3
            cs.flags[i] *= abs(rr0[i]) <= 2*eps() ? false : sqrt(rrs[i] / rr0[i]) > cs.tol
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
    for P in iterator(pfield)
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

        for P in iterator(pfield)
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
