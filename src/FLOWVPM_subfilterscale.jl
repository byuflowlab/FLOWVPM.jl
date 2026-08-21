#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence schemes for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
=###############################################################################


################################################################################
# DIAGNOSTIC: env-gated per-particle CSV logger for the dynamic procedure
################################################################################
# Set SFS_WATCH_INDICES="3475,3476" (or "100:110" or comma-mix) and
# SFS_WATCH_LOG="debug/logs/sfs_watch.csv" to record, every time
# dynamicprocedure_pseudo3level_afterUJ runs, the raw nume/deno, the
# previous Lagrangian-average state, the clamp path, and the final C[1]
# for each watched particle index. A separate row is emitted from
# `clipping_backscatter` whenever it fires on a watched particle.

const _SFS_WATCH = Base.RefValue{Any}(:uninit)
const _SFS_WATCH_LOCK = ReentrantLock()
const _SFS_WATCH_STEP = Threads.Atomic{Int}(-1)
_sfs_watch_step_tick() = Threads.atomic_add!(_SFS_WATCH_STEP, 1) + 1
_sfs_watch_step_now()  = _SFS_WATCH_STEP[]

function _sfs_watch_state()
    s = _SFS_WATCH[]
    s === :uninit || return s
    spec = get(ENV, "SFS_WATCH_INDICES", "")
    if isempty(spec)
        _SFS_WATCH[] = nothing
        return nothing
    end
    inds = Set{Int}()
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        if occursin(":", tok)
            a, b = split(tok, ":")
            for k in parse(Int, a):parse(Int, b)
                push!(inds, k)
            end
        else
            push!(inds, parse(Int, tok))
        end
    end
    path = get(ENV, "SFS_WATCH_LOG", "debug/logs/sfs_watch.csv")
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, "phase,nt,idx,sigma,G1,G2,G3,",
                "M1,M2,M3,M4,M5,M6,",
                "nume_raw,deno_raw,",
                "C2_old,C3_old,nume_rlx,deno_rlx,",
                "abs_ratio,clamp,C1_pre_force,C1_final,",
                "SFS1,SFS2,SFS3,GdotSFS")
    flush(io)
    _SFS_WATCH[] = (inds, io)
    return _SFS_WATCH[]
end

function _sfs_watch_row(phase, nt, idx, σ, G, M, nume_raw, deno_raw,
                       C2_old, C3_old, nume_rlx, deno_rlx, absratio, clamp,
                       C1_pre, C1_final, SFS)
    s = _sfs_watch_state()
    s === nothing && return
    inds, io = s
    idx in inds || return
    GdotSFS = G[1]*SFS[1] + G[2]*SFS[2] + G[3]*SFS[3]
    lock(_SFS_WATCH_LOCK) do
        print(io, phase, ",", nt, ",", idx, ",", σ)
        for v in G; print(io, ",", v); end
        for j in 1:6; print(io, ",", M[j]); end
        print(io, ",", nume_raw, ",", deno_raw, ",",
                    C2_old, ",", C3_old, ",", nume_rlx, ",", deno_rlx, ",",
                    absratio, ",", clamp, ",", C1_pre, ",", C1_final)
        for v in SFS; print(io, ",", v); end
        println(io, ",", GdotSFS)
        flush(io)
    end
    return nothing
end

################################################################################
# ABSTRACT SFS SCHEME TYPE
################################################################################
abstract type SubFilterScale{R} end

# types for dispatch
struct BeforeUJ end
struct AfterUJ end

# Make SFS object callable
"""
    Implementation of calculations associated with subfilter-scale turbulence
model.

The model is expected to be called in two stages surrounding the calculation of the
induced velocity, as:

```julia
this_sfs_model(pfield::ParticleField, beforeUJ::BeforeUJ)

pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

this_sfs_model(pfield::ParticleField, afterUJ::AfterUJ)
```

(See implementation of `ConstantSFS` as an example.)

NOTE1: The UJ_fmm requires <:SubFilterScale objects to contain a `sfs.model` field,
which is a function that computes the SFS contribution to the stretching term.

NOTE2: Any control strategy is implemented as a function that returns `true`
whenever the SFS model needs to be clipped. Subsequently, the model coefficient
of the targeted particle will be turned to zero.
"""
function (SFS::SubFilterScale)(pfield, ::BeforeUJ)
    error("SFS evaluation not implemented!")
end

function (SFS::SubFilterScale)(pfield, ::AfterUJ)
    error("SFS evaluation not implemented!")
end
##### END OF SFS SCHEME ########################################################





################################################################################
# NO SFS SCHEME
################################################################################
struct NoSFS{R,TM} <: SubFilterScale{R}
    model::TM
end

null_model(args...) = nothing

NoSFS{R}() where R = NoSFS{R,typeof(null_model)}(null_model)

function (SFS::NoSFS)(pfield, ::BeforeUJ; optargs...)
    return nothing
end

function (SFS::NoSFS)(pfield, ::AfterUJ; optargs...)
    return nothing
end

"""
Returns true if SFS scheme implements an SFS model
"""
isSFSenabled(SFS::SubFilterScale) = !(typeof(SFS) <: NoSFS)
##### END OF NO SFS SCHEME #####################################################





################################################################################
# CONSTANT-COEFFICIENT SFS SCHEME
################################################################################
"""
    Subfilter-scale scheme with an associated constant model coefficient.
"""
struct ConstantSFS{R,Tmodel,Tcontrols,Tclippings} <: SubFilterScale{R}
    model::Tmodel                 # Model of subfilter scale contributions
    Cs::R                           # Model coefficient
    controls::Tcontrols    # Control strategies
    clippings::Tclippings   # Clipping strategies

    function ConstantSFS{R,Tmodel,Tcontrols,Tclippings}(model; Cs=R(1), controls=(),
                                            clippings=()) where {R,Tmodel,Tcontrols,Tclippings}
        return new(model, Cs, controls, clippings)
    end
end

function ConstantSFS(model::Tmodel; Cs::R=FLOAT_TYPE(1.0), controls::Tcontrols=(), clippings::Tclippings=()) where {R,Tmodel,Tcontrols,Tclippings}
    return ConstantSFS{R,Tmodel,Tcontrols,Tclippings}(model; Cs=Cs, controls=controls, clippings=clippings)
end

function (SFS::ConstantSFS)(pfield, ::BeforeUJ; a=1, b=1)
    return nothing
end

function (SFS::ConstantSFS)(pfield, ::AfterUJ; a=1, b=1)

    # Recognize Euler step or Runge-Kutta's first substep
    if a==1 || a==0

        if pfield.particles isa Array

            # "Calculate" model coefficient
            for i in 1:pfield.np
                pfield.particles[STATIC_INDEX,i] != 0 && continue
                pfield.particles[C_INDEX[1],i] = SFS.Cs
            end

            # Apply clipping strategies
            for clipping in SFS.clippings
                for i in 1:pfield.np
                    pfield.particles[STATIC_INDEX,i] != 0 && continue

                    if clipping(pfield, i)
                        # Clip SFS model by nullifying the model coefficient
                        pfield.particles[C_INDEX[1],i] = 0
                    end

                end
            end

            # Apply control strategies
            # NOTE: Shouldn't these strategies applied to every RK substep?
            #       Possibly, but only if they are all continuous (magnitude control
            #       is not).
            for control in SFS.controls
                if pfield.np > MIN_MT_NP
                    Threads.@threads for i in 1:pfield.np
                        pfield.particles[STATIC_INDEX,i] != 0 && continue
                        control(pfield, i)
                    end
                else
                    for i in 1:pfield.np
                        pfield.particles[STATIC_INDEX,i] != 0 && continue
                        control(pfield, i)
                    end
                end
            end

        else

            # "Calculate" model coefficient
            _constantsfs_coefficient_broadcast!(pfield, SFS.Cs)

            # Apply clipping strategies
            for clipping in SFS.clippings
                _clip_broadcast!(clipping, pfield)
            end

            # Apply control strategies
            for control in SFS.controls
                _control_broadcast!(control, pfield)
            end

        end

    end
end

"GPU-compatible broadcast path for `ConstantSFS`'s model-coefficient assignment."
function _constantsfs_coefficient_broadcast!(pfield, Cs)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= 1 .- view(P, STATIC_INDEX, :)
    C1 = view(P, C_INDEX[1], :)

    C1 .= ifelse.(active .> 0, Cs, C1)

    return nothing
end
##### END OF CONSTANT SFS SCHEME ###############################################





################################################################################
# DYNAMIC-PROCEDURE SFS SCHEME
################################################################################
"""
    Subfilter-scale scheme with an associated dynamic procedure for calculating
the model coefficient.
"""
struct DynamicSFS{R,Tmodel,Tpb,Tpa,Tcontrols,Tclippings} <: SubFilterScale{R}

    model::Tmodel                 # Model of subfilter scale contributions
    procedure_beforeUJ::Tpb             # Dynamic procedure
    procedure_afterUJ::Tpa             # Dynamic procedure

    controls::Tcontrols    # Control strategies
    clippings::Tclippings   # Clipping strategies

    alpha::R                        # Scaling factor of test filter width
    rlxf::R                         # Relaxation factor for Lagrangian average
    minC::R                         # Minimum value for model coefficient
    maxC::R                         # Maximum value for model coefficient

    function DynamicSFS{R,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}(model, procedure_beforeUJ=dynamicprocedure_pseudo3level_beforeUJ, procedure_afterUJ=dynamicprocedure_pseudo3level_afterUJ;
                            controls=(), clippings=(),
                            alpha=0.667, rlxf=0.005, minC=0, maxC=1) where {R,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}

        return new(model, procedure_beforeUJ, procedure_afterUJ,
                        controls, clippings, alpha, rlxf, minC, maxC)

    end
end

DynamicSFS(model::Tmodel, procedure_beforeUJ::Tpb=dynamicprocedure_pseudo3level_beforeUJ, procedure_afterUJ::Tpa=dynamicprocedure_pseudo3level_afterUJ;
        controls::Tcontrols=(), clippings::Tclippings=(), optargs...
    ) where {Tmodel,Tpb,Tpa,Tcontrols,Tclippings} =
        DynamicSFS{FLOAT_TYPE,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}(model, procedure_beforeUJ, procedure_afterUJ;
            controls=controls, clippings=clippings, optargs...)

DynamicSFS(model, procedures::Tuple; kwargs...) = DynamicSFS(model, procedures...; kwargs...)

function (SFS::DynamicSFS)(pfield, ::BeforeUJ; a=1, b=1)

    # Recognize Euler step or Runge-Kutta's first substep
    if a==1 || a==0

        # Calculate model coefficient through dynamic procedure
        # NOTE: The procedure also calculates UJ and SFS model
        SFS.procedure_beforeUJ(pfield, SFS, SFS.alpha, SFS.rlxf, SFS.minC, SFS.maxC)

    end
end

function (SFS::DynamicSFS)(pfield, ::AfterUJ; a=1, b=1)

    # Recognize Euler step or Runge-Kutta's first substep
    if a==1 || a==0

        # finish dynamic procedure
        # NOTE: procedure_afterUJ (dynamicprocedure_pseudo3level_afterUJ /
        # dynamicprocedure_sensorfunction) interleaves scalar-reduction
        # bookkeeping with `pfield.UJ(...)` N-body re-evaluation at a
        # different filter width -- CPU-only follow-up, same class as the
        # zeta/RBF procedures left CPU-only in FLOWVPM_viscous.jl. Not
        # force-masked into the broadcast fork below.
        SFS.procedure_afterUJ(pfield, SFS, SFS.alpha, SFS.rlxf, SFS.minC, SFS.maxC)

        if pfield.particles isa Array

            # Apply clipping strategies
            for clipping in SFS.clippings
                if pfield.np > MIN_MT_NP
                    Threads.@threads for i in 1:pfield.np
                        # Skip static particles
                        pfield.particles[STATIC_INDEX,i] != 0 && continue

                        fired = clipping(pfield, i)
                        if fired
                            # Clip SFS model by nullifying the model coefficient
                            pfield.particles[C_INDEX[1],i] *= 0
                        end
                        _sfs_watch_clip(pfield, i, fired)
                    end
                else
                    for i in 1:pfield.np
                        # Skip static particles
                        pfield.particles[STATIC_INDEX,i] != 0 && continue

                        fired = clipping(pfield, i)
                        if fired
                            # Clip SFS model by nullifying the model coefficient
                            pfield.particles[C_INDEX[1],i] *= 0
                        end
                        _sfs_watch_clip(pfield, i, fired)
                    end
                end
            end

            # Apply control strategies
            # NOTE: Shouldn't these strategies applied to every RK substep?
            #       Possibly, but only if they are all continuous (magnitude control
            #       is not).
            for control in SFS.controls
                if pfield.np > MIN_MT_NP
                    Threads.@threads for i in 1:pfield.np
                        pfield.particles[STATIC_INDEX,i] != 0 && continue
                        control(pfield, i)
                    end
                else
                    for i in 1:pfield.np
                        pfield.particles[STATIC_INDEX,i] != 0 && continue
                        control(pfield, i)
                    end
                end
            end

        else

            # Apply clipping strategies
            for clipping in SFS.clippings
                _clip_broadcast!(clipping, pfield)
            end

            # Apply control strategies
            for control in SFS.controls
                _control_broadcast!(control, pfield)
            end

        end

    end
end

function _sfs_watch_clip(pfield, i::Int, fired::Bool)
    s = _sfs_watch_state()
    s === nothing && return
    inds, io = s
    i in inds || return
    p = get_particle(pfield, i)
    G = get_Gamma(p)
    SFS = get_SFS(p)
    GdotSFS = G[1]*SFS[1] + G[2]*SFS[2] + G[3]*SFS[3]
    C = get_C(p)
    lock(_SFS_WATCH_LOCK) do
        # phase="clip-fired"/"clip-pass", many fields blank, C1_final and Γ·SFS populated
        print(io, fired ? "clip-fired" : "clip-pass", ",", _sfs_watch_step_now(), ",", i, ",",
                  get_sigma(p)[])
        for v in G; print(io, ",", v); end
        for _ in 1:6; print(io, ","); end  # M empty
        # nume_raw,deno_raw,C2_old,C3_old,nume_rlx,deno_rlx,absratio,clamp,C1_pre,C1_final
        print(io, ",,,,,,,,,", C[1])
        for v in SFS; print(io, ",", v); end
        println(io, ",", GdotSFS)
        flush(io)
    end
end
##### END OF DYNAMIC SFS SCHEME ################################################




##### CLIPPING STRATEGIES ######################################################
# NOTE: Clipping strategies are expected to return `true` to indicate that
#       the model coefficient must be nullified.

"""
    Backscatter control strategy of SFS enstrophy production by clipping of the
SFS model. See 20210901 notebook for derivation.
"""
function clipping_backscatter(P)
    Gamma = get_Gamma(P)
    return get_C(P)[1]*(Gamma[1]*get_SFS1(P) + Gamma[2]*get_SFS2(P) + Gamma[3]*get_SFS3(P)) < 0
end

function clipping_backscatter(pfield, i::Int)
    C = pfield.particles[C_INDEX[1], i]
    G1 = pfield.particles[GAMMA_INDEX[1], i]
    G2 = pfield.particles[GAMMA_INDEX[2], i]
    G3 = pfield.particles[GAMMA_INDEX[3], i]
    S1 = pfield.particles[SFS_INDEX[1], i]
    S2 = pfield.particles[SFS_INDEX[2], i]
    S3 = pfield.particles[SFS_INDEX[3], i]
    return C*(G1*S1 + G2*S2 + G3*S3) < 0
end

"""
    _clip_broadcast!(clipping, pfield)

Dispatches to a whole-field broadcast implementation of a clipping strategy.
A custom clipping strategy used with GPU/`CuArray` particle fields must define
a matching method of `FLOWVPM._clip_broadcast!(::typeof(custom_clipping),
pfield)` (see `_clip_broadcast!(::typeof(clipping_backscatter), pfield)` for
an example) -- same extensibility pattern as `_relax_broadcast!` in
FLOWVPM_relaxation.jl.
"""
_clip_broadcast!(clipping, pfield) = error(
    "No whole-field broadcast implementation registered for clipping strategy `$clipping`. " *
    "Since GPU particle fields run clipping strategies through `_clip_broadcast!` (no " *
    "per-particle CPU loop fallback), a custom clipping strategy must define a matching " *
    "method of `FLOWVPM._clip_broadcast!(::typeof($clipping), pfield)`.")

"GPU-compatible broadcast path for `clipping_backscatter`."
function _clip_broadcast!(::typeof(clipping_backscatter), pfield)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= 1 .- view(P, STATIC_INDEX, :)
    C1 = view(P, C_INDEX[1], :)
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)
    S1, S2, S3 = view(P, SFS_INDEX[1], :), view(P, SFS_INDEX[2], :), view(P, SFS_INDEX[3], :)

    clip = view(Sc, 2, :)
    clip .= active .* (C1 .* (G1.*S1 .+ G2.*S2 .+ G3.*S3) .< 0)

    C1 .= ifelse.(clip .> 0, zero(eltype(C1)), C1)

    return nothing
end
##### END OF CLIPPING STRATEGIES ###############################################



##### CONTROL STRATEGIES #######################################################
# NOTE: Control strategies are expected to modify either SFS term or the model
#       model coefficient directly, or both.

"""
    Directional control strategy of SFS enstrophy production forcing the model
to affect only the vortex strength magnitude and not the vortex orientation.
See 20210901 notebook for derivation.
"""
function control_directional(P)

    aux = get_SFS1(P)*get_Gamma(P)[1] + get_SFS2(P)*get_Gamma(P)[2] + get_SFS3(P)*get_Gamma(P)[3]
    aux /= (get_Gamma(P)[1]*get_Gamma(P)[1] + get_Gamma(P)[2]*get_Gamma(P)[2] + get_Gamma(P)[3]*get_Gamma(P)[3])

    # Replaces old SFS with the direcionally controlled SFS
    get_SFS(P) .= aux*get_Gamma(P)
end

function control_directional(pfield, i::Int)
    G1 = pfield.particles[GAMMA_INDEX[1], i]
    G2 = pfield.particles[GAMMA_INDEX[2], i]
    G3 = pfield.particles[GAMMA_INDEX[3], i]
    S1 = pfield.particles[SFS_INDEX[1], i]
    S2 = pfield.particles[SFS_INDEX[2], i]
    S3 = pfield.particles[SFS_INDEX[3], i]

    aux = S1*G1 + S2*G2 + S3*G3
    aux /= (G1*G1 + G2*G2 + G3*G3)

    # Replaces old SFS with the direcionally controlled SFS
    pfield.particles[SFS_INDEX[1], i] = aux*G1
    pfield.particles[SFS_INDEX[2], i] = aux*G2
    pfield.particles[SFS_INDEX[3], i] = aux*G3
end

"""
    Forward-scatter projection control: keep SFS unchanged where the model
forward-scatters (Γ·SFS ≥ 0), and subtract only the backscatter-pointing
parallel component otherwise. The perpendicular (vortex-tilting) component
is preserved in either case. Equivalent to projecting SFS onto the
half-space {v : Γ·v ≥ 0}.

Softer alternative to `clipping_backscatter`: where the clip nullifies the
dynamic coefficient (removing all SFS regularization on backscatter
particles), this control preserves SFS magnitude in the dissipative and
rotational directions and only removes the directional component that
would amplify |Γ|.
"""
function control_no_backscatter_projection(P)
    Γ = get_Gamma(P)
    SFS = get_SFS(P)
    g2 = Γ[1]*Γ[1] + Γ[2]*Γ[2] + Γ[3]*Γ[3]
    g2 > 0 || return
    aux = (SFS[1]*Γ[1] + SFS[2]*Γ[2] + SFS[3]*Γ[3]) / g2
    if aux < 0
        SFS[1] -= aux*Γ[1]
        SFS[2] -= aux*Γ[2]
        SFS[3] -= aux*Γ[3]
    end
end

function control_no_backscatter_projection(pfield, i::Int)
    G1 = pfield.particles[GAMMA_INDEX[1], i]
    G2 = pfield.particles[GAMMA_INDEX[2], i]
    G3 = pfield.particles[GAMMA_INDEX[3], i]
    g2 = G1*G1 + G2*G2 + G3*G3
    g2 > 0 || return
    S1 = pfield.particles[SFS_INDEX[1], i]
    S2 = pfield.particles[SFS_INDEX[2], i]
    S3 = pfield.particles[SFS_INDEX[3], i]
    aux = (S1*G1 + S2*G2 + S3*G3) / g2
    if aux < 0
        pfield.particles[SFS_INDEX[1], i] = S1 - aux*G1
        pfield.particles[SFS_INDEX[2], i] = S2 - aux*G2
        pfield.particles[SFS_INDEX[3], i] = S3 - aux*G3
    end
end

"""
    _control_broadcast!(control, pfield)

Dispatches to a whole-field broadcast implementation of a control strategy.
A custom control strategy used with GPU/`CuArray` particle fields must define
a matching method of `FLOWVPM._control_broadcast!(::typeof(custom_control),
pfield)` -- same extensibility pattern as `_clip_broadcast!` above and
`_relax_broadcast!` in FLOWVPM_relaxation.jl.
"""
_control_broadcast!(control, pfield) = error(
    "No whole-field broadcast implementation registered for control strategy `$control`. " *
    "Since GPU particle fields run control strategies through `_control_broadcast!` (no " *
    "per-particle CPU loop fallback), a custom control strategy must define a matching " *
    "method of `FLOWVPM._control_broadcast!(::typeof($control), pfield)`.")

"GPU-compatible broadcast path for `control_directional`."
function _control_broadcast!(::typeof(control_directional), pfield)
    P = pfield.particles
    Sc = pfield.scratch

    active = view(Sc, 1, :); active .= 1 .- view(P, STATIC_INDEX, :)
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)
    S1, S2, S3 = view(P, SFS_INDEX[1], :), view(P, SFS_INDEX[2], :), view(P, SFS_INDEX[3], :)

    aux = view(Sc, 2, :)
    aux .= (S1.*G1 .+ S2.*G2 .+ S3.*G3) ./ (G1.^2 .+ G2.^2 .+ G3.^2)

    S1 .= ifelse.(active .> 0, aux .* G1, S1)
    S2 .= ifelse.(active .> 0, aux .* G2, S2)
    S3 .= ifelse.(active .> 0, aux .* G3, S3)

    return nothing
end

"""
    Magnitude control strategy of SFS enstrophy production limiting the
magnitude of the forward scattering (diffussion) of the model.
See 20210901 notebook for derivation.
"""
function control_magnitude(P, pfield)

    # Estimate Δt
    if pfield.nt == 0
        # error("Logic error: It was not possible to estimate time step.")
        nothing
    elseif get_C(P)[1] != 0
        deltat::Real = pfield.t / pfield.nt

        f::Real = pfield.formulation.f
        zeta0::Real = pfield.kernel.zeta(0)

        SFS = get_SFS(P)

        aux = SFS[1]*get_Gamma(P)[1] + SFS[2]*get_Gamma(P)[2] + SFS[3]*get_Gamma(P)[3]
        aux /= get_Gamma(P)[1]*get_Gamma(P)[1] + get_Gamma(P)[2]*get_Gamma(P)[2] + get_Gamma(P)[3]*get_Gamma(P)[3]
        aux -= (1+3*f)*(zeta0/get_sigma(P)[]^3) / deltat / get_C(P)[1]

        # f_p filter criterion
        if aux > 0
            SFS .+= -aux .* get_Gamma(P)
        end
    end
end

function control_magnitude(pfield, i::Int)
    C = pfield.particles[C_INDEX[1], i]

    # Estimate Δt
    if pfield.nt == 0
        # error("Logic error: It was not possible to estimate time step.")
        nothing
    elseif C != 0
        deltat::Real = pfield.t / pfield.nt

        f::Real = pfield.formulation.f
        zeta0::Real = pfield.kernel.zeta(0)

        G1 = pfield.particles[GAMMA_INDEX[1], i]
        G2 = pfield.particles[GAMMA_INDEX[2], i]
        G3 = pfield.particles[GAMMA_INDEX[3], i]
        S1 = pfield.particles[SFS_INDEX[1], i]
        S2 = pfield.particles[SFS_INDEX[2], i]
        S3 = pfield.particles[SFS_INDEX[3], i]

        aux = S1*G1 + S2*G2 + S3*G3
        aux /= (G1*G1 + G2*G2 + G3*G3)
        aux -= (1+3*f)*(zeta0/pfield.particles[SIGMA_INDEX, i]^3) / deltat / C

        # f_p filter criterion
        if aux > 0
            pfield.particles[SFS_INDEX[1], i] = -aux*G1
            pfield.particles[SFS_INDEX[2], i] = -aux*G2
            pfield.particles[SFS_INDEX[3], i] = -aux*G3
        end
    end
end

"GPU-compatible broadcast path for `control_magnitude`."
function _control_broadcast!(::typeof(control_magnitude), pfield)
    pfield.nt == 0 && return nothing

    P = pfield.particles
    Sc = pfield.scratch

    deltat = pfield.t / pfield.nt
    f = pfield.formulation.f
    zeta0 = pfield.kernel.zeta(0)

    active = view(Sc, 1, :); active .= 1 .- view(P, STATIC_INDEX, :)
    C1 = view(P, C_INDEX[1], :)
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)
    S1, S2, S3 = view(P, SFS_INDEX[1], :), view(P, SFS_INDEX[2], :), view(P, SFS_INDEX[3], :)
    sigma = view(P, SIGMA_INDEX, :)

    nonzeroC = view(Sc, 2, :); nonzeroC .= active .* (C1 .!= 0)
    safeC = view(Sc, 3, :); safeC .= ifelse.(C1 .!= 0, C1, one(eltype(C1)))

    aux = view(Sc, 4, :)
    aux .= (S1.*G1 .+ S2.*G2 .+ S3.*G3) ./ (G1.^2 .+ G2.^2 .+ G3.^2)
    aux .-= (1+3*f) .* (zeta0 ./ sigma.^3) ./ deltat ./ safeC

    apply = view(Sc, 5, :); apply .= nonzeroC .* (aux .> 0)

    S1 .= ifelse.(apply .> 0, -aux .* G1, S1)
    S2 .= ifelse.(apply .> 0, -aux .* G2, S2)
    S3 .= ifelse.(apply .> 0, -aux .* G3, S3)

    return nothing
end
##### END OF CONTROL STRATEGIES ################################################



##### DYNAMICS PROCEDURES ######################################################
# NOTE: Dynamic procedures are expected to calculate the model coefficient of
#       each particle
# NOTE 2: All dynamic procedures are expected to evaluate UJ and SFS terms at
#       the domain filter scale, which will be used by the time integration
#       routine so make sure they are stored in the memory (see implementation
#       of `ConstantSFS` as an example).

"""
    Dynamic procedure for SFS model coefficient based on enstrophy and
derivative balance between resolved and unresolved domain, numerically
implemented through pseudo-three filtering levels. See 20210901 notebook for
derivation.

# NOTES
* `rlxf` = Δ𝑡/𝑇 ≤ 1 is the relaxation factor of the Lagrangian average, where Δ𝑡
is the time step of the simulation, and 𝑇 is the time length of the ensemble
average.

* The scaling constant becomes 1 for \$\\alpha_\\tau = 1\$ (but notice that the
derivative approximation becomes zero at that point). Hence, the
pseudo-three-level procedure converges to the two-level procedure for
\$\\alpha_\\tau \\rightarrow 1\$**.

* The scaling constant tends to zero when \$\\alpha_\\tau \\rightarrow 2/3\$. Hence,
it can be used to arbitrarely attenuate the SFS contributions with \$\\alpha_\\tau
\\rightarrow 2/3\$, or let it trully be a self-regulated dynamic procedure with
\$\\alpha_\\tau \\rightarrow 1\$.

* \$\\alpha_\\tau\$ should not be made smaller than \$2/3\$ as the constant becomes
negative beyond that point. This strains the assumption that \$\\sigma_\\tau\$ is
small enough to approximate the singular velocity field as \$\\mathbf{u} \\approx
\\mathbf{\\tilde{u}}\$, which now is only true if \$\\sigma\$ is small enough.

𝛼𝜏=0.999 ⇒ 3𝛼𝜏−2=0.997
𝛼𝜏=0.990 ⇒ 3𝛼𝜏−2=0.970
𝛼𝜏=0.900 ⇒ 3𝛼𝜏−2=0.700
𝛼𝜏=0.833 ⇒ 3𝛼𝜏−2=0.499
𝛼𝜏=0.750 ⇒ 3𝛼𝜏−2=0.250
𝛼𝜏=0.700 ⇒ 3𝛼𝜏−2=0.100
𝛼𝜏=0.675 ⇒ 3𝛼𝜏−2=0.025
𝛼𝜏=0.670 ⇒ 3𝛼𝜏−2=0.010
𝛼𝜏=0.667 ⇒ 3𝛼𝜏−2=0.001
𝛼𝜏=0.6667⇒ 3𝛼𝜏−2=0.0001
"""
function dynamicprocedure_pseudo3level_beforeUJ(pfield, SFS::SubFilterScale{R},
                                       alpha::Real, rlxf::Real,
                                       minC::Real, maxC::Real) where {R}
    haskey(ENV, "SFS_WATCH_INDICES") && println("[sfs_watch] beforeUJ called, np=", pfield.np)

    # Storage terms: (Γ⋅∇)dUdσ <=> p.M[:, 1], dEdσ <=> p.M[:, 2],
    #                C=<Γ⋅L>/<Γ⋅m> <=> get_C(P)[1], <Γ⋅L> <=> get_C(p)[2], <Γ⋅m> <=> get_C(p)[3]

    # ERROR CASES
    if minC < 0
        error("Invalid C bounds: Got a negative bound for minC ($(minC))")
    elseif maxC < 0
            error("Invalid C bounds: Got a negative bound for maxC ($(maxC))")
    elseif minC > maxC
        error("Invalid C bounds: minC > maxC ($(minC) > $(maxC))")
    end

    # -------------- CALCULATIONS WITH TEST FILTER WIDTH -----------------------
    # Replace domain filter width with test filter width
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[SIGMA_INDEX,i] *= alpha
        end
    else
        for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[SIGMA_INDEX,i] *= alpha
        end
    end

    # Calculate UJ with test filter
    pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

    # Empty temporal memory
    zeroR::R = zero(R)
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
        end
    else
        for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
        end
    end

    # Calculate stretching and SFS
    Threads.@threads for i in 1:pfield.np
        p = get_particle(pfield, i)
        # Skip static particles
        pfield.particles[STATIC_INDEX,i] != 0 && continue

        M = get_M(p)
        J = get_J(p)
        Gamma = get_Gamma(p)

        # Calculate and store stretching with test filter under p.M[:, 1]
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            M[1] = J[1]*Gamma[1]+J[2]*Gamma[2]+J[3]*Gamma[3]
            M[2] = J[4]*Gamma[1]+J[5]*Gamma[2]+J[6]*Gamma[3]
            M[3] = J[7]*Gamma[1]+J[8]*Gamma[2]+J[9]*Gamma[3]
        else
            # Classic scheme (Γ⋅∇)U
            M[1] = J[1]*Gamma[1]+J[4]*Gamma[2]+J[7]*Gamma[3]
            M[2] = J[2]*Gamma[1]+J[5]*Gamma[2]+J[8]*Gamma[3]
            M[3] = J[3]*Gamma[1]+J[6]*Gamma[2]+J[9]*Gamma[3]
        end

        # Calculate and store SFS with test filter under p.M[:, 2]
        M[4] = get_SFS1(p)
        M[5] = get_SFS2(p)
        M[6] = get_SFS3(p)
    end


    # -------------- CALCULATIONS WITH DOMAIN FILTER WIDTH ---------------------
    # Restore domain filter width
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[SIGMA_INDEX,i] /= alpha
        end
    else
        for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[SIGMA_INDEX,i] /= alpha
        end
    end

    _sfs_watch_beforeUJ_dump(pfield)

    return nothing
end

function _sfs_watch_beforeUJ_dump(pfield)
    s = _sfs_watch_state()
    s === nothing && return
    inds, io = s
    lock(_SFS_WATCH_LOCK) do
        for i in 1:pfield.np
            i in inds || continue
            p = get_particle(pfield, i)
            M = get_M(p)
            G = get_Gamma(p)
            σ = get_sigma(p)[]
            print(io, "beforeUJ_end,", _sfs_watch_step_now(), ",", i, ",", σ)
            for v in G; print(io, ",", v); end
            for j in 1:6; print(io, ",", M[j]); end
            print(io, ",,,,,,,,,,")  # nume_raw..C1_final fields blank
            print(io, ",,,,")        # SFS fields blank
            println(io)
        end
        flush(io)
    end
end

function dynamicprocedure_pseudo3level_afterUJ(pfield, SFS::SubFilterScale{R},
                                       alpha::Real, rlxf::Real,
                                       minC::Real, maxC::Real;
                                       force_positive::Bool=false) where {R}

    # Storage terms: (Γ⋅∇)dUdσ <=> p.M[:, 1], dEdσ <=> p.M[:, 2],
    #                C=<Γ⋅L>/<Γ⋅m> <=> get_C(P)[1], <Γ⋅L> <=> get_C(p)[2], <Γ⋅m> <=> get_C(p)[3]

    # ERROR CASES
    if minC < 0
        error("Invalid C bounds: Got a negative bound for minC ($(minC))")
    elseif maxC < 0
            error("Invalid C bounds: Got a negative bound for maxC ($(maxC))")
    elseif minC > maxC
        error("Invalid C bounds: minC > maxC ($(minC) > $(maxC))")
    end

    # Calculate stretching and SFS
    Threads.@threads for i in 1:pfield.np
        p = get_particle(pfield, i)
        # Skip static particles
        is_static(p) && continue
        M = get_M(p)
        J = get_J(p)
        Gamma = get_Gamma(p)

        # Calculate stretching with domain filter and substract from test filter
        # stored under p.M[:, 1], resulting in (Γ⋅∇)dUdσ
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            M[1] -= J[1]*Gamma[1]+J[2]*Gamma[2]+J[3]*Gamma[3]
            M[2] -= J[4]*Gamma[1]+J[5]*Gamma[2]+J[6]*Gamma[3]
            M[3] -= J[7]*Gamma[1]+J[8]*Gamma[2]+J[9]*Gamma[3]
        else
            # Classic scheme (Γ⋅∇)U
            M[1] -= J[1]*Gamma[1]+J[4]*Gamma[2]+J[7]*Gamma[3]
            M[2] -= J[2]*Gamma[1]+J[5]*Gamma[2]+J[8]*Gamma[3]
            M[3] -= J[3]*Gamma[1]+J[6]*Gamma[2]+J[9]*Gamma[3]
        end

        # Calculate SFS with domain filter and substract from test filter stored
        # under p.M[:, 2], resulting in dEdσ
        M[4] -= get_SFS1(p)
        M[5] -= get_SFS2(p)
        M[6] -= get_SFS3(p)
    end


    # -------------- CALCULATE COEFFICIENT -------------------------------------
    zeta0::R = pfield.kernel.zeta(0)
    _sfs_step = _sfs_watch_step_tick()

    Threads.@threads for i in 1:pfield.np
        p = get_particle(pfield, i)
        # Skip static particles
        is_static(p) && continue
        M = get_M(p)
        C_p = get_C(p)
        Gamma = get_Gamma(p)

        # Calculate numerator and denominator
        nume = M[1]*Gamma[1] + M[2]*Gamma[2] + M[3]*Gamma[3]
        nume *= 3*alpha - 2
        deno = M[4]*Gamma[1] + M[5]*Gamma[2] + M[6]*Gamma[3]
        deno /= zeta0/get_sigma(p)[]^3

        # Snapshot raw values + prior Lagrangian state for diagnostics.
        nume_raw = nume
        deno_raw = deno
        C2_old = C_p[2]
        C3_old = C_p[3]

        # Initialize denominator to something other than zero
        if C_p[3] == 0
            C_p[3] = deno
            if C_p[3] == 0
                C_p[3] = eps()
            end
        end

        # Lagrangian average of numerator and denominator
        nume = rlxf*nume + (1-rlxf)*C_p[2]
        deno = rlxf*deno + (1-rlxf)*C_p[3]

        nume_rlx = nume
        deno_rlx = deno
        absratio = abs(nume/deno)
        clamp_path = 0  # 0=none, 1=maxC, 2=maxC+deno_bump, 3=minC

        # Enforce maximum and minimum |C| values
        if abs(nume/deno) > maxC            # Case: C is too large
            clamp_path = 1
            # Avoid case of denominator becoming zero
            if abs(deno) < abs(C_p[3])
                deno = sign(deno) * abs(C_p[3])
                clamp_path = 2
            end

            # Enforce maximum value of |Cd|
            if abs(nume/deno) >= maxC
                nume = sign(nume) * abs(deno) * maxC
            end

        elseif abs(nume/deno) < minC        # Case: C is too small
            clamp_path = 3
            # Enforce minimum value of |Cd|
            nume = sign(nume) * abs(deno) * minC

        end

        # Save numerator and denominator of model coefficient
        C_p[2] = nume
        C_p[3] = deno

        # Store model coefficient
        C_p[1] = C_p[2] / C_p[3]

        if isnan(C_p[1])
            println("nume: ", nume)
            println("deno: ", deno)
            println("M: ", M)
            println("Gamma: ", Gamma)
            println("J: ", get_J(p))
            error("NaN in dynamicprocedure_pseudo3level_afterUJ")
        end

        C1_pre = C_p[1]
        # Force the coefficient to be positive
        C_p[1] *= sign(C_p[1])^force_positive

        _sfs_watch_row("dyn", _sfs_step, i, get_sigma(p)[], Gamma, M,
                       nume_raw, deno_raw, C2_old, C3_old,
                       nume_rlx, deno_rlx, absratio, clamp_path,
                       C1_pre, C_p[1], get_SFS(p))
    end

    # Flush temporal memory
    zeroR::R = zero(R)
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
        end
    else
        for i in 1:pfield.np
            pfield.particles[STATIC_INDEX,i] != 0 && continue
            pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
        end
    end

    return nothing
end


"""
    Dynamic procedure for SFS model coefficient based on sensor function of
enstrophy between resolved and unresolved domain, numerically
implemented through a test filter. See 20210901 notebook for derivation.
"""
function dynamicprocedure_sensorfunction(pfield, SFS::SubFilterScale{R},
                                           alpha::Real, lambdacrit::Real,
                                           minC::Real, maxC::Real;
                                           sensor=Lmbd->Lmbd < 0 ? 1 : Lmbd <= 1 ? 0.5*(1 + sin(pi/2 - Lmbd*pi)) : 0,
                                           Lambda=(lmbd, lmbdcrit) -> (lmbd - lmbdcrit) / (1 - lmbdcrit)
                                         ) where {R}

    # Storage terms: f(λ) <=> get_C(p)[1], test-filter ξ <=> get_C(p)[2], primary-filter ξ <=> get_C(p)[3]

    # ERROR CASES
    if minC < 0
        error("Invalid C bounds: Got a negative bound for minC ($(minC))")
    elseif maxC < 0
            error("Invalid C bounds: Got a negative bound for maxC ($(maxC))")
    elseif minC > maxC
        error("Invalid C bounds: minC > maxC ($(minC) > $(maxC))")
    end

    # -------------- CALCULATIONS WITH TEST FILTER WIDTH -----------------------
    # Replace domain filter width with test filter width
    for p in iterator(pfield)
        get_sigma(p)[] *= alpha
    end

    # Calculate UJ with test filter
    pfield.UJ(pfield; sfs=false, reset=true, reset_sfs=false)

    # Store test-filter ξ under get_C(p)[2]
    for p in iterator(pfield)
        get_C(p)[2] = get_W1(p)^2 + get_W2(p)^2 + get_W3(p)^2
    end

    # -------------- CALCULATIONS WITH DOMAIN FILTER WIDTH ---------------------
    # Restore domain filter width
    for p in iterator(pfield)
        get_sigma(p)[] /= alpha
    end

    # Calculate UJ with domain filter
    pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

    # Store domain-filter ξ under get_C(p)[3]
    for p in iterator(pfield)
        get_C(p)[3] = get_W1(p)^2 + get_W2(p)^2 + get_W3(p)^2
    end

    # -------------- CALCULATE COEFFICIENT -------------------------------------
    for p in iterator(pfield)
        Lmbd = Lambda(get_C(p)[2]/get_C(p)[3], lambdacrit)
        get_C(p)[1] = minC + sensor(Lmbd)*( maxC - minC )
    end

    return nothing
end
##### END OF DYNAMICS PROCEDURES ###############################################
