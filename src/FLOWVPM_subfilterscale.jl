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

        # "Calculate" model coefficient
        for p in iterator(pfield)
            p[37] = SFS.Cs
        end

        # Apply clipping strategies
        for clipping in SFS.clippings
            for p in iterator(pfield)

                if clipping(p, pfield)
                    # Clip SFS model by nullifying the model coefficient
                    p[37] *= 0
                end

            end
        end

        # Apply control strategies
        # NOTE: Shouldn't these strategies applied to every RK substep?
        #       Possibly, but only if they are all continuous (magnitude control
        #       is not).
        for control in SFS.controls
            for p in iterator(pfield)
                control(p, pfield)
            end
        end

    end
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

    function DynamicSFS{R,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}(model, procedure_beforeUJ, procedure_afterUJ;
                            controls=(), clippings=(),
                            alpha=0.667, rlxf=0.005, minC=0, maxC=1) where {R,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}

        return new(model, procedure_beforeUJ, procedure_afterUJ,
                        controls, clippings, alpha, rlxf, minC, maxC)

    end
end

DynamicSFS(model::Tmodel, procedure_beforeUJ::Tpb, procedure_afterUJ::Tpa; 
        controls::Tcontrols=(), clippings::Tclippings=(), optargs...
    ) where {Tmodel,Tpb,Tpa,Tcontrols,Tclippings} = 
        DynamicSFS{FLOAT_TYPE,Tmodel,Tpb,Tpa,Tcontrols,Tclippings}(model, procedure_beforeUJ, procedure_afterUJ; 
            controls=controls, clippings=clippings, optargs...)

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
        SFS.procedure_afterUJ(pfield, SFS, SFS.alpha, SFS.rlxf, SFS.minC, SFS.maxC)

        # Apply clipping strategies
        for clipping in SFS.clippings
            for p in iterator(pfield)

                if clipping(p, pfield)
                    # Clip SFS model by nullifying the model coefficient
                    p[37] *= 0
                end

            end
        end

        # Apply control strategies
        # NOTE: Shouldn't these strategies applied to every RK substep?
        #       Possibly, but only if they are all continuous (magnitude control
        #       is not).
        for control in SFS.controls
            for p in iterator(pfield)
                control(p, pfield)
            end
        end

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
function clipping_backscatter(P, pfield)
    return P[37]*(P[4]*get_SFS1(P) + P[5]*get_SFS2(P) + P[6]*get_SFS3(P)) < 0
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
function control_directional(P, pfield)

    aux = get_SFS1(P)*P[4] + get_SFS2(P)*P[5] + get_SFS3(P)*P[6]
    aux /= (P[4]*P[4] + P[5]*P[5] + P[6]*P[6])

    # Replaces old SFS with the direcionally controlled SFS
    add_SFS1(P, -get_SFS1(P) + aux*P[4])
    add_SFS2(P, -get_SFS2(P) + aux*P[5])
    add_SFS3(P, -get_SFS3(P) + aux*P[6])
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
    elseif P[37] != 0
        deltat::R = pfield.t / pfield.nt

        f::R = pfield.formulation.f
        zeta0::R = pfield.kernel.zeta(0)

        aux = get_SFS1(P)*P[4] + get_SFS2(P)*P[5] + get_SFS3(P)*P[6]
        aux /= P[4]*P[4] + P[5]*P[5] + P[6]*P[6]
        aux -= (1+3*f)*(zeta0/P[7]^3) / deltat / P[37]

        # f_p filter criterion
        if aux > 0
            add_SFS1(P, -aux*P[4])
            add_SFS2(P, -aux*P[5])
            add_SFS3(P, -aux*P[6])
        end
    end
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

    # Storage terms: (Γ⋅∇)dUdσ <=> p.M[:, 1], dEdσ <=> p.M[:, 2],
    #                C=<Γ⋅L>/<Γ⋅m> <=> p[37], <Γ⋅L> <=> p[38], <Γ⋅m> <=> p[39]

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
        p[7] *= alpha
    end

    # Calculate UJ with test filter
    pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

    # Empty temporal memory
    zeroR::R = zero(R)
    for p in iterator(pfield); p.M .= zeroR; end;

    # Calculate stretching and SFS
    for p in iterator(pfield)

        # Calculate and store stretching with test filter under p.M[:, 1]
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p[28] = p[16]*p[4]+p[17]*p[5]+p[18]*p[6]
            p[29] = p[19]*p[4]+p[20]*p[5]+p[21]*p[6]
            p[30] = p[22]*p[4]+p[23]*p[5]+p[24]*p[6]
        else
            # Classic scheme (Γ⋅∇)U
            p[28] = p[16]*p[4]+p[19]*p[5]+p[22]*p[6]
            p[29] = p[17]*p[4]+p[20]*p[5]+p[23]*p[6]
            p[30] = p[18]*p[4]+p[21]*p[5]+p[24]*p[6]
        end

        # Calculate and store SFS with test filter under p.M[:, 2]
        p[31] = get_SFS1(p)
        p[32] = get_SFS2(p)
        p[33] = get_SFS3(p)
    end


    # -------------- CALCULATIONS WITH DOMAIN FILTER WIDTH ---------------------
    # Restore domain filter width
    for p in iterator(pfield)
        p[7] /= alpha
    end

    return nothing
end

function dynamicprocedure_pseudo3level_afterUJ(pfield, SFS::SubFilterScale{R},
                                       alpha::Real, rlxf::Real,
                                       minC::Real, maxC::Real;
                                       force_positive::Bool=false) where {R}

    # Storage terms: (Γ⋅∇)dUdσ <=> p.M[:, 1], dEdσ <=> p.M[:, 2],
    #                C=<Γ⋅L>/<Γ⋅m> <=> p[37], <Γ⋅L> <=> p[38], <Γ⋅m> <=> p[39]

    # ERROR CASES
    if minC < 0
        error("Invalid C bounds: Got a negative bound for minC ($(minC))")
    elseif maxC < 0
            error("Invalid C bounds: Got a negative bound for maxC ($(maxC))")
    elseif minC > maxC
        error("Invalid C bounds: minC > maxC ($(minC) > $(maxC))")
    end

    # Calculate stretching and SFS
    for p in iterator(pfield)

        # Calculate stretching with domain filter and substract from test filter
        # stored under p.M[:, 1], resulting in (Γ⋅∇)dUdσ
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p[28] -= p[16]*p[4]+p[17]*p[5]+p[18]*p[6]
            p[29] -= p[19]*p[4]+p[20]*p[5]+p[21]*p[6]
            p[30] -= p[22]*p[4]+p[23]*p[5]+p[24]*p[6]
        else
            # Classic scheme (Γ⋅∇)U
            p[28] -= p[16]*p[4]+p[19]*p[5]+p[22]*p[6]
            p[29] -= p[17]*p[4]+p[20]*p[5]+p[23]*p[6]
            p[30] -= p[18]*p[4]+p[21]*p[5]+p[24]*p[6]
        end

        # Calculate SFS with domain filter and substract from test filter stored
        # under p.M[:, 2], resulting in dEdσ
        p[31] -= get_SFS1(p)
        p[32] -= get_SFS2(p)
        p[33] -= get_SFS3(p)
    end


    # -------------- CALCULATE COEFFICIENT -------------------------------------
    zeta0::R = pfield.kernel.zeta(0)

    for p in iterator(pfield)

        # Calculate numerator and denominator
        nume = p[28]*p[4] + p[29]*p[5] + p[30]*p[6]
        nume *= 3*alpha - 2
        deno = p[31]*p[4] + p[32]*p[5] + p[33]*p[6]
        deno /= zeta0/p[7]^3

        # Initialize denominator to something other than zero
        if p[39] == 0
            p[39] = deno
        end

        # Lagrangian average of numerator and denominator
        nume = rlxf*nume + (1-rlxf)*p[38]
        deno = rlxf*deno + (1-rlxf)*p[39]

        # Enforce maximum and minimum |C| values
        if abs(nume/deno) > maxC            # Case: C is too large

            # Avoid case of denominator becoming zero
            if abs(deno) < abs(p[39])
                deno = sign(deno) * abs(p[39])
            end

            # Enforce maximum value of |Cd|
            if abs(nume/deno) >= maxC
                nume = sign(nume) * abs(deno) * maxC
            end

        elseif abs(nume/deno) < minC        # Case: C is too small

            # Enforce minimum value of |Cd|
            nume = sign(nume) * abs(deno) * minC

        end

        # Save numerator and denominator of model coefficient
        p[38] = nume
        p[39] = deno

        # Store model coefficient
        p[37] = p[38] / p[39]

        # Force the coefficient to be positive
        p[37] *= sign(p[37])^force_positive
    end

    # Flush temporal memory
    for p in iterator(pfield); p.M .= zero(R); end;

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

    # Storage terms: f(λ) <=> p[37], test-filter ξ <=> p[38], primary-filter ξ <=> p[39]

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
        p[7] *= alpha
    end

    # Calculate UJ with test filter
    pfield.UJ(pfield; sfs=false, reset=true, reset_sfs=false)

    # Store test-filter ξ under p[38]
    for p in iterator(pfield)
        p[38] = get_W1(p)^2 + get_W2(p)^2 + get_W3(p)^2
    end

    # -------------- CALCULATIONS WITH DOMAIN FILTER WIDTH ---------------------
    # Restore domain filter width
    for p in iterator(pfield)
        p[7] /= alpha
    end

    # Calculate UJ with domain filter
    pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

    # Store domain-filter ξ under p[39]
    for p in iterator(pfield)
        p[39] = get_W1(p)^2 + get_W2(p)^2 + get_W3(p)^2
    end

    # -------------- CALCULATE COEFFICIENT -------------------------------------
    for p in iterator(pfield)
        Lmbd = Lambda(p[38]/p[39], lambdacrit)
        p[37] = minC + sensor(Lmbd)*( maxC - minC )
    end

    return nothing
end
##### END OF DYNAMICS PROCEDURES ###############################################
