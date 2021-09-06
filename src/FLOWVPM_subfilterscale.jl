#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence schemes for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


################################################################################
# ABSTRACT SFS SCHEME TYPE
################################################################################
abstract type SubFilterScale{R} end

# Make SFS object callable
"""
    Implementation of calculations associated with subfilter-scale turbulence
model.

NOTE: Any implementation is expected to evaluate UJ and SFS terms of the
particles which will be used by the time integration routine so make sure they
are stored in the memory (see implementation of `ConstantSFS` as an example).

NOTE2: Any control strategy is implemented as a function that returns `true`
whenever the SFS model needs to be clipped. Subsequently, the model coefficient
of the targeted particle will be turned to zero.
"""
function (SFS::SubFilterScale)(pfield)
    error("SFS evaluation not implemented!")
end
##### END OF SFS SCHEME ########################################################





################################################################################
# NO SFS SCHEME
################################################################################
struct NoSFS{R} <: SubFilterScale{R} end

function (SFS::NoSFS)(pfield; optargs...)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)
end

"""
Returns true if SFS scheme implements an SFS model
"""
isSFSenabled(SFS::SubFilterScale) = typeof(SFS).name != NoSFS.body.name
##### END OF NO SFS SCHEME #####################################################





################################################################################
# CONSTANT-COEFFICIENT SFS SCHEME
################################################################################
struct ConstantSFS{R} <: SubFilterScale{R}
    model::Function                 # Model of subfilter scale contributions
    Cs::R                           # Model coefficient
    controls::Array{Function, 1}    # Control strategies
    clippings::Array{Function, 1}   # Clipping strategies

    function ConstantSFS{R}(model; Cs=R(1), controls=Function[],
                                            clippings=Function[]) where {R}
        return new(model, Cs, controls, clippings)
    end
end

function ConstantSFS(model; Cs::R=RealFMM(1.0), optargs...) where {R}
    return ConstantSFS{R}(model; Cs, optargs...)
end

function (SFS::ConstantSFS)(pfield; a=1, b=1)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sfs(pfield)
    SFS.model(pfield)

    # Recognize Euler step or Runge-Kutta's first substep
    if a==1 || a==0

        # "Calculate" model coefficient
        for p in iterator(pfield)
            p.C[1] = SFS.Cs
        end

        # Apply clipping strategies
        for clipping in SFS.clippings
            for p in iterator(pfield)

                if clipping(p, pfield)
                    # Clip SFS model by nullifying the model coefficient
                    p.C[1] *= 0
                end

            end
        end

        # Apply control strategies
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
struct DynamicSFS{R} <: SubFilterScale{R}
    model::Function                 # Model of subfilter scale contributions
end

function (SFS::DynamicSFS)(pfield; a=1, b=1)
    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sfs(pfield)
    SFS.model(pfield)

    # "Calculate" model coefficient
    nothing
end
##### END OF DYNAMIC SFS SCHEME ################################################




##### CLIPPING STRATEGIES ######################################################
"""
    Backscatter control strategy of SFS enstrophy production by clipping of the
SFS model. See 20210901 notebook for derivation.
"""
function clipping_backscatter(P::Particle, pfield)
    return P.Gamma[1]*get_SFS1(P) + P.Gamma[2]*get_SFS2(P) + P.Gamma[3]*get_SFS3(P) < 0
end
##### END OF CLIPPING STRATEGIES ###############################################



##### CONTROL STRATEGIES #######################################################
"""
    Directional control strategy of SFS enstrophy production forcing the model
to affect only the vortex strength magnitude and not the vortex orientation.
See 20210901 notebook for derivation.
"""
function control_directional(P::Particle, pfield)

    aux = get_SFS1(P)*P.Gamma[1] + get_SFS2(P)*P.Gamma[2] + get_SFS3(P)*P.Gamma[3]
    aux /= (P.Gamma[1]*P.Gamma[1] + P.Gamma[2]*P.Gamma[2] + P.Gamma[3]*P.Gamma[3])

    # Replaces old SFS with the direcionally controlled SFS
    add_SFS1(P, -get_SFS1(P) + aux*P.Gamma[1])
    add_SFS2(P, -get_SFS2(P) + aux*P.Gamma[2])
    add_SFS3(P, -get_SFS3(P) + aux*P.Gamma[3])
end

"""
    Magnitude control strategy of SFS enstrophy production limiting the
magnitude of the forward scattering (diffussion) of the model.
See 20210901 notebook for derivation.
"""
function control_magnitude(P::Particle{R}, pfield) where {R}

    # Estimate Δt
    if pfield.nt == 0
        # error("Logic error: It was not possible to estimate time step.")
        nothing
    elseif P.C[1] != 0
        deltat::R = pfield.t / pfield.nt

        f::R = pfield.formulation.f
        zeta0::R = pfield.kernel.zeta(0)

        aux = get_SFS1(P)*P.Gamma[1] + get_SFS2(P)*P.Gamma[2] + get_SFS3(P)*P.Gamma[3]
        aux /= P.Gamma[1]*P.Gamma[1] + P.Gamma[2]*P.Gamma[2] + P.Gamma[3]*P.Gamma[3]
        aux -= (1+3*f)*(zeta0/P.sigma[1]^3) / deltat / P.C[1]

        # f_p filter criterion
        if aux > 0
            add_SFS1(P, -aux*P.Gamma[1])
            add_SFS2(P, -aux*P.Gamma[2])
            add_SFS3(P, -aux*P.Gamma[3])
        end
    end
end
##### END OF CONTROL STRATEGIES ################################################
