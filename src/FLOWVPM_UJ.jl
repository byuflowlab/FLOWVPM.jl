#=##############################################################################
# DESCRIPTION
    Particle-to-particle interactions calculation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################


"""
  `UJ_direct(pfield)`

Calculates the velocity and Jacobian that the field exerts on itself by direct
particle-to-particle interaction, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_direct(pfield::ParticleField;
        rbf::Bool=false, sfs::Bool=false,
        reset=true, reset_sfs=false,
        optargs...
    )

    # reset
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs
        _reset_particles_sfs(pfield)
    end

    fmm.direct!(pfield; scalar_potential=false, hessian=true)
    sfs && Estr_direct!(pfield)
    inflow_turbulence(pfield, pfield.inflow_turbulence)
end

"""
  `UJ_direct(source, target)`

Calculates the velocity and Jacobian that the field `source` exerts on every
particle of  field `target`, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_direct(source::ParticleField, target::ParticleField)
    return fmm.direct!(target, source)
end


"""
  `UJ_fmm(pfield)`

Calculates the velocity and Jacobian that the field exerts on itself through
a fast-multipole approximation, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_fmm(
        pfield::ParticleField{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, useGPU};
        verbose::Bool=false,
        rbf::Bool=false,
        sfs::Bool=false,
        sfs_type::Int=-1, # unused
        transposed_sfs::Bool=true, # unused
        reset::Bool=true,
        reset_sfs::Bool=false,
        autotune::Bool=true,
    ) where {useGPU}

    # reset # TODO should this really have an elseif in between?
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs || sfs
        _reset_particles_sfs(pfield)
    end

    # extract FMM options
    fmm_options = pfield.fmm

    if rbf
        # calculate vorticity
        zeta_fmm(pfield)
    else
        # Calculate FMM of vector potential
        args = fmm.fmm!(pfield; 
                        expansion_order=fmm_options.p-1, 
                        leaf_size_source=fmm_options.ncrit, 
                        multipole_acceptance=fmm_options.theta, 
                        error_tolerance=fmm.PowerRelativeGradient{fmm_options.relative_tolerance, fmm_options.absolute_tolerance, true}(), 
                        tune=true,
                        shrink_recenter=fmm_options.shrink_recenter,
                        nearfield_device=(useGPU>0),
                        scalar_potential=false,
                        hessian=true,
                        silence_warnings=!verbose)
        optargs, cache, target_tree, source_tree, m2l_list, direct_list, _ = args

        # autotune p and ncrit
        if autotune
            new_p = fmm_options.autotune_p ? optargs.expansion_order+1 : fmm_options.p
            new_ncrit = fmm_options.autotune_ncrit ? optargs.leaf_size_source[1] : fmm_options.ncrit
            pfield.fmm = FMM(new_p, new_ncrit, fmm_options.theta,
                            fmm_options.shrink_recenter,
                            fmm_options.relative_tolerance,
                            fmm_options.absolute_tolerance,
                            fmm_options.autotune_p,
                            fmm_options.autotune_ncrit,
                            fmm_options.autotune_reg_error,
                            fmm_options.default_rho_over_sigma)
        end

        # This should be concurrent_direct=(pfield.useGPU > 0)
        # But until multithread_direct!() works for the target_indices argument,
        # we'll leave it true

        # now calculate SFS contribution
        # NOTE: this must be performed after velocity gradients are calculated, and
        #       therefore cannot be included in the direct function of the FMM
        sfs && Estr_fmm!(pfield, pfield, target_tree, source_tree, direct_list)
        inflow_turbulence(pfield, pfield.inflow_turbulence)
    end

    return nothing
end
