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
        reset=true, reset_sfs=false, hessian::Bool=false,
        optargs...
    )

    # reset
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs
        _reset_particles_sfs(pfield)
    end

    pfield.toggle_rbf = rbf # if true, computes the direct contribution to the vorticity field computed using the zeta function
    pfield.toggle_sfs = sfs # if true, triggers addition of the SFS model contribution in the direct function

    # TODO: This direct call should be multithreaded but it goes through the FMM
    fmm.direct!(pfield; scalar_potential=false, hessian=(sfs || hessian))
    sfs && Estr_direct!(pfield)
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
        pfield::ParticleField{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, useGPU, <:Any};
        verbose::Bool=false, # unused
        rbf::Bool=false,
        sfs::Bool=false,
        sfs_type::Int=-1, # unused
        transposed_sfs::Bool=true, # unused
        reset::Bool=true,
        reset_sfs::Bool=false,
        hessian::Bool=false,
    ) where {useGPU}

    # reset # TODO should this really have an elseif in between?
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs || sfs
        _reset_particles_sfs(pfield)
    end

    pfield.toggle_rbf = rbf # if true, computes the direct contribution to the vorticity field computed using the zeta function
    pfield.toggle_sfs = sfs # if true, triggers addition of the SFS model contribution in the direct function

    # extract FMM options
    fmm_options = pfield.fmm

    if rbf
        # calculate vorticity
        zeta_fmm(pfield)
    else
        # Calculate FMM of vector potential
        args = fmm.fmm!(pfield; 
                        expansion_order=fmm_options.p-1+!isnothing(fmm_options.ε_tol), 
                        leaf_size_source=fmm_options.ncrit, 
                        multipole_acceptance=fmm_options.theta, 
                        error_tolerance=fmm_options.ε_tol, 
                        shrink_recenter=fmm_options.nonzero_sigma, 
                        nearfield_device=(useGPU>0), 
                        scalar_potential=false,
                        hessian=(hessian || sfs))
        _, _, target_tree, source_tree, m2l_list, direct_list, _ = args

        # This should be concurrent_direct=(pfield.useGPU > 0)
        # But until multithread_direct!() works for the target_indices argument,
        # we'll leave it true

        # now calculate SFS contribution
        # NOTE: this must be performed after velocity gradients are calculated, and
        #       therefore cannot be included in the direct function of the FMM
        sfs && Estr_fmm!(pfield, pfield, target_tree, source_tree, direct_list)
    end

    return nothing
end
