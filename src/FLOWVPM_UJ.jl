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

    pfield.toggle_rbf = rbf # if true, computes the direct contribution to the vorticity field computed using the zeta function
    pfield.toggle_sfs = sfs # if true, triggers addition of the SFS model contribution in the direct function

    # return UJ_direct(pfield, pfield)
    fmm.direct!(pfield)
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
        sort::Bool=true
    ) where {useGPU}

    # reset # TODO should this really have an elseif in between?
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs
        _reset_particles_sfs(pfield)
    end

    # define P2P function
    pfield.toggle_rbf = rbf # if true, computes the direct contribution to the vorticity field computed using the zeta function

    # extract FMM options
    fmm_options = pfield.fmm
    farfield = !rbf
    #@show typeof(pfield)

    # Calculate FMM of vector potential
    pfield.toggle_sfs = false
    fmm.fmm!(pfield; expansion_order=fmm_options.p-1, leaf_size=fmm_options.ncrit, multipole_threshold=fmm_options.theta, ε_tol=fmm_options.ε_tol, nearfield=true, farfield=farfield, unsort_bodies=sort, shrink_recenter=fmm_options.nonzero_sigma, lamb_helmholtz=true, nearfield_device=(useGPU>0))

    # SFS contribution
    if sfs
        pfield.toggle_sfs = true
        fmm.fmm!(pfield; expansion_order=fmm_options.p-1, leaf_size=fmm_options.ncrit, multipole_threshold=fmm_options.theta, ε_tol=fmm_options.ε_tol, nearfield=true, farfield=false, unsort_bodies=sort, shrink_recenter=fmm_options.nonzero_sigma, lamb_helmholtz=true, nearfield_device=(useGPU>0))
        pfield.toggle_sfs = false
    end


    # This should be concurrent_direct=(pfield.useGPU > 0)
    # But until multithread_direct!() works for the target_indices argument,
    # we'll leave it true

    return nothing
end
