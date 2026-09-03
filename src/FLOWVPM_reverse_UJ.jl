
# forward pass: pfield -> pfield
# more specifically, pfield.particles[U] -> pfield.particles[U], pfield.particles[J] -> pfield.particles[J], pfield.particles[S] -> pfield.particles[S], pfield.fmm -> pfield.fmm

function UJ_fmm(
    _pfield::ParticleField{<:ReverseDiff.TrackedReal, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, useGPU};
    verbose::Bool=false,
    rbf::Bool=false,
    sfs::Bool=false,
    sfs_type::Int=-1, # unused
    transposed_sfs::Bool=true, # unused
    reset::Bool=true,
    reset_sfs::Bool=false,
    autotune::Bool=true,
) where {useGPU}

    pfield = ReverseDiff.value(_pfield)

    Ustar = U
    Jstar = J
    Sstar = S

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
                        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit), 
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
                            fmm_options.default_rho_over_sigma,
                            fmm_options.min_ncrit)
        end

        # This should be concurrent_direct=(pfield.useGPU > 0)
        # But until multithread_direct!() works for the target_indices argument,
        # we'll leave it true

        # now calculate SFS contribution
        # NOTE: this must be performed after velocity gradients are calculated, and
        #       therefore cannot be included in the direct function of the FMM
        sfs && Estr_fmm!(pfield, pfield, target_tree, source_tree, direct_list)
    end
    tp = ReverseDiff.tape(pfield)
    record!(tp,
            ReverseDiff.SpecialInstruction,
            UJ_fmm,
            (_pfield, verbose, rbf, sfs, sfs_type, reset, reset_sfs, autotune),
            nothing,
            Ustar, Jstar, Sstar, setting_star)
    return nothing
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(UJ_fmm)})

    pfield, verbose, rbf, sfs, sfs_type, reset, reset_sfs, autotune = instruction
    Ustar, Jstar, Sstar, setting_star = cache

    # reset U, J, S
    pfield.U = Ustar
    pfield.J = Jstar
    pfield.S = Sstar

    # pack pfield (augmented with Ubar and Jbar) into data structures
    augmented_pfield_1 = AugmentedParticleField_U_sigma(pfield, Ubar, Jbar, Sbar)
    augmented_pfield_2 = AugmentedParticleField_X_Gamma(pfield, Ubar, Jbar, Sbar)

    

    # call FMM on this new data structure
    args1 = fmm.fmm!(augmented_pfield_1; 
                        expansion_order=fmm_options.p-1, 
                        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit), 
                        multipole_acceptance=fmm_options.theta, 
                        error_tolerance=fmm.PowerRelativeGradient{fmm_options.relative_tolerance, fmm_options.absolute_tolerance, true}(), 
                        tune=true,
                        shrink_recenter=fmm_options.shrink_recenter,
                        nearfield_device=(useGPU>0),
                        scalar_potential=true,
                        hessian=false,
                        silence_warnings=!verbose)
    optargs, cache, target_tree, source_tree, m2l_list, direct_list, _ = args1

    # call FMM on this new data structure
    args2 = fmm.fmm!(augmented_pfield_1; 
                        expansion_order=fmm_options.p-1, 
                        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit), 
                        multipole_acceptance=fmm_options.theta, 
                        error_tolerance=fmm.PowerRelativeGradient{fmm_options.relative_tolerance, fmm_options.absolute_tolerance, true}(), 
                        tune=true,
                        shrink_recenter=fmm_options.shrink_recenter,
                        nearfield_device=(useGPU>0),
                        scalar_potential=false,
                        hessian=false,
                        silence_warnings=!verbose)
    optargs, cache, target_tree, source_tree, m2l_list, direct_list, _ = args2

    # reset/reset sfs should be handled automatically

    
    return nothing

end

# forward pass is basically the original call but with all the buffers already available.
function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(UJ_fmm)})

    pfield, verbose, rbf, sfs, sfs_type, reset, reset_sfs, autotune = instruction
    Ustar, Jstar, Sstar, setting_star = cache

    # do a normal forward call but with pre-saved settings
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
                        leaf_size_source=max(fmm_options.ncrit, fmm_options.min_ncrit), 
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
                            fmm_options.default_rho_over_sigma,
                            fmm_options.min_ncrit)
        end

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

# things we need:
# augmented pfield, which will probably end up smaller because we can skip things like M
struct AugmentedParticleField{TF}

    particles::AbstractArray
    # other settings needed for the direct interaction

end

# constructor for augmented pfield
function AugmentedParticleField(pfield, Ubar, Jbar, Sbar)



end

# reverse interaction (direct!)
function fmm.direct!(target_system, target_index, derivatives_switch::DerivativesSwitch{PS,VS,GS}, source_system::AugmentedParticleField, source_buffer, source_index) where {PS,VS,GS}

end

# body_to_multipole!
function fmm.body_to_multipole!(system::AugmentedParticleField, args...)
    fmm.body_to_multipole!(fmm.Point{fmm.Vortex}, system, args...)
end

# source_system_to_buffer!
function source_system_to_buffer!(buffer, i_buffer, system::AugmentedParticleField, i_body)

    buffer[1:3, i_buffer] .= system_position[i_body]
    # also need: strength (U, J, Gamma)
    # and sigma

end

# eltype
eltype(apf::AugmentedParticleField{TF}) where TF = TF

# data_per_body
fmm.data_per_body(apf::AugmentedParticleField) = 20

# get_position
fmm.get_position(apf::AugmentedParticleField, i) = apf.particles[X_INDEX, i]

# strength_dims
fmm.strength_dims(apf::AugmentedParticleField) = 15

# get_n_bodies
fmm.get_n_bodies(apf::AugmentedParticleField) = size(apf.particles)[2]

# buffer_to_target_system
function fmm.buffer_to_target_system!(target_system::AugmentedParticleField, i_target, ::FastMultipole.DerivativesSwitch{PS,VS,GS}, target_buffer, i_buffer) where {PS,VS,GS}

    TF = eltype(target_buffer)
    scalar_potential = PS ? FastMultipole.get_scalar_potential(target_buffer, i_buffer) : zero(TF)
    velocity = VS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,TF})
    hessian = GS ? FastMultipole.get_hessian(target_buffer, i_buffer) : zero(SMatrix{3,3,TF,9})

    target_system.potential[i_POTENTIAL[1], i_target] = scalar_potential
    target_system.potential[i_GRADIENT, i_target] .= velocity
    for (jj,j) in enumerate(i_HESSIAN)
        target_system.potential[j, i_target] = hessian[jj]
    end

end