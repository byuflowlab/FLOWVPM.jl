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

    return UJ_direct(pfield, pfield)
end

"""
  `UJ_direct(source, target)`

Calculates the velocity and Jacobian that the field `source` exerts on every
particle of  field `target`, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_direct(source::ParticleField, target::ParticleField)
    return UJ_direct(source, target, source.kernel, source.toggle_sfs, source.transposed)
end

function UJ_direct(sources, targets, kernel::Kernel, toggle_sfs, transposed)
    return UJ_direct(sources, targets, kernel.g_dgdr, kernel.zeta, toggle_sfs, transposed)
end

function UJ_direct(sources, targets, g_dgdr::Function, zeta, toggle_sfs, transposed)

    r = zero(eltype(sources.particles))
    for Pi in iterate(targets)
        for Pj in iterate(sources)

            dX1 = get_X(Pi)[1] - get_X(Pj)[1]
            dX2 = get_X(Pi)[2] - get_X(Pj)[2]
            dX3 = get_X(Pi)[3] - get_X(Pj)[3]
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3

            if !iszero(r2)
                r = sqrt(r2) 

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = g_dgdr(r/get_sigma(Pj)[])

                # K × Γp
                crss1 = -const4 / r^3 * ( dX2*get_Gamma(Pj)[3] - dX3*get_Gamma(Pj)[2] )
                crss2 = -const4 / r^3 * ( dX3*get_Gamma(Pj)[1] - dX1*get_Gamma(Pj)[3] )
                crss3 = -const4 / r^3 * ( dX1*get_Gamma(Pj)[2] - dX2*get_Gamma(Pj)[1] )

                # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                get_U(Pi)[1] += g_sgm * crss1
                get_U(Pi)[2] += g_sgm * crss2
                get_U(Pi)[3] += g_sgm * crss3

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(get_sigma(Pj)[]*r) - 3*g_sgm /r^2
                # j=1
                get_J(Pi)[1] += aux * crss1 * dX1
                get_J(Pi)[2] += aux * crss2 * dX1
                get_J(Pi)[3] += aux * crss3 * dX1
                # j=2
                get_J(Pi)[4] += aux * crss1 * dX2
                get_J(Pi)[5] += aux * crss2 * dX2
                get_J(Pi)[6] += aux * crss3 * dX2
                # j=3
                get_J(Pi)[7] += aux * crss1 * dX3
                get_J(Pi)[8] += aux * crss2 * dX3
                get_J(Pi)[9] += aux * crss3 * dX3

                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux = - const4 * g_sgm / r^3

                # j=1
                get_J(Pi)[2] -= aux * get_Gamma(Pj)[3]
                get_J(Pi)[3] += aux * get_Gamma(Pj)[2]
                # j=2
                get_J(Pi)[4] += aux * get_Gamma(Pj)[3]
                get_J(Pi)[6] -= aux * get_Gamma(Pj)[1]
                # j=3
                get_J(Pi)[7] -= aux * get_Gamma(Pj)[2]
                get_J(Pi)[8] += aux * get_Gamma(Pj)[1]

            end
            if toggle_sfs
                Estr_direct(Pi, Pj, r, zeta, transposed)
            end
        end
    end
    return nothing
end


"""
  `UJ_fmm(pfield)`

Calculates the velocity and Jacobian that the field exerts on itself through
a fast-multipole approximation, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_fmm(pfield::ParticleField; 
        verbose::Bool=false, # unused
        rbf::Bool=false, 
        sfs::Bool=false, 
        sfs_type::Int=-1, # unused
        transposed_sfs::Bool=true, # unused
        reset::Bool=true, 
        reset_sfs::Bool=false,
        sort::Bool=true
    )

    # reset # TODO should this really have an elseif in between?
    if reset
        _reset_particles(pfield)
    end
    if reset_sfs
        _reset_particles_sfs(pfield)
    end

    # define P2P function
    pfield.toggle_rbf = rbf # if true, computes the direct contribution to the vorticity field computed using the zeta function
    pfield.toggle_sfs = sfs # if true, triggers addition of the SFS model contribution in the direct function

    # extract FMM options
    fmm_options = pfield.fmm
    farfield = !rbf

    # Calculate FMM of vector potential
    fmm.fmm!(pfield; expansion_order=fmm_options.p-1, n_per_branch=fmm_options.ncrit, theta=fmm_options.theta, ndivisions=100, nearfield=true, farfield=farfield, unsort_bodies=sort, shrink_recenter=fmm_options.nonzero_sigma)

    return nothing
end
