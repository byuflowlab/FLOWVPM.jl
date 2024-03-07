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

            dX1 = Pi[1] - Pj[1]
            dX2 = Pi[2] - Pj[2]
            dX3 = Pi[3] - Pj[3]
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3

            if !iszero(r2)
                r = sqrt(r2) 

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = g_dgdr(r/Pj[7])

                # K × Γp
                crss1 = -const4 / r^3 * ( dX2*Pj[6] - dX3*Pj[5] )
                crss2 = -const4 / r^3 * ( dX3*Pj[4] - dX1*Pj[6] )
                crss3 = -const4 / r^3 * ( dX1*Pj[5] - dX2*Pj[4] )

                # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                Pi[10] += g_sgm * crss1
                Pi[11] += g_sgm * crss2
                Pi[12] += g_sgm * crss3

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(Pj[7]*r) - 3*g_sgm /r^2
                # j=1
                Pi[16] += aux * crss1 * dX1
                Pi[17] += aux * crss2 * dX1
                Pi[18] += aux * crss3 * dX1
                # j=2
                Pi[19] += aux * crss1 * dX2
                Pi[20] += aux * crss2 * dX2
                Pi[21] += aux * crss3 * dX2
                # j=3
                Pi[22] += aux * crss1 * dX3
                Pi[23] += aux * crss2 * dX3
                Pi[24] += aux * crss3 * dX3

                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux = - const4 * g_sgm / r^3

                # j=1
                Pi[17] -= aux * Pj[6]
                Pi[18] += aux * Pj[5]
                # j=2
                Pi[19] += aux * Pj[6]
                Pi[21] -= aux * Pj[4]
                # j=3
                Pi[22] -= aux * Pj[5]
                Pi[23] += aux * Pj[4]

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
