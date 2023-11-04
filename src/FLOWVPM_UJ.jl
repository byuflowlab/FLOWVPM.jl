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
    return UJ_direct( iterator(source; include_static=true),
                    iterator(target; include_static=true),
                    source.kernel, source.toggle_sfs, source.transposed)
end

function UJ_direct(sources, targets, kernel::Kernel, toggle_sfs, transposed)
    return UJ_direct(sources, targets, kernel.g_dgdr, kernel.zeta, toggle_sfs, transposed)
end

function UJ_direct(sources, targets, g_dgdr::Function, zeta, toggle_sfs, transposed)

    r = zero(eltype(eltype(sources)))
    for Pi in targets
        for Pj in sources

            dX1 = Pi.X[1] - Pj.X[1]
            dX2 = Pi.X[2] - Pj.X[2]
            dX3 = Pi.X[3] - Pj.X[3]
            r2 = dX1*dX1 + dX2*dX2 + dX3*dX3

            if !iszero(r2)
                r = sqrt(r2) 

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = g_dgdr(r/Pj.sigma[1])

                # K × Γp
                crss1 = -const4 / r^3 * ( dX2*Pj.Gamma[3] - dX3*Pj.Gamma[2] )
                crss2 = -const4 / r^3 * ( dX3*Pj.Gamma[1] - dX1*Pj.Gamma[3] )
                crss3 = -const4 / r^3 * ( dX1*Pj.Gamma[2] - dX2*Pj.Gamma[1] )

                # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                Pi.U[1] += g_sgm * crss1
                Pi.U[2] += g_sgm * crss2
                Pi.U[3] += g_sgm * crss3

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(Pj.sigma[1]*r) - 3*g_sgm /r^2
                # j=1
                Pi.J[1, 1] += aux * crss1 * dX1
                Pi.J[2, 1] += aux * crss2 * dX1
                Pi.J[3, 1] += aux * crss3 * dX1
                # j=2
                Pi.J[1, 2] += aux * crss1 * dX2
                Pi.J[2, 2] += aux * crss2 * dX2
                Pi.J[3, 2] += aux * crss3 * dX2
                # j=3
                Pi.J[1, 3] += aux * crss1 * dX3
                Pi.J[2, 3] += aux * crss2 * dX3
                Pi.J[3, 3] += aux * crss3 * dX3

                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux = - const4 * g_sgm / r^3

                # j=1
                Pi.J[2, 1] -= aux * Pj.Gamma[3]
                Pi.J[3, 1] += aux * Pj.Gamma[2]
                # j=2
                Pi.J[1, 2] += aux * Pj.Gamma[3]
                Pi.J[3, 2] -= aux * Pj.Gamma[1]
                # j=3
                Pi.J[1, 3] -= aux * Pj.Gamma[2]
                Pi.J[2, 3] += aux * Pj.Gamma[1]

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
        transposed_sfs::Bool=true,
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
    fmm.fmm!(pfield; expansion_order=fmm_options.p, n_per_branch=fmm_options.ncrit, theta=fmm_options.theta, ndivisions=100, nearfield=true, farfield=farfield, unsort_bodies=sort, shrink_recenter=false)

    return nothing
end
