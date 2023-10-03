#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence models for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
=###############################################################################


"""
    Model of vortex-stretching SFS contributions evaluated with direct
particle-to-particle interactions. See 20210901 notebook for derivation.
"""
function Estr_direct(pfield::ParticleField)
  return Estr_direct(   iterator(pfield; include_static=true),
                        iterator(pfield; include_static=true),
                        pfield.kernel.zeta, pfield.transposed)
end

function Estr_direct(sources, targets, zeta, transposed)

    for p in targets
        for q in sources

            # Stretching term
            if transposed
                # Transposed scheme (Γq⋅∇')(Up - Uq)
                S1 = (p.J[1,1] - q.J[1,1])*q.Gamma[1]+(p.J[2,1] - q.J[2,1])*q.Gamma[2]+(p.J[3,1] - q.J[3,1])*q.Gamma[3]
                S2 = (p.J[1,2] - q.J[1,2])*q.Gamma[1]+(p.J[2,2] - q.J[2,2])*q.Gamma[2]+(p.J[3,2] - q.J[3,2])*q.Gamma[3]
                S3 = (p.J[1,3] - q.J[1,3])*q.Gamma[1]+(p.J[2,3] - q.J[2,3])*q.Gamma[2]+(p.J[3,3] - q.J[3,3])*q.Gamma[3]
            else
                # Classic scheme (Γq⋅∇)(Up - Uq)
                S1 = (p.J[1,1] - q.J[1,1])*q.Gamma[1]+(p.J[1,2] - q.J[1,2])*q.Gamma[2]+(p.J[1,3] - q.J[1,3])*q.Gamma[3]
                S2 = (p.J[2,1] - q.J[2,1])*q.Gamma[1]+(p.J[2,2] - q.J[2,2])*q.Gamma[2]+(p.J[2,3] - q.J[2,3])*q.Gamma[3]
                S3 = (p.J[3,1] - q.J[3,1])*q.Gamma[1]+(p.J[3,2] - q.J[3,2])*q.Gamma[2]+(p.J[3,3] - q.J[3,3])*q.Gamma[3]
            end

            dX1 = p.X[1] - q.X[1]
            dX2 = p.X[2] - q.X[2]
            dX3 = p.X[3] - q.X[3]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            zeta_sgm = zeta(r/q.sigma[1]) / q.sigma[1]^3

            # Add ζ_σ (Γq⋅∇)(Up - Uq)
            add_SFS1(p, zeta_sgm*S1)
            add_SFS2(p, zeta_sgm*S2)
            add_SFS3(p, zeta_sgm*S3)
        end
    end
end

@inline function Estr_direct(target_system, target_index, source_system, source_index, transposed)
    return nothing
end

@inline function Estr_direct(target_system::ParticleField, target_index, source_system::ParticleField, source_index, transposed)
    for p in view(target_system.particles, target_index)
        for q in view(source_system.particles, source_index)

            # Stretching term
            if transposed
                # Transposed scheme (Γq⋅∇')(Up - Uq)
                S1 = (p.J[1,1] - q.J[1,1])*q.Gamma[1]+(p.J[2,1] - q.J[2,1])*q.Gamma[2]+(p.J[3,1] - q.J[3,1])*q.Gamma[3]
                S2 = (p.J[1,2] - q.J[1,2])*q.Gamma[1]+(p.J[2,2] - q.J[2,2])*q.Gamma[2]+(p.J[3,2] - q.J[3,2])*q.Gamma[3]
                S3 = (p.J[1,3] - q.J[1,3])*q.Gamma[1]+(p.J[2,3] - q.J[2,3])*q.Gamma[2]+(p.J[3,3] - q.J[3,3])*q.Gamma[3]
            else
                # Classic scheme (Γq⋅∇)(Up - Uq)
                S1 = (p.J[1,1] - q.J[1,1])*q.Gamma[1]+(p.J[1,2] - q.J[1,2])*q.Gamma[2]+(p.J[1,3] - q.J[1,3])*q.Gamma[3]
                S2 = (p.J[2,1] - q.J[2,1])*q.Gamma[1]+(p.J[2,2] - q.J[2,2])*q.Gamma[2]+(p.J[2,3] - q.J[2,3])*q.Gamma[3]
                S3 = (p.J[3,1] - q.J[3,1])*q.Gamma[1]+(p.J[3,2] - q.J[3,2])*q.Gamma[2]+(p.J[3,3] - q.J[3,3])*q.Gamma[3]
            end

            dX1 = p.X[1] - q.X[1]
            dX2 = p.X[2] - q.X[2]
            dX3 = p.X[3] - q.X[3]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            zeta_sgm = source_system.kernel.zeta(r/q.sigma[1]) / q.sigma[1]^3

            # Add ζ_σ (Γq⋅∇)(Up - Uq)
            add_SFS1(p, zeta_sgm*S1)
            add_SFS2(p, zeta_sgm*S2)
            add_SFS3(p, zeta_sgm*S3)
        end
    end
end


"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    UJ_fmm(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end
