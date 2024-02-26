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
@inline function Estr_direct(target_particle::Particle, source_particle::Particle, r, zeta, transposed)
    # Stretching term
    if transposed
        # Transposed scheme (Γq⋅∇')(Up - Uq)
        S1 = (target_particle.J[1,1] - source_particle.J[1,1])*source_particle.Gamma[1]+(target_particle.J[2,1] - source_particle.J[2,1])*source_particle.Gamma[2]+(target_particle.J[3,1] - source_particle.J[3,1])*source_particle.Gamma[3]
        S2 = (target_particle.J[1,2] - source_particle.J[1,2])*source_particle.Gamma[1]+(target_particle.J[2,2] - source_particle.J[2,2])*source_particle.Gamma[2]+(target_particle.J[3,2] - source_particle.J[3,2])*source_particle.Gamma[3]
        S3 = (target_particle.J[1,3] - source_particle.J[1,3])*source_particle.Gamma[1]+(target_particle.J[2,3] - source_particle.J[2,3])*source_particle.Gamma[2]+(target_particle.J[3,3] - source_particle.J[3,3])*source_particle.Gamma[3]
    else
        # Classic scheme (Γq⋅∇)(Up - Uq)
        S1 = (p.J[1,1] - source_particle.J[1,1])*source_particle.Gamma[1]+(p.J[1,2] - source_particle.J[1,2])*source_particle.Gamma[2]+(p.J[1,3] - source_particle.J[1,3])*source_particle.Gamma[3]
        S2 = (p.J[2,1] - source_particle.J[2,1])*source_particle.Gamma[1]+(p.J[2,2] - source_particle.J[2,2])*source_particle.Gamma[2]+(p.J[2,3] - source_particle.J[2,3])*source_particle.Gamma[3]
        S3 = (p.J[3,1] - source_particle.J[3,1])*source_particle.Gamma[1]+(p.J[3,2] - source_particle.J[3,2])*source_particle.Gamma[2]+(p.J[3,3] - source_particle.J[3,3])*source_particle.Gamma[3]
    end

    zeta_sgm = (r/source_particle.var[7]) / source_particle.var[7]^3

    # Add ζ_σ (Γq⋅∇)(Up - Uq)
    add_SFS1(target_particle, zeta_sgm*S1)
    add_SFS2(target_particle, zeta_sgm*S2)
    add_SFS3(target_particle, zeta_sgm*S3)
end

"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    UJ_fmm(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end
