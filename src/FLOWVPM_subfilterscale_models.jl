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
        S1 = (target_particle.var[16] - source_particle.var[16])*source_particle.var[4]+(target_particle.var[17] - source_particle.var[17])*source_particle.var[5]+(target_particle.var[18] - source_particle.var[18])*source_particle.var[6]
        S2 = (target_particle.var[19] - source_particle.var[19])*source_particle.var[4]+(target_particle.var[20] - source_particle.var[20])*source_particle.var[5]+(target_particle.var[21] - source_particle.var[21])*source_particle.var[6]
        S3 = (target_particle.var[22] - source_particle.var[22])*source_particle.var[4]+(target_particle.var[23] - source_particle.var[23])*source_particle.var[5]+(target_particle.var[24] - source_particle.var[24])*source_particle.var[6]
    else
        # Classic scheme (Γq⋅∇)(Up - Uq)
        S1 = (p.var[16] - source_particle.var[16])*source_particle.var[4]+(p.var[19] - source_particle.var[19])*source_particle.var[5]+(p.var[22] - source_particle.var[22])*source_particle.var[6]
        S2 = (p.var[17] - source_particle.var[17])*source_particle.var[4]+(p.var[20] - source_particle.var[20])*source_particle.var[5]+(p.var[23] - source_particle.var[23])*source_particle.var[6]
        S3 = (p.var[18] - source_particle.var[18])*source_particle.var[4]+(p.var[21] - source_particle.var[21])*source_particle.var[5]+(p.var[24] - source_particle.var[24])*source_particle.var[6]
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
