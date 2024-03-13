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
@inline function Estr_direct(target_particle, source_particle, r, zeta, transposed)
    # Stretching term
    if transposed
        # Transposed scheme (Γq⋅∇')(Up - Uq)
        S1 = (get_J(target_particle)[1] - get_J(source_particle)[1])*get_Gamma(source_particle)[1]+(get_J(target_particle)[2] - get_J(source_particle)[2])*get_Gamma(source_particle)[2]+(get_J(target_particle)[3] - get_J(source_particle)[3])*get_Gamma(source_particle)[3]
        S2 = (get_J(target_particle)[4] - get_J(source_particle)[4])*get_Gamma(source_particle)[1]+(get_J(target_particle)[5] - get_J(source_particle)[5])*get_Gamma(source_particle)[2]+(get_J(target_particle)[6] - get_J(source_particle)[6])*get_Gamma(source_particle)[3]
        S3 = (get_J(target_particle)[7] - get_J(source_particle)[7])*get_Gamma(source_particle)[1]+(get_J(target_particle)[8] - get_J(source_particle)[8])*get_Gamma(source_particle)[2]+(get_J(target_particle)[9] - get_J(source_particle)[9])*get_Gamma(source_particle)[3]
    else
        # Classic scheme (Γq⋅∇)(Up - Uq)
        S1 = (get_J(p)[1] - get_J(source_particle)[1])*get_Gamma(source_particle)[1]+(get_J(p)[4] - get_J(source_particle)[4])*get_Gamma(source_particle)[2]+(get_J(p)[7] - get_J(source_particle)[7])*get_Gamma(source_particle)[3]
        S2 = (get_J(p)[2] - get_J(source_particle)[2])*get_Gamma(source_particle)[1]+(get_J(p)[5] - get_J(source_particle)[5])*get_Gamma(source_particle)[2]+(get_J(p)[8] - get_J(source_particle)[8])*get_Gamma(source_particle)[3]
        S3 = (get_J(p)[3] - get_J(source_particle)[3])*get_Gamma(source_particle)[1]+(get_J(p)[6] - get_J(source_particle)[6])*get_Gamma(source_particle)[2]+(get_J(p)[9] - get_J(source_particle)[9])*get_Gamma(source_particle)[3]
    end

    zeta_sgm = (r/get_sigma(source_particle)[]) / get_sigma(source_particle)[]^3

    # Add ζ_σ (Γq⋅∇)(Up - Uq)
    get_SFS(target_particle) .+= zeta_sgm*S1, zeta_sgm*S2, zeta_sgm*S3
end

"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    UJ_fmm(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end
