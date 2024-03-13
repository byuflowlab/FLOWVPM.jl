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
    GS = get_Gamma(source_particle)
    JS = get_J(source_particle)
    JT = get_J(target_particle)

    # Stretching term
    if transposed
        # Transposed scheme (Γq⋅∇')(Up - Uq)
        S1 = (JT[1] - JS[1])*GS[1]+(JT[2] - JS[2])*GS[2]+(JT[3] - JS[3])*GS[3]
        S2 = (JT[4] - JS[4])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[6] - JS[6])*GS[3]
        S3 = (JT[7] - JS[7])*GS[1]+(JT[8] - JS[8])*GS[2]+(JT[9] - JS[9])*GS[3]
    else
        # Classic scheme (Γq⋅∇)(Up - Uq)
        S1 = (JT[1] - JS[1])*GS[1]+(JT[4] - JS[4])*GS[2]+(JT[7] - JS[7])*GS[3]
        S2 = (JT[2] - JS[2])*GS[1]+(JT[5] - JS[5])*GS[2]+(JT[8] - JS[8])*GS[3]
        S3 = (JT[3] - JS[3])*GS[1]+(JT[6] - JS[6])*GS[2]+(JT[9] - JS[9])*GS[3]
    end

    zeta_sgm = (r/get_sigma(source_particle)[]) / get_sigma(source_particle)[]^3

    # Add ζ_σ (Γq⋅∇)(Up - Uq)
    get_SFS(target_particle)[1] += zeta_sgm*S1
    get_SFS(target_particle)[2] += zeta_sgm*S2
    get_SFS(target_particle)[3] += zeta_sgm*S3
end

"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    UJ_fmm(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end
