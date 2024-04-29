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

"""
    SFS model wrapper that hides the static particles from the model in order
to avoid potential numerical instabilities encountered at solid surfaces.
"""
function E_nostaticparticles(pfield, args...; E=Estr_fmm, optargs...)

    @assert pfield.np < pfield.maxparticles "Sorting of particles is needed"*
        " but all pre-allocated memory is already in use"

    org_np = pfield.np
    iaux = pfield.np + 1

    # Fetch auxiliary memory
    paux = get_particle(pfield, iaux; emptyparticle=true)

    # Iterate over particles
    for pi in pfield.np:-1:1

        # Fetch target particles
        p = get_particle(pfield, pi)

        # Case that we found a static particle
        if p.static[1]

            if pi==pfield.np
                nothing

            # Swap this particle with last particle
            else

                # Fetch last particle
                pnp = get_particle(pfield, pfield.np)

                # Store static particle in auxiliary memory
                fmm.overwriteBody(pfield.bodies, iaux-1, pi-1)
                paux.circulation .= p.circulation
                paux.C .= p.C
                paux.static .= p.static

                # Move last particle into the static particle's memory
                fmm.overwriteBody(pfield.bodies, pi-1, pfield.np-1)
                p.circulation .= pnp.circulation
                p.C .= pnp.C
                p.static .= pnp.static

                # Move static particle into the last particle's memory
                fmm.overwriteBody(pfield.bodies, pfield.np-1, iaux-1)
                pnp.circulation .= paux.circulation
                pnp.C .= paux.C
                pnp.static .= paux.static

            end

            # Move "end of array" pointer to hide the static particle
            pfield.np -= 1
        end
    end

    # Call SFS model without the static particles
    E(pfield, args...; optargs...)

    # Restore static particles back to the field
    # pfield.np = org_np

    # NOTE: Here we add the auxiliary memory to the field and then remove it.
    #       This is to make sure that the memory is cleaned and avoid potential
    #       bugs
    pfield.np = org_np + 1
    remove_particle(pfield, pfield.np)

    # # Sort particles to restore the original indexing
    # sort!(iterator(pfield), by = p->p.index[1])

end
