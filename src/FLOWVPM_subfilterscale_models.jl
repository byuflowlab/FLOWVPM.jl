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


"""
    Model of vortex-stretching SFS contributions evaluated with fast multipole
method. See 20210901 notebook for derivation.
"""
function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs=reset_sfs,
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
