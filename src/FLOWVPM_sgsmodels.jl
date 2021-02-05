#=##############################################################################
# DESCRIPTION
    Implementation of subgrid-scale models associated to large-eddy simulation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jan 2021
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

function sgs_stretching1_direct(pfield::ParticleField)
  return sgs_stretching1_direct(iterator(pfield), iterator(pfield),
                                        pfield.kernel.zeta, pfield.transposed)
end

function sgs_stretching1_direct(sources, targets, zeta, transposed)

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

            if (p.Gamma[1]*S1 + p.Gamma[2]*S2 + p.Gamma[3]*S3) >= 0 # Backscattering filtering criterion
                dX1 = p.X[1] - q.X[1]
                dX2 = p.X[2] - q.X[2]
                dX3 = p.X[3] - q.X[3]
                r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

                zeta_sgm = 1/q.sigma[1]^3*zeta(r/q.sigma[1])

                # Add -ζ_σ (Γq⋅∇)(Up - Uq)
                add_SGS1(p, -zeta_sgm*S1)
                add_SGS2(p, -zeta_sgm*S2)
                add_SGS3(p, -zeta_sgm*S3)
            end
        end
    end
end

function sgs_stretching1_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sgs=true, sgs_type=1, reset_sgs=reset_sgs,
                            transposed_sgs=pfield.transposed, optargs...)
end
