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


function sgs_stretching2_direct(pfield::ParticleField)
  return sgs_stretching2_direct(iterator(pfield), iterator(pfield),
                                        pfield.kernel.zeta, pfield.transposed)
end

function sgs_stretching2_direct(sources, targets, zeta, transposed)

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

            zeta_sgm = 1/q.sigma[1]^3*zeta(r/q.sigma[1])

            # Add -ζ_σ (Γq⋅∇)(Up - Uq)
            add_SGS1(p, -zeta_sgm*S1)
            add_SGS2(p, -zeta_sgm*S2)
            add_SGS3(p, -zeta_sgm*S3)
        end

        if (p.Gamma[1]*get_SGS1(p) + p.Gamma[2]*get_SGS2(p) + p.Gamma[3]*get_SGS3(p)) >= 0 # Backscattering filtering criterion
            add_SGS1(p, -get_SGS1(p))
            add_SGS2(p, -get_SGS2(p))
            add_SGS3(p, -get_SGS3(p))
        end
    end
end

function sgs_stretching0_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sgs=true, sgs_type=0, reset_sgs=reset_sgs,
                            transposed_sgs=pfield.transposed, optargs...)
end

function sgs_stretching1_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sgs=true, sgs_type=1, reset_sgs=reset_sgs,
                            transposed_sgs=pfield.transposed, optargs...)
end

function sgs_stretching2_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sgs=true, sgs_type=2, reset_sgs=reset_sgs,
                            transposed_sgs=pfield.transposed, optargs...)
end

function sgs_M2_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sgs=true, sgs_type=5, reset_sgs=reset_sgs,
                            transposed_sgs=pfield.transposed, optargs...)
end

function sgs_stretching3_fmm(pfield::ParticleField; reset_sgs=true, optargs...)
    sgs_M2_fmm(pfield; reset_sgs=true, optargs...)
    sgs_M2_fmm(pfield; reset_sgs=false, optargs...)
end

"""
    Generate a filtered SGS model such that the resulting SGS model never flips
the vortex strength direction in an Euler step. See notebook 20210211.
"""
function generate_sgs_lowfiltered(SGS::Function)

    function lowfiltered_sgs(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V};
                                                    optargs...) where {R, V, R2}

        # Evaluate original SGS model
        SGS(pfield; optargs...)

        # Estimate Δ𝑡
        if pfield.nt == 0
            return nothing
        else
            deltat::R = pfield.t / pfield.nt
        end

        f::R2 = pfield.formulation.f
        zeta0::R = pfield.kernel.zeta(0)

        for p in iterator(pfield)

            aux = (1+3*f)*(zeta0/p.sigma[1]^3) / deltat
            aux += (get_SGS1(p)*p.Gamma[1] + get_SGS2(p)*p.Gamma[2] + get_SGS3(p)*p.Gamma[3]
                    ) / (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # f_p filter criterion
            if aux < 0
                add_SGS1(p, -aux*p.Gamma[1])
                add_SGS2(p, -aux*p.Gamma[2])
                add_SGS3(p, -aux*p.Gamma[3])
            end
        end
    end

    function lowfiltered_sgs(pfield::ParticleField{R, <:ClassicVPM, V};
                                                        optargs...) where {R, V}

        # Evaluate original SGS model
        SGS(pfield; optargs...)

        # Estimate Δ𝑡
        if pfield.nt == 0
            return nothing
        else
            dt::R = pfield.t / pfield.nt
        end

        zeta0::R = pfield.kernel.zeta(0)

        # Filter the SGS contribution at every particle
        for p in iterator(pfield)

            aux = (zeta0/p.sigma[1]^3) / dt
            aux += (get_SGS1(p)*p.Gamma[1] + get_SGS2(p)*p.Gamma[2] + get_SGS3(p)*p.Gamma[3]
                    ) / (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # f_p filter criterion
            if aux < 0
                add_SGS1(p, -aux*p.Gamma[1])
                add_SGS2(p, -aux*p.Gamma[2])
                add_SGS3(p, -aux*p.Gamma[3])
            end
        end
    end

    return lowfiltered_sgs
end


function generate_sgs_stronglowfiltered(SGS::Function)

    function strong_lowfiltered_sgs(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V};
                                                    optargs...) where {R, V, R2}

        # Evaluate original SGS model
        SGS(pfield; optargs...)

        # Estimate Δ𝑡
        if pfield.nt == 0
            return nothing
        else
            deltat::R = pfield.t / pfield.nt
        end

        f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
        zeta0::R = pfield.kernel.zeta(0)

        for p in iterator(pfield)

            # Calculate stretching
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U
                S1 = p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3]
                S2 = p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3]
                S3 = p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
            else
                # Classic scheme (Γ⋅∇)U
                S1 = p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3]
                S2 = p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3]
                S3 = p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
            end

            S1 *= (1-3*g)*(zeta0/p.sigma[1]^3)
            S2 *= (1-3*g)*(zeta0/p.sigma[1]^3)
            S3 *= (1-3*g)*(zeta0/p.sigma[1]^3)

            aux = (S1+get_SGS1(p))*p.Gamma[1] + (S2+get_SGS2(p))*p.Gamma[2] + (S3+get_SGS3(p))*p.Gamma[3]
            aux /= (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])
            aux += (1+3*f)*(zeta0/p.sigma[1]^3) / deltat

            # f_p filter criterion
            if aux < 0
                add_SGS1(p, -aux*p.Gamma[1])
                add_SGS2(p, -aux*p.Gamma[2])
                add_SGS3(p, -aux*p.Gamma[3])
            end
        end
    end


    return strong_lowfiltered_sgs
end

"""
    Generate a filtered SGS model such that the resulting SGS model does not
reorient the vortex strength but only affects its magnitude.
See notebook 20210211.
"""
function generate_sgs_directionfiltered(SGS::Function)

    function directionfiltered_sgs(pfield::ParticleField; optargs...)

        # Evaluate original SGS model
        SGS(pfield; optargs...)

        for p in iterator(pfield)
            aux = get_SGS1(p)*p.Gamma[1] + get_SGS2(p)*p.Gamma[2] + get_SGS3(p)*p.Gamma[3]
            aux /= (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # Replaces old SGS with the direcional filtered SGS
            if aux < 0
                add_SGS1(p, -get_SGS1(p) + aux*p.Gamma[1])
                add_SGS2(p, -get_SGS2(p) + aux*p.Gamma[2])
                add_SGS3(p, -get_SGS3(p) + aux*p.Gamma[3])
            end
        end
    end


    return directionfiltered_sgs
end
