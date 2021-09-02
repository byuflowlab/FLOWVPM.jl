#=##############################################################################
# DESCRIPTION
    Subfilter-scale (SFS) turbulence models for large eddy simulation. See
20210901 notebook for theory and implementation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


"""
    Model of vortex-stretching SFS contributions. See 20210901 notebook for
derivation.
"""
function Estr_direct(pfield::ParticleField)
  return Estr_direct(iterator(pfield), iterator(pfield),
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

            zeta_sgm = 1/q.sigma[1]^3*zeta(r/q.sigma[1])

            # Add -ζ_σ (Γq⋅∇)(Up - Uq)
            add_SFS1(p, -zeta_sgm*S1)
            add_SFS2(p, -zeta_sgm*S2)
            add_SFS3(p, -zeta_sgm*S3)
        end
    end
end


function Estr_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs=reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end



function sfs_stretching1_direct(pfield::ParticleField)
  return sfs_stretching1_direct(iterator(pfield), iterator(pfield),
                                        pfield.kernel.zeta, pfield.transposed)
end

function sfs_stretching1_direct(sources, targets, zeta, transposed)

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
                add_SFS1(p, -zeta_sgm*S1)
                add_SFS2(p, -zeta_sgm*S2)
                add_SFS3(p, -zeta_sgm*S3)
            end
        end
    end
end


function sfs_stretching2_direct(pfield::ParticleField)
  return sfs_stretching2_direct(iterator(pfield), iterator(pfield),
                                        pfield.kernel.zeta, pfield.transposed)
end

function sfs_stretching2_direct(sources, targets, zeta, transposed)

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
            add_SFS1(p, -zeta_sgm*S1)
            add_SFS2(p, -zeta_sgm*S2)
            add_SFS3(p, -zeta_sgm*S3)
        end

        if (p.Gamma[1]*get_SFS1(p) + p.Gamma[2]*get_SFS2(p) + p.Gamma[3]*get_SFS3(p)) >= 0 # Backscattering filtering criterion
            add_SFS1(p, -get_SFS1(p))
            add_SFS2(p, -get_SFS2(p))
            add_SFS3(p, -get_SFS3(p))
        end
    end
end

function sfs_stretching0_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=0, reset_sfs=reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end

function sfs_stretching1_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=1, reset_sfs=reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end

function sfs_stretching2_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=2, reset_sfs=reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end

function sfs_M2_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    call_FLOWExaFMM(pfield; reset=false, sfs=true, sfs_type=5, reset_sfs=reset_sfs,
                            transposed_sfs=pfield.transposed, optargs...)
end

function sfs_stretching3_fmm(pfield::ParticleField; reset_sfs=true, optargs...)
    sfs_M2_fmm(pfield; reset_sfs=true, optargs...)
    sfs_M2_fmm(pfield; reset_sfs=false, optargs...)
end

"""
    Generate a filtered SFS model such that the resulting SFS model never flips
the vortex strength direction in an Euler step. See notebook 20210211.
"""
function generate_sfs_lowfiltered(SFS::Function)

    function lowfiltered_sfs(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale};
                                                    optargs...) where {R, V, R2}

        # Evaluate original SFS model
        SFS(pfield; optargs...)

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
            aux += (get_SFS1(p)*p.Gamma[1] + get_SFS2(p)*p.Gamma[2] + get_SFS3(p)*p.Gamma[3]
                    ) / (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # f_p filter criterion
            if aux < 0
                add_SFS1(p, -aux*p.Gamma[1])
                add_SFS2(p, -aux*p.Gamma[2])
                add_SFS3(p, -aux*p.Gamma[3])
            end
        end
    end

    function lowfiltered_sfs(pfield::ParticleField{R, <:ClassicVPM, V, <:SubFilterScale};
                                                        optargs...) where {R, V}

        # Evaluate original SFS model
        SFS(pfield; optargs...)

        # Estimate Δ𝑡
        if pfield.nt == 0
            return nothing
        else
            dt::R = pfield.t / pfield.nt
        end

        zeta0::R = pfield.kernel.zeta(0)

        # Filter the SFS contribution at every particle
        for p in iterator(pfield)

            aux = (zeta0/p.sigma[1]^3) / dt
            aux += (get_SFS1(p)*p.Gamma[1] + get_SFS2(p)*p.Gamma[2] + get_SFS3(p)*p.Gamma[3]
                    ) / (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # f_p filter criterion
            if aux < 0
                add_SFS1(p, -aux*p.Gamma[1])
                add_SFS2(p, -aux*p.Gamma[2])
                add_SFS3(p, -aux*p.Gamma[3])
            end
        end
    end

    return lowfiltered_sfs
end


function generate_sfs_stronglowfiltered(SFS::Function)

    function strong_lowfiltered_sfs(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale};
                                                    optargs...) where {R, V, R2}

        # Evaluate original SFS model
        SFS(pfield; optargs...)

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

            aux = (S1+get_SFS1(p))*p.Gamma[1] + (S2+get_SFS2(p))*p.Gamma[2] + (S3+get_SFS3(p))*p.Gamma[3]
            aux /= (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])
            aux += (1+3*f)*(zeta0/p.sigma[1]^3) / deltat

            # f_p filter criterion
            if aux < 0
                add_SFS1(p, -aux*p.Gamma[1])
                add_SFS2(p, -aux*p.Gamma[2])
                add_SFS3(p, -aux*p.Gamma[3])
            end
        end
    end


    return strong_lowfiltered_sfs
end

"""
    Generate a filtered SFS model such that the resulting SFS model does not
reorient the vortex strength but only affects its magnitude.
See notebook 20210211.
"""
function generate_sfs_directionfiltered(SFS::Function)

    function directionfiltered_sfs(pfield::ParticleField; optargs...)

        # Evaluate original SFS model
        SFS(pfield; optargs...)

        for p in iterator(pfield)
            aux = get_SFS1(p)*p.Gamma[1] + get_SFS2(p)*p.Gamma[2] + get_SFS3(p)*p.Gamma[3]
            aux /= (p.Gamma[1]*p.Gamma[1] + p.Gamma[2]*p.Gamma[2] + p.Gamma[3]*p.Gamma[3])

            # Replaces old SFS with the direcional filtered SFS
            if aux < 0
                add_SFS1(p, -get_SFS1(p) + aux*p.Gamma[1])
                add_SFS2(p, -get_SFS2(p) + aux*p.Gamma[2])
                add_SFS3(p, -get_SFS3(p) + aux*p.Gamma[3])
            end
        end
    end


    return directionfiltered_sfs
end
