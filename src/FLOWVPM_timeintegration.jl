#=##############################################################################
# DESCRIPTION
    Time integration schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

"""
Steps the field forward in time by dt in a first-order Euler integration scheme.
"""
function euler(pfield::ParticleField, dt::Real; relax::Bool=false)

    # Reset U and J from previous step
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Update the particle field
    for p in iterator(pfield)

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Update vectorial circulation
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p.Gamma[1] += dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
            p.Gamma[2] += dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
            p.Gamma[3] += dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        else
            # Classic scheme (Γ⋅∇)U
            p.Gamma[1] += dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
            p.Gamma[2] += dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
            p.Gamma[3] += dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        end

        # Viscous scheme: core spreading
        p.sigma[1] = sqrt(p.sigma[1]^2 + 2*pfield.nu*dt)

        # Relaxation: Align vectorial circulation with local vorticity
        if relax
            nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)
            p.Gamma[1] = (1-pfield.rlxf)*p.Gamma[1] + pfield.rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
            p.Gamma[2] = (1-pfield.rlxf)*p.Gamma[2] + pfield.rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
            p.Gamma[3] = (1-pfield.rlxf)*p.Gamma[3] + pfield.rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw
        end

    end

    return nothing
end




"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{T}, dt::Real; relax::Bool=false) where {T}

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Creates storage terms
    q_U = zeros(T, 3, get_np(pfield))
    q_str = zeros(T, 3, get_np(pfield))
    q_sgm2 = zeros(T, get_np(pfield))

    for (a,b) in ((0, 1/3), (-5/9, 15/16), (-153/128, 8/15))

        # Resets U and J from previous step
        _reset_particles(pfield)

        # Calculates interactions between particles: U and J
        pfield.UJ(pfield, pfield)

        # Updates the particle field
        for (pi, p) in enumerate(iterator(pfield))

            # Low-storage RK step
            ## Velocity
            q_U[1, pi] = a*q_U[1, pi] + dt*(p.U[1] + Uinf[1])
            q_U[2, pi] = a*q_U[2, pi] + dt*(p.U[2] + Uinf[2])
            q_U[3, pi] = a*q_U[3, pi] + dt*(p.U[3] + Uinf[3])

            ## Stretching
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U
                q_str[1, pi] = a*q_str[1, pi] + dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
                q_str[2, pi] = a*q_str[2, pi] + dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
                q_str[3, pi] = a*q_str[3, pi] + dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            else
                # Classic scheme (Γ⋅∇)U
                q_str[1, pi] = a*q_str[1, pi] + dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
                q_str[2, pi] = a*q_str[2, pi] + dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
                q_str[3, pi] = a*q_str[3, pi] + dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            end

            ## Core growth
            q_sgm2[pi] = a*q_sgm2[pi] + dt*2*pfield.nu

            # Updates position
            p.X[1] += b*q_U[1, pi]
            p.X[2] += b*q_U[2, pi]
            p.X[3] += b*q_U[3, pi]

            # Updates vectorial circulation
            p.Gamma[1] += b*q_str[1, pi]
            p.Gamma[2] += b*q_str[2, pi]
            p.Gamma[3] += b*q_str[3, pi]

            # Viscous scheme: core spreading
            p.sigma[1] = sqrt(p.sigma[1]^2 + b*q_sgm2[pi])

        end

    end


    # Relaxation: Aligns vectorial circulation with local vorticity
    if relax

        # Resets U and J from previous step
        _reset_particles(pfield)

        # Calculates interactions between particles: U and J
        # NOTE: Technically we have to calculate J at the final location,
        #       but in MyVPM I just used the J calculated in the last RK step
        #       and it worked just fine. So maybe I perhaps I can save computation
        #       by not calculating UJ again.
        pfield.UJ(pfield, pfield)

        for p in iterator(pfield)
            nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)
            p.Gamma[1] = (1-pfield.rlxf)*p.Gamma[1] + pfield.rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
            p.Gamma[2] = (1-pfield.rlxf)*p.Gamma[2] + pfield.rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
            p.Gamma[3] = (1-pfield.rlxf)*p.Gamma[3] + pfield.rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw
        end
    end

    return nothing
end
