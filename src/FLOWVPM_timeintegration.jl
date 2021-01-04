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
function euler(pfield::ParticleField{PType, F, V}, dt::Real; relax::Bool=false
                                                 ) where {PType<:Particle, F, V}

    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Update the particle field: convection and stretching
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

        # Relaxation: Align vectorial circulation with local vorticity
        if relax
            nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

            b2 =  1 - 2*(1-pfield.rlxf)*pfield.rlxf*(1 - (
                                                            p.Gamma[1]*(p.J[3,2]-p.J[2,3]) +
                                                            p.Gamma[2]*(p.J[1,3]-p.J[3,1]) +
                                                            p.Gamma[3]*(p.J[2,1]-p.J[1,2])
                                                          ) / (nrmGamma*nrmw))

            p.Gamma[1] = (1-pfield.rlxf)*p.Gamma[1] + pfield.rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
            p.Gamma[2] = (1-pfield.rlxf)*p.Gamma[2] + pfield.rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
            p.Gamma[3] = (1-pfield.rlxf)*p.Gamma[3] + pfield.rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

            # Normalize the direction of the new vector to maintain the same strength
            p.Gamma ./= sqrt(b2)
        end

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end


"""
Steps the field forward in time by dt in a first-order Euler integration scheme.
"""
function euler(pfield::ParticleField{PType, F, V}, dt::Real; relax::Bool=false
                                             ) where {PType<:ParticleTube, F, V}

    # Reset U and J to zero
    _reset_particles(pfield)

    # Convert vortex tube length into vectorial circulation
    for P in iterator(pfield)
        P.Gamma .= P.l
        P.Gamma .*= P.circulation[1]
    end

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Store stretching under M[:, 4]
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p.M[1,4] = 2/5*(p.J[1,1]*p.l[1]+p.J[2,1]*p.l[2]+p.J[3,1]*p.l[3])
            p.M[2,4] = 2/5*(p.J[1,2]*p.l[1]+p.J[2,2]*p.l[2]+p.J[3,2]*p.l[3])
            p.M[3,4] = 2/5*(p.J[1,3]*p.l[1]+p.J[2,3]*p.l[2]+p.J[3,3]*p.l[3])
        else
            # Classic scheme (Γ⋅∇)U
            p.M[1,4] = 2/5*(p.J[1,1]*p.l[1]+p.J[1,2]*p.l[2]+p.J[1,3]*p.l[3])
            p.M[2,4] = 2/5*(p.J[2,1]*p.l[1]+p.J[2,2]*p.l[2]+p.J[2,3]*p.l[3])
            p.M[3,4] = 2/5*(p.J[3,1]*p.l[1]+p.J[3,2]*p.l[2]+p.J[3,3]*p.l[3])
        end

        # Update cross-sectional area according to stretching
        # p.sigma[1] -= dt * ( p.sigma[1]/(2*sqrt(p.l[1]^2+p.l[2]^2+p.l[3]^2)) * sqrt(p.M[1,4]^2+p.M[2,4]^2+p.M[3,4]^2) )
        p.sigma[1] -= dt * ( p.sigma[1]/(2*(p.l[1]^2+p.l[2]^2+p.l[3]^2)) * (p.l[1]*p.M[1,4]+p.l[2]*p.M[2,4]+p.l[3]*p.M[3,4]) )

        # Update vortex tube length
        p.l[1] += dt*p.M[1,4]
        p.l[2] += dt*p.M[2,4]
        p.l[3] += dt*p.M[3,4]

        # Relaxation: Align vectorial circulation with local vorticity
        if relax
            nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrml = sqrt(p.l[1]^2 + p.l[2]^2 + p.l[3]^2)

            b2 =  1 - 2*(1-pfield.rlxf)*pfield.rlxf*(1 - (
                                                            p.l[1]*(p.J[3,2]-p.J[2,3]) +
                                                            p.l[2]*(p.J[1,3]-p.J[3,1]) +
                                                            p.l[3]*(p.J[2,1]-p.J[1,2])
                                                          ) / (nrml*nrmw))

            p.l[1] = (1-pfield.rlxf)*p.l[1] + pfield.rlxf*nrml*(p.J[3,2]-p.J[2,3])/nrmw
            p.l[2] = (1-pfield.rlxf)*p.l[2] + pfield.rlxf*nrml*(p.J[1,3]-p.J[3,1])/nrmw
            p.l[3] = (1-pfield.rlxf)*p.l[3] + pfield.rlxf*nrml*(p.J[2,1]-p.J[1,2])/nrmw

            # Normalize the direction of the new vector to maintain the same strength
            p.l ./= sqrt(b2)
        end

        # Convert vortex tube length into vectorial circulation
        p.Gamma .= p.l
        p.Gamma .*= p.circulation[1]

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end




"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{PType, F, V}, dt::Real;
                                relax::Bool=false) where {PType<:Particle{T}, F, V}

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3]

    # Reset storage memory to zero
    for p in iterator(pfield); p.M .= zero(T); end;

    # Runge-Kutta inner steps
    for (a,b) in (T.((0, 1/3)), T.((-5/9, 15/16)), T.((-153/128, 8/15)))

        # Resets U and J from previous step
        _reset_particles(pfield)

        # Calculates interactions between particles: U and J
        pfield.UJ(pfield)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            ## Stretching
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            else
                # Classic scheme (Γ⋅∇)U
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            end

            # Updates position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            # Updates vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

        end

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)

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
        pfield.UJ(pfield)

        for p in iterator(pfield)
            nrmw = sqrt(    (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

            b2 =  1 - 2*(1-pfield.rlxf)*pfield.rlxf*(1 - (
                                                            p.Gamma[1]*(p.J[3,2]-p.J[2,3]) +
                                                            p.Gamma[2]*(p.J[1,3]-p.J[3,1]) +
                                                            p.Gamma[3]*(p.J[2,1]-p.J[1,2])
                                                          ) / (nrmGamma*nrmw))

            p.Gamma[1] = (1-pfield.rlxf)*p.Gamma[1] + pfield.rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
            p.Gamma[2] = (1-pfield.rlxf)*p.Gamma[2] + pfield.rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
            p.Gamma[3] = (1-pfield.rlxf)*p.Gamma[3] + pfield.rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

            # Normalize the direction of the new vector to maintain the same strength
            p.Gamma ./= sqrt(b2)
        end
    end

    return nothing
end







"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{PType, F, V}, dt::Real;
                        relax::Bool=false) where {PType<:ParticleTube{T}, F, V}

    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    # Convert vortex tube length into vectorial circulation
    for P in iterator(pfield)
        P.Gamma .= P.l
        P.Gamma .*= P.circulation[1]
    end

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3], qsmg <=> p.M[2, 3], ql <=> p.M[:, 5]

    # Reset storage memory to zero
    for p in iterator(pfield); p.M .= zero(T); end;

    # Runge-Kutta inner steps
    for (a,b) in (T.((0, 1/3)), T.((-5/9, 15/16)), T.((-153/128, 8/15)))

        # Resets U and J from previous step
        _reset_particles(pfield)

        # Calculates interactions between particles: U and J
        pfield.UJ(pfield)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            # Store stretching under M[:, 4]
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U
                p.M[1,4] = 2/5*(p.J[1,1]*p.l[1]+p.J[2,1]*p.l[2]+p.J[3,1]*p.l[3])
                p.M[2,4] = 2/5*(p.J[1,2]*p.l[1]+p.J[2,2]*p.l[2]+p.J[3,2]*p.l[3])
                p.M[3,4] = 2/5*(p.J[1,3]*p.l[1]+p.J[2,3]*p.l[2]+p.J[3,3]*p.l[3])
            else
                # Classic scheme (Γ⋅∇)U
                p.M[1,4] = 2/5*(p.J[1,1]*p.l[1]+p.J[1,2]*p.l[2]+p.J[1,3]*p.l[3])
                p.M[2,4] = 2/5*(p.J[2,1]*p.l[1]+p.J[2,2]*p.l[2]+p.J[2,3]*p.l[3])
                p.M[3,4] = 2/5*(p.J[3,1]*p.l[1]+p.J[3,2]*p.l[2]+p.J[3,3]*p.l[3])
            end

            ## Storage of vortex tube length stretching
            p.M[1, 5] = a*p.M[1, 5] + dt*(p.M[1,4])
            p.M[2, 5] = a*p.M[2, 5] + dt*(p.M[2,4])
            p.M[3, 5] = a*p.M[3, 5] + dt*(p.M[3,4])

            ## Storage of cross-sectional area
            p.M[2, 3] = a*p.M[2, 3] - dt*(
                        p.sigma[1]/(2*(p.l[1]^2+p.l[2]^2+p.l[3]^2)) * (p.l[1]*p.M[1,4]+p.l[2]*p.M[2,4]+p.l[3]*p.M[3,4]))

            ## Storeage of Gamma stretching
            p.M[1, 2] = a*p.M[1, 2] + dt*(p.circulation[1]*p.M[1,4])
            p.M[2, 2] = a*p.M[2, 2] + dt*(p.circulation[1]*p.M[2,4])
            p.M[3, 2] = a*p.M[3, 2] + dt*(p.circulation[1]*p.M[3,4])

            # Updates position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            # Update vortex tube length
            p.l[1] += b*p.M[1, 5]
            p.l[2] += b*p.M[2, 5]
            p.l[3] += b*p.M[3, 5]

            # Update cross-sectional area according to stretching
            p.sigma[1] += b*p.M[2, 3]

            # Updates vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

        end

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)

    end

    # Convert vortex tube length into vectorial circulation
    for p in iterator(pfield)
        p.Gamma .= p.l
        p.Gamma .*= p.circulation[1]
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
        pfield.UJ(pfield)

        for p in iterator(pfield)
            nrmw = sqrt(    (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                            (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                            (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
            nrml = sqrt(p.l[1]^2 + p.l[2]^2 + p.l[3]^2)

            b2 =  1 - 2*(1-pfield.rlxf)*pfield.rlxf*(1 - (
                                                            p.l[1]*(p.J[3,2]-p.J[2,3]) +
                                                            p.l[2]*(p.J[1,3]-p.J[3,1]) +
                                                            p.l[3]*(p.J[2,1]-p.J[1,2])
                                                          ) / (nrml*nrmw))

            p.l[1] = (1-pfield.rlxf)*p.l[1] + pfield.rlxf*nrml*(p.J[3,2]-p.J[2,3])/nrmw
            p.l[2] = (1-pfield.rlxf)*p.l[2] + pfield.rlxf*nrml*(p.J[1,3]-p.J[3,1])/nrmw
            p.l[3] = (1-pfield.rlxf)*p.l[3] + pfield.rlxf*nrml*(p.J[2,1]-p.J[1,2])/nrmw

            # Normalize the direction of the new vector to maintain the same strength
            p.l ./= sqrt(b2)

            # Convert relaxed vortex tube length into vectorial circulation
            p.Gamma .= p.l
            p.Gamma .*= p.circulation[1]
        end
    end

    return nothing
end
