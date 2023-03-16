#=##############################################################################
# DESCRIPTION
    Time integration schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

"""
Steps the field forward in time by dt in a first-order Euler integration scheme.
"""
function euler(pfield::ParticleField{R, <:ClassicVPM, V, <:SubFilterScale},
                                dt::Real; relax::Bool=false) where {R, V}

    # Evaluate UJ, SFS, and C
    # NOTE: UJ evaluation is now performed inside the SFS scheme
    pfield.SFS(pfield)

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        C::R = p.C[1]

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Update vectorial circulation
        ## Vortex stretching contributions
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

        ## Subfilter-scale contributions -Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
        p.Gamma[1] -= dt*C*get_SFS1(p) * p.sigma[1]^3/zeta0
        p.Gamma[2] -= dt*C*get_SFS2(p) * p.sigma[1]^3/zeta0
        p.Gamma[3] -= dt*C*get_SFS3(p) * p.sigma[1]^3/zeta0

        # Relaxation: Align vectorial circulation to local vorticity
        if relax
            pfield.relaxation(p)
        end

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end









"""
Steps the field forward in time by dt in a first-order Euler integration scheme
using the VPM reformulation. See notebook 20210104.
"""
function euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale},
                              dt::Real; relax::Bool=false ) where {R, V, R2}

    # Evaluate UJ, SFS, and C
    # NOTE: UJ evaluation is now performed inside the SFS scheme
    pfield.SFS(pfield)

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        C::R = p.C[1]

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Store stretching S under MM[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
            MM[2] = (p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
            MM[3] = (p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
            MM[2] = (p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
            MM[3] = (p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        MM[4] = (f+g)/(1+3*f) * (MM[1]*p.Gamma[1] + MM[2]*p.Gamma[2] + MM[3]*p.Gamma[3])
        MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*p.Gamma[1] + C*get_SFS2(p)*p.Gamma[2] + C*get_SFS3(p)*p.Gamma[3]) * p.sigma[1]^3/zeta0
        MM[4] /= p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ - Cϵ)
        p.Gamma[1] += dt * (MM[1] - 3*MM[4]*p.Gamma[1] - C*get_SFS1(p)*p.sigma[1]^3/zeta0)
        p.Gamma[2] += dt * (MM[2] - 3*MM[4]*p.Gamma[2] - C*get_SFS2(p)*p.sigma[1]^3/zeta0)
        p.Gamma[3] += dt * (MM[3] - 3*MM[4]*p.Gamma[3] - C*get_SFS3(p)*p.sigma[1]^3/zeta0)

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        p.sigma[1] -= dt * ( p.sigma[1] * MM[4] )

        # Relaxation: Alig vectorial circulation to local vorticity
        if relax
            pfield.relaxation(p)
        end

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end












"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{R, <:ClassicVPM, V, <:SubFilterScale},
                            dt::Real; relax::Bool=false) where {R, V}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); p.M .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is now performed inside the SFS scheme
        pfield.SFS(pfield; a=a, b=b)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            C::R = p.C[1]

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            # Update position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            ## Stretching + SFS contributions
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3] - C*get_SFS1(p)*p.sigma[1]^3/zeta0)
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3] - C*get_SFS2(p)*p.sigma[1]^3/zeta0)
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3] - C*get_SFS3(p)*p.sigma[1]^3/zeta0)
            else
                # Classic scheme (Γ⋅∇)U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3] - C*get_SFS1(p)*p.sigma[1]^3/zeta0)
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3] - C*get_SFS2(p)*p.sigma[1]^3/zeta0)
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3] - C*get_SFS3(p)*p.sigma[1]^3/zeta0)
            end

            # Update vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

        end

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)

    end


    # Relaxation: Align vectorial circulation to local vorticity
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
            # Align particle strength
            pfield.relaxation(p)
        end
    end

    return nothing
end












"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme using the VPM reformulation. See Notebook entry 20180105
(RK integration) and notebook 20210104 (reformulation).
"""
function rungekutta3(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale},
                     dt::Real; relax::Bool=false ) where {R, V, R2}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3],
    #                      qsmg <=> p.M[2, 3], Z <=> MM[4], S <=> MM[1:3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); p.M .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is now performed inside the SFS scheme
        pfield.SFS(pfield; a=a, b=b)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            C::R = p.C[1]

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            # Update position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3]
                MM[2] = p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3]
                MM[3] = p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3]
                MM[2] = p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3]
                MM[3] = p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3]
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*p.Gamma[1] + MM[2]*p.Gamma[2] + MM[3]*p.Gamma[3])
            MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*p.Gamma[1] + C*get_SFS2(p)*p.Gamma[2] + C*get_SFS3(p)*p.Gamma[3]) * p.sigma[1]^3/zeta0
            MM[4] /= p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            p.M[1, 2] = a*p.M[1, 2] + dt*(MM[1] - 3*MM[4]*p.Gamma[1] - C*get_SFS1(p)*p.sigma[1]^3/zeta0)
            p.M[2, 2] = a*p.M[2, 2] + dt*(MM[2] - 3*MM[4]*p.Gamma[2] - C*get_SFS2(p)*p.sigma[1]^3/zeta0)
            p.M[3, 2] = a*p.M[3, 2] + dt*(MM[3] - 3*MM[4]*p.Gamma[3] - C*get_SFS3(p)*p.sigma[1]^3/zeta0)

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            p.M[2, 3] = a*p.M[2, 3] - dt*( p.sigma[1] * MM[4] )

            # Update vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

            # Update cross-sectional area
            p.sigma[1] += b*p.M[2, 3]

        end

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)

    end


    # Relaxation: Align vectorial circulation to local vorticity
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
            # Align particle strength
            pfield.relaxation(p)
        end
    end

    return nothing
end