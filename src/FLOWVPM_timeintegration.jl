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
function euler(pfield::ParticleField{R, <:ClassicVPM, V, <:SubFilterScale, <:Any, <:Any, <:Any},
                                dt::Real; relax::Bool=false, custom_UJ=nothing) where {R, V}

    # Evaluate UJ, SFS, and C
    # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
    pfield.SFS(pfield, BeforeUJ())
    if isnothing(custom_UJ)
        pfield.UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    else
        custom_UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    end
    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        C::R = p[37]

        # Update position
        p[1] += dt*(p[10] + Uinf[1])
        p[2] += dt*(p[11] + Uinf[2])
        p[3] += dt*(p[12] + Uinf[3])

        # Update vectorial circulation
        ## Vortex stretching contributions
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p[4] += dt*(p[16]*p[4]+p[17]*p[5]+p[18]*p[6])
            p[5] += dt*(p[19]*p[4]+p[20]*p[5]+p[21]*p[6])
            p[6] += dt*(p[22]*p[4]+p[23]*p[5]+p[24]*p[6])
        else
            # Classic scheme (Γ⋅∇)U
            p[4] += dt*(p[16]*p[4]+p[19]*p[5]+p[22]*p[6])
            p[5] += dt*(p[17]*p[4]+p[20]*p[5]+p[23]*p[6])
            p[6] += dt*(p[18]*p[4]+p[21]*p[5]+p[24]*p[6])
        end

        ## Subfilter-scale contributions -Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
        p[4] -= dt*C*get_SFS1(p) * p[7]^3/zeta0
        p[5] -= dt*C*get_SFS2(p) * p[7]^3/zeta0
        p[6] -= dt*C*get_SFS3(p) * p[7]^3/zeta0

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
function euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale, <:Any, <:Any, <:Any},
                              dt::Real; relax::Bool=false, custom_UJ=nothing) where {R, V, R2}
    # Evaluate UJ, SFS, and C
    # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
    pfield.SFS(pfield, BeforeUJ())
    if isnothing(custom_UJ)
        pfield.UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    else
        custom_UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    end
    pfield.SFS(pfield, AfterUJ())
    
    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        C::R = p[37]

        # Update position
        p[1] += dt*(p[10] + Uinf[1])
        p[2] += dt*(p[11] + Uinf[2])
        p[3] += dt*(p[12] + Uinf[3])

        # Store stretching S under MM[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1] = (p[16]*p[4]+p[17]*p[5]+p[18]*p[6])
            MM[2] = (p[19]*p[4]+p[20]*p[5]+p[21]*p[6])
            MM[3] = (p[22]*p[4]+p[23]*p[5]+p[24]*p[6])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM[1] = (p[16]*p[4]+p[19]*p[5]+p[22]*p[6])
            MM[2] = (p[17]*p[4]+p[20]*p[5]+p[23]*p[6])
            MM[3] = (p[18]*p[4]+p[21]*p[5]+p[24]*p[6])
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        MM[4] = (f+g)/(1+3*f) * (MM[1]*p[4] + MM[2]*p[5] + MM[3]*p[6])
        MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*p[4] + C*get_SFS2(p)*p[5] + C*get_SFS3(p)*p[6]) * p[7]^3/zeta0
        MM[4] /= p[4]^2 + p[5]^2 + p[6]^2

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ - Cϵ)
        p[4] += dt * (MM[1] - 3*MM[4]*p[4] - C*get_SFS1(p)*p[7]^3/zeta0)
        p[5] += dt * (MM[2] - 3*MM[4]*p[5] - C*get_SFS2(p)*p[7]^3/zeta0)
        p[6] += dt * (MM[3] - 3*MM[4]*p[6] - C*get_SFS3(p)*p[7]^3/zeta0)

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        p[7] -= dt * ( p[7] * MM[4] )

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
function rungekutta3(pfield::ParticleField{R, <:ClassicVPM, V, <:SubFilterScale, <:Any, <:Any, <:Any},
                            dt::Real; relax::Bool=false, custom_UJ=nothing) where {R, V}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p[34]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); p[28:36] .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
        pfield.SFS(pfield, BeforeUJ(); a=a, b=b)
        if isnothing(custom_UJ)
            pfield.UJ(pfield; reset_sfs=true, reset=true, sfs=true)
        else
            custom_UJ(pfield; reset_sfs=true, reset=true, sfs=true)
        end
        pfield.SFS(pfield, AfterUJ(); a=a, b=b)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            C::R = p[37]

            # Low-storage RK step
            ## Velocity
            p[28] = a*p[28] + dt*(p[10] + Uinf[1])
            p[29] = a*p[29] + dt*(p[11] + Uinf[2])
            p[30] = a*p[30] + dt*(p[12] + Uinf[3])

            # Update position
            p[1] += b*p[28]
            p[2] += b*p[29]
            p[3] += b*p[30]

            ## Stretching + SFS contributions
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                p[31] = a*p[31] + dt*(p[16]*p[4]+p[17]*p[5]+p[18]*p[6] - C*get_SFS1(p)*p[7]^3/zeta0)
                p[32] = a*p[32] + dt*(p[19]*p[4]+p[20]*p[5]+p[21]*p[6] - C*get_SFS2(p)*p[7]^3/zeta0)
                p[33] = a*p[33] + dt*(p[22]*p[4]+p[23]*p[5]+p[24]*p[6] - C*get_SFS3(p)*p[7]^3/zeta0)
            else
                # Classic scheme (Γ⋅∇)U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                p[31] = a*p[31] + dt*(p[16]*p[4]+p[19]*p[5]+p[22]*p[6] - C*get_SFS1(p)*p[7]^3/zeta0)
                p[32] = a*p[32] + dt*(p[17]*p[4]+p[20]*p[5]+p[23]*p[6] - C*get_SFS2(p)*p[7]^3/zeta0)
                p[33] = a*p[33] + dt*(p[18]*p[4]+p[21]*p[5]+p[24]*p[6] - C*get_SFS3(p)*p[7]^3/zeta0)
            end

            # Update vectorial circulation
            p[4] += b*p[31]
            p[5] += b*p[32]
            p[6] += b*p[33]

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
function rungekutta3(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale, <:Any, <:Any, <:Any},
                     dt::Real; relax::Bool=false, custom_UJ=nothing) where {R, V, R2}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p[34],
    #                      qsmg <=> p[35], Z <=> MM[4], S <=> MM[1:3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); p[28:36] .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
        pfield.SFS(pfield, BeforeUJ(); a=a, b=b)
        if isnothing(custom_UJ)
            pfield.UJ(pfield; reset_sfs=true, reset=true, sfs=true)
        else
            custom_UJ(pfield; reset_sfs=true, reset=true, sfs=true)
        end
        pfield.SFS(pfield, AfterUJ(); a=a, b=b)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            C::R = p[37]

            # Low-storage RK step
            ## Velocity
            p[28] = a*p[28] + dt*(p[10] + Uinf[1])
            p[29] = a*p[29] + dt*(p[11] + Uinf[2])
            p[30] = a*p[30] + dt*(p[12] + Uinf[3])

            # Update position
            p[1] += b*p[28]
            p[2] += b*p[29]
            p[3] += b*p[30]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = p[16]*p[4]+p[17]*p[5]+p[18]*p[6]
                MM[2] = p[19]*p[4]+p[20]*p[5]+p[21]*p[6]
                MM[3] = p[22]*p[4]+p[23]*p[5]+p[24]*p[6]
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = p[16]*p[4]+p[19]*p[5]+p[22]*p[6]
                MM[2] = p[17]*p[4]+p[20]*p[5]+p[23]*p[6]
                MM[3] = p[18]*p[4]+p[21]*p[5]+p[24]*p[6]
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*p[4] + MM[2]*p[5] + MM[3]*p[6])
            MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*p[4] + C*get_SFS2(p)*p[5] + C*get_SFS3(p)*p[6]) * p[7]^3/zeta0
            MM[4] /= p[4]^2 + p[5]^2 + p[6]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            p[31] = a*p[31] + dt*(MM[1] - 3*MM[4]*p[4] - C*get_SFS1(p)*p[7]^3/zeta0)
            p[32] = a*p[32] + dt*(MM[2] - 3*MM[4]*p[5] - C*get_SFS2(p)*p[7]^3/zeta0)
            p[33] = a*p[33] + dt*(MM[3] - 3*MM[4]*p[6] - C*get_SFS3(p)*p[7]^3/zeta0)

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            p[35] = a*p[35] - dt*( p[7] * MM[4] )

            # Update vectorial circulation
            p[4] += b*p[31]
            p[5] += b*p[32]
            p[6] += b*p[33]

            # Update cross-sectional area
            p[7] += b*p[35]

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
