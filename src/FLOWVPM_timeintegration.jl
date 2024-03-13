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

        C::R = get_C(p)[1]

        # Update position
        get_X(p)[1] += dt*(get_U(p)[1] + Uinf[1])
        get_X(p)[2] += dt*(get_U(p)[2] + Uinf[2])
        get_X(p)[3] += dt*(get_U(p)[3] + Uinf[3])

        # Update vectorial circulation
        ## Vortex stretching contributions
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            get_Gamma(p)[1] += dt*(get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[2]*get_Gamma(p)[2]+get_J(p)[3]*get_Gamma(p)[3])
            get_Gamma(p)[2] += dt*(get_J(p)[4]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[6]*get_Gamma(p)[3])
            get_Gamma(p)[3] += dt*(get_J(p)[7]*get_Gamma(p)[1]+get_J(p)[8]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3])
        else
            # Classic scheme (Γ⋅∇)U
            get_Gamma(p)[1] += dt*(get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[4]*get_Gamma(p)[2]+get_J(p)[7]*get_Gamma(p)[3])
            get_Gamma(p)[2] += dt*(get_J(p)[2]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[8]*get_Gamma(p)[3])
            get_Gamma(p)[3] += dt*(get_J(p)[3]*get_Gamma(p)[1]+get_J(p)[6]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3])
        end

        ## Subfilter-scale contributions -Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
        get_Gamma(p)[1] -= dt*C*get_SFS1(p) * get_sigma(p)[]^3/zeta0
        get_Gamma(p)[2] -= dt*C*get_SFS2(p) * get_sigma(p)[]^3/zeta0
        get_Gamma(p)[3] -= dt*C*get_SFS3(p) * get_sigma(p)[]^3/zeta0

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

        C::R = get_C(p)[1]

        # Update position
        get_X(p)[1] += dt*(get_U(p)[1] + Uinf[1])
        get_X(p)[2] += dt*(get_U(p)[2] + Uinf[2])
        get_X(p)[3] += dt*(get_U(p)[3] + Uinf[3])

        # Store stretching S under MM[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1] = (get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[2]*get_Gamma(p)[2]+get_J(p)[3]*get_Gamma(p)[3])
            MM[2] = (get_J(p)[4]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[6]*get_Gamma(p)[3])
            MM[3] = (get_J(p)[7]*get_Gamma(p)[1]+get_J(p)[8]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM[1] = (get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[4]*get_Gamma(p)[2]+get_J(p)[7]*get_Gamma(p)[3])
            MM[2] = (get_J(p)[2]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[8]*get_Gamma(p)[3])
            MM[3] = (get_J(p)[3]*get_Gamma(p)[1]+get_J(p)[6]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3])
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        MM[4] = (f+g)/(1+3*f) * (MM[1]*get_Gamma(p)[1] + MM[2]*get_Gamma(p)[2] + MM[3]*get_Gamma(p)[3])
        MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*get_Gamma(p)[1] + C*get_SFS2(p)*get_Gamma(p)[2] + C*get_SFS3(p)*get_Gamma(p)[3]) * get_sigma(p)[]^3/zeta0
        MM[4] /= get_Gamma(p)[1]^2 + get_Gamma(p)[2]^2 + get_Gamma(p)[3]^2

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ - Cϵ)
        get_Gamma(p)[1] += dt * (MM[1] - 3*MM[4]*get_Gamma(p)[1] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
        get_Gamma(p)[2] += dt * (MM[2] - 3*MM[4]*get_Gamma(p)[2] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
        get_Gamma(p)[3] += dt * (MM[3] - 3*MM[4]*get_Gamma(p)[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        get_sigma(p)[] -= dt * ( get_sigma(p)[] * MM[4] )

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

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); get_M(p) .= zeroR; end;

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

            C::R = get_C(p)[1]

            # Low-storage RK step
            ## Velocity
            get_M(p)[1] = a*get_M(p)[1] + dt*(get_U(p)[1] + Uinf[1])
            get_M(p)[2] = a*get_M(p)[2] + dt*(get_U(p)[2] + Uinf[2])
            get_M(p)[3] = a*get_M(p)[3] + dt*(get_U(p)[3] + Uinf[3])

            # Update position
            get_X(p)[1] += b*get_M(p)[1]
            get_X(p)[2] += b*get_M(p)[2]
            get_X(p)[3] += b*get_M(p)[3]

            ## Stretching + SFS contributions
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                get_M(p)[4] = a*get_M(p)[4] + dt*(get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[2]*get_Gamma(p)[2]+get_J(p)[3]*get_Gamma(p)[3] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
                get_M(p)[5] = a*get_M(p)[5] + dt*(get_J(p)[4]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[6]*get_Gamma(p)[3] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
                get_M(p)[6] = a*get_M(p)[6] + dt*(get_J(p)[7]*get_Gamma(p)[1]+get_J(p)[8]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)
            else
                # Classic scheme (Γ⋅∇)U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                get_M(p)[4] = a*get_M(p)[4] + dt*(get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[4]*get_Gamma(p)[2]+get_J(p)[7]*get_Gamma(p)[3] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
                get_M(p)[5] = a*get_M(p)[5] + dt*(get_J(p)[2]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[8]*get_Gamma(p)[3] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
                get_M(p)[6] = a*get_M(p)[6] + dt*(get_J(p)[3]*get_Gamma(p)[1]+get_J(p)[6]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)
            end

            # Update vectorial circulation
            get_Gamma(p)[1] += b*get_M(p)[4]
            get_Gamma(p)[2] += b*get_M(p)[5]
            get_Gamma(p)[3] += b*get_M(p)[6]

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

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7],
    #                      qsmg <=> get_M(p)[8], Z <=> MM[4], S <=> MM[1:3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); get_M(p) .= zeroR; end;

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

            C::R = get_C(p)[1]

            # Low-storage RK step
            ## Velocity
            get_M(p)[1] = a*get_M(p)[1] + dt*(get_U(p)[1] + Uinf[1])
            get_M(p)[2] = a*get_M(p)[2] + dt*(get_U(p)[2] + Uinf[2])
            get_M(p)[3] = a*get_M(p)[3] + dt*(get_U(p)[3] + Uinf[3])

            # Update position
            get_X(p)[1] += b*get_M(p)[1]
            get_X(p)[2] += b*get_M(p)[2]
            get_X(p)[3] += b*get_M(p)[3]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[2]*get_Gamma(p)[2]+get_J(p)[3]*get_Gamma(p)[3]
                MM[2] = get_J(p)[4]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[6]*get_Gamma(p)[3]
                MM[3] = get_J(p)[7]*get_Gamma(p)[1]+get_J(p)[8]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3]
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = get_J(p)[1]*get_Gamma(p)[1]+get_J(p)[4]*get_Gamma(p)[2]+get_J(p)[7]*get_Gamma(p)[3]
                MM[2] = get_J(p)[2]*get_Gamma(p)[1]+get_J(p)[5]*get_Gamma(p)[2]+get_J(p)[8]*get_Gamma(p)[3]
                MM[3] = get_J(p)[3]*get_Gamma(p)[1]+get_J(p)[6]*get_Gamma(p)[2]+get_J(p)[9]*get_Gamma(p)[3]
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*get_Gamma(p)[1] + MM[2]*get_Gamma(p)[2] + MM[3]*get_Gamma(p)[3])
            MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*get_Gamma(p)[1] + C*get_SFS2(p)*get_Gamma(p)[2] + C*get_SFS3(p)*get_Gamma(p)[3]) * get_sigma(p)[]^3/zeta0
            MM[4] /= get_Gamma(p)[1]^2 + get_Gamma(p)[2]^2 + get_Gamma(p)[3]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            get_M(p)[4] = a*get_M(p)[4] + dt*(MM[1] - 3*MM[4]*get_Gamma(p)[1] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
            get_M(p)[5] = a*get_M(p)[5] + dt*(MM[2] - 3*MM[4]*get_Gamma(p)[2] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
            get_M(p)[6] = a*get_M(p)[6] + dt*(MM[3] - 3*MM[4]*get_Gamma(p)[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            get_M(p)[8] = a*get_M(p)[8] - dt*( get_sigma(p)[] * MM[4] )

            # Update vectorial circulation
            get_Gamma(p)[1] += b*get_M(p)[4]
            get_Gamma(p)[2] += b*get_M(p)[5]
            get_Gamma(p)[3] += b*get_M(p)[6]

            # Update cross-sectional area
            get_sigma(p)[] += b*get_M(p)[8]

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
