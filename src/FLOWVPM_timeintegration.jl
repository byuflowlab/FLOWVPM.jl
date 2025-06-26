#=##############################################################################
# DESCRIPTION
    Time integration schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

"""
    euler(pfield::ParticleField, dt::Real; relax::Bool=false, custom_UJ=nothing)

Convects the `pfield` by timestep `dt` using a forward Euler step.

# Arguments
- `pfield::ParticleField` The particle field to integrate.
- `dt::Real` The time step.
- `relax::Bool` Whether to apply relaxation (default: false).
- `custom_UJ` Optional custom function for updating U and J.

"""
function euler(pfield::ParticleField, dt; relax::Bool=false, custom_UJ=nothing)

    # Evaluate UJ, SFS, and C
    # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
    pfield.SFS(pfield, BeforeUJ())
    if isnothing(custom_UJ)
        pfield.UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    else
        custom_UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    end

    _euler(pfield, dt; relax)

    return nothing
end

"""
Steps the field forward in time by dt in a first-order Euler integration scheme.
"""
function _euler(pfield::ParticleField{R, <:ClassicVPM, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},
                                dt; relax::Bool=false) where {R, V}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        C::R = get_C(p)[1]

        # Update position
        get_X(p) .+= dt*(get_U(p) .+ Uinf)

        # Update vectorial circulation
        ## Vortex stretching contributions
        J = get_J(p)
        G = get_Gamma(p)
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            G[1] += dt*(J[1]*G[1]+J[2]*G[2]+J[3]*G[3])
            G[2] += dt*(J[4]*G[1]+J[5]*G[2]+J[6]*G[3])
            G[3] += dt*(J[7]*G[1]+J[8]*G[2]+J[9]*G[3])
        else
            # Classic scheme (Γ⋅∇)U
            G[1] += dt*(J[1]*G[1]+J[4]*G[2]+J[7]*G[3])
            G[2] += dt*(J[2]*G[1]+J[5]*G[2]+J[8]*G[3])
            G[3] += dt*(J[3]*G[1]+J[6]*G[2]+J[9]*G[3])
        end

        ## Subfilter-scale contributions -Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
        G .-= dt*C*get_SFS(p) * get_sigma(p)[]^3/zeta0

        # Relaxation: Align vectorial circulation to local vorticity
        if relax
            pfield.relaxation(p)
        end

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

end







"""
Steps the field forward in time by dt in a first-order Euler integration scheme
using the VPM reformulation. See notebook 20210104.
"""
function _euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},
                               dt::Real; relax::Bool=false) where {R, V, R2}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t) # can I get rid of this annotation without breaking ReverseDiff? @Eric
    # Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM = pfield.M # @Eric
    # MM::Array{<:Real, 1} = pfield.M

    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for (i_p,p) in enumerate(iterator(pfield))

        C::R = get_C(p)[1]

        # Update position
        X = get_X(p)
        U = get_U(p)
        for i in 1:3
            X[i] += dt*(U[i] + Uinf[i])
        end
        # get_X(p) .+= dt*(get_U(p) .+ Uinf)

        # Store stretching S under MM[1:3]
        J = get_J(p)
        G = get_Gamma(p)
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1] = (J[1]*G[1]+J[2]*G[2]+J[3]*G[3])
            MM[2] = (J[4]*G[1]+J[5]*G[2]+J[6]*G[3])
            MM[3] = (J[7]*G[1]+J[8]*G[2]+J[9]*G[3])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM[1] = (J[1]*G[1]+J[4]*G[2]+J[7]*G[3])
            MM[2] = (J[2]*G[1]+J[5]*G[2]+J[8]*G[3])
            MM[3] = (J[3]*G[1]+J[6]*G[2]+J[9]*G[3])
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        MM[4] = (f+g)/(1+3*f) * (MM[1]*G[1] + MM[2]*G[2] + MM[3]*G[3])
        MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*G[1] + C*get_SFS2(p)*G[2] + C*get_SFS3(p)*G[3]) * get_sigma(p)[]^3/zeta0
        MM[4] /= G[1]^2 + G[2]^2 + G[3]^2

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ - Cϵ)
        SFS = get_SFS(p)
        sigma3 = get_sigma(p)[]^3
        for i in 1:3
            G[i] += dt * (MM[i] - 3*MM[4]*G[i] - C*SFS[i]*sigma3/zeta0)
        end
        # G .+= dt * (MM[1:3] - 3*MM[4]*G - C*get_SFS(p)*get_sigma(p)[]^3/zeta0)

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        get_sigma(p)[] -= dt * ( get_sigma(p)[] * MM[4] )

        # Relaxation: Align vectorial circulation to local vorticity

        if relax
            pfield.relaxation(p)
        end
    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

end







"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{R, <:ClassicVPM, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},
                            dt::R3; relax::Bool=false, custom_UJ=nothing) where {R, V, R3}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7]

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); get_M(p) .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))

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
            M = get_M(p); G = get_Gamma(p); J = get_J(p)
            ## Velocity
            M[1] = a*M[1] + dt*(get_U(p)[1] + Uinf[1])
            M[2] = a*M[2] + dt*(get_U(p)[2] + Uinf[2])
            M[3] = a*M[3] + dt*(get_U(p)[3] + Uinf[3])

            # Update position
            get_X(p)[1] += b*M[1]
            get_X(p)[2] += b*M[2]
            get_X(p)[3] += b*M[3]

            ## Stretching + SFS contributions
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                M[4] = a*M[4] + dt*(J[1]*G[1]+J[2]*G[2]+J[3]*G[3] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
                M[5] = a*M[5] + dt*(J[4]*G[1]+J[5]*G[2]+J[6]*G[3] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
                M[6] = a*M[6] + dt*(J[7]*G[1]+J[8]*G[2]+J[9]*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)
            else
                # Classic scheme (Γ⋅∇)U - Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
                M[4] = a*M[4] + dt*(J[1]*G[1]+J[4]*G[2]+J[7]*G[3] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
                M[5] = a*M[5] + dt*(J[2]*G[1]+J[5]*G[2]+J[8]*G[3] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
                M[6] = a*M[6] + dt*(J[3]*G[1]+J[6]*G[2]+J[9]*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)
            end

            # Update vectorial circulation
            G[1] += b*M[4]
            G[2] += b*M[5]
            G[3] += b*M[6]

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

    rungekutta3(pfield::ParticleField, dt::Real; relax::Bool=false, custom_UJ=nothing)

Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme using the VPM reformulation. See Notebook entry 20180105
(RK integration) and notebook 20210104 (reformulation).

# Arguments
- `pfield::ParticleField` The particle field to integrate.
- `dt::R3` The time step.
- `relax::Bool` Whether to apply relaxation (default: false).
- `custom_UJ` Optional custom function for updating U and J.

"""
function rungekutta3(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},
                     dt::R3; relax::Bool=false, custom_UJ=nothing) where {R, V, R2, R3}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7],
    #                      qsmg <=> get_M(p)[8], Z <=> MM[4], S <=> MM[1:3]

    # Calculate freestream
    # Uinf::Array{R, 1} = R.(pfield.Uinf(pfield.t)) # now infers its type from pfield. although tbh this isn't correct; a functor for U would be a cleaner implementation.
    Uinf = SVector{3,R}(pfield.Uinf(pfield.t)) # now infers its type from pfield. although tbh this isn't correct; a functor for U would be a cleaner implementation.

    MM = pfield.M # eltype(pfield.M) = R
    # MM::Array{R, 1} = pfield.M # eltype(pfield.M) = R
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g # formulation floating-point type may end up as Float64 even if AD is used. (double check this)
    #zeta0::R = pfield.kernel.zeta(0)
    zeta0::Float64 = pfield.kernel.zeta(0.0) # zeta0 should have the same type as 0.0, which is Float64.
    # Reset storage memory to zero
    zeroR::R = zero(R)
    for p in iterator(pfield); get_M(p) .= zeroR; end;

    # Runge-Kutta inner steps
    for (a,b) in (((0.0, 1/3)), ((-5/9, 15/16)), ((-153/128, 8/15))) # doing type conversions on fixed floating-point numbers is redundant.

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
        #println("tape entries before SFS 1: $(length(ReverseDiff.tape(pfield.particles[1].X[1])) - l)")
        #l = length(ReverseDiff.tape(pfield.particles[1].X[1]))
        pfield.SFS(pfield, BeforeUJ(); a=a, b=b)
        if isnothing(custom_UJ)
            pfield.UJ(pfield; reset_sfs=true, reset=true, sfs=pfield.toggle_sfs)
        else
            custom_UJ(pfield; reset_sfs=true, reset=true, sfs=pfield.toggle_sfs)
        end
        pfield.SFS(pfield, AfterUJ(); a=a, b=b)
        #println("tape entries after SFS 2/before time marching: $(length(ReverseDiff.tape(pfield.particles[1].X[1])) - l)")
        #l = length(ReverseDiff.tape(pfield.particles[1].X[1]))
        # Update the particle field: convection and stretching
        update_particle_states(pfield,MM,a,b,dt,Uinf,f, g, zeta0)

        #=for p in iterator(pfield)

            C::R = get_C(p)[1]

            # Low-storage RK step
            ## Velocity
            M = get_M(p); G = get_Gamma(p); J = get_J(p)
            M[1] = a*M[1] + dt*(get_U(p)[1] + Uinf[1])
            M[2] = a*M[2] + dt*(get_U(p)[2] + Uinf[2])
            M[3] = a*M[3] + dt*(get_U(p)[3] + Uinf[3])

            # Update position
            get_X(p)[1] += b*M[1]
            get_X(p)[2] += b*M[2]
            get_X(p)[3] += b*M[3]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = J[1]*G[1]+J[2]*G[2]+J[3]*G[3]
                MM[2] = J[4]*G[1]+J[5]*G[2]+J[6]*G[3]
                MM[3] = J[7]*G[1]+J[8]*G[2]+J[9]*G[3]
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = J[1]*G[1]+J[4]*G[2]+J[7]*G[3]
                MM[2] = J[2]*G[1]+J[5]*G[2]+J[8]*G[3]
                MM[3] = J[3]*G[1]+J[6]*G[2]+J[9]*G[3]
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*G[1] + MM[2]*G[2] + MM[3]*G[3])
            MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*G[1] + C*get_SFS2(p)*G[2] + C*get_SFS3(p)*G[3]) * get_sigma(p)[]^3/zeta0
            MM[4] /= G[1]^2 + G[2]^2 + G[3]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            M[4] = a*M[4] + dt*(MM[1] - 3*MM[4]*G[1] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
            M[5] = a*M[5] + dt*(MM[2] - 3*MM[4]*G[2] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
            M[6] = a*M[6] + dt*(MM[3] - 3*MM[4]*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            M[8] = a*M[8] - dt*( get_sigma(p)[] * MM[4] )

            # Update vectorial circulation
            G[1] += b*M[4]
            G[2] += b*M[5]
            G[3] += b*M[6]

            # Update cross-sectional area
            get_sigma(p)[] += b*M[8]

        end=#

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)
        #println("tape entries after diffusion: $(length(ReverseDiff.tape(pfield.particles[1].X[1])) - l)")
        #l = length(ReverseDiff.tape(pfield.particles[1].X[1]))

    end

    # something here breaks ForwardDiff # will need to re-enable and make sure this works now. @eric I removed the comments- want to test this?
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

    #println("tape entries after time step: $(length(ReverseDiff.tape(pfield.particles[1].X[1])) - l)")
    #l = length(ReverseDiff.tape(pfield.particles[1].X[1]))
    #println("")
    return nothing
end


function update_particle_states(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},MM,a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}

    for p in iterator(pfield)

        C::R = get_C(p)[1]

            # Low-storage RK step
            ## Velocity
            M = get_M(p); G = get_Gamma(p); J = get_J(p)
            M[1] = a*M[1] + dt*(get_U(p)[1] + Uinf[1])
            M[2] = a*M[2] + dt*(get_U(p)[2] + Uinf[2])
            M[3] = a*M[3] + dt*(get_U(p)[3] + Uinf[3])

            # Update position
            get_X(p)[1] += b*M[1]
            get_X(p)[2] += b*M[2]
            get_X(p)[3] += b*M[3]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = J[1]*G[1]+J[2]*G[2]+J[3]*G[3]
                MM[2] = J[4]*G[1]+J[5]*G[2]+J[6]*G[3]
                MM[3] = J[7]*G[1]+J[8]*G[2]+J[9]*G[3]
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = J[1]*G[1]+J[4]*G[2]+J[7]*G[3]
                MM[2] = J[2]*G[1]+J[5]*G[2]+J[8]*G[3]
                MM[3] = J[3]*G[1]+J[6]*G[2]+J[9]*G[3]
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*G[1] + MM[2]*G[2] + MM[3]*G[3])
            MM[4] -= f/(1+3*f) * (C*get_SFS1(p)*G[1] + C*get_SFS2(p)*G[2] + C*get_SFS3(p)*G[3]) * get_sigma(p)[]^3/zeta0
            MM[4] /= G[1]^2 + G[2]^2 + G[3]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            M[4] = a*M[4] + dt*(MM[1] - 3*MM[4]*G[1] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
            M[5] = a*M[5] + dt*(MM[2] - 3*MM[4]*G[2] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
            M[6] = a*M[6] + dt*(MM[3] - 3*MM[4]*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            M[8] = a*M[8] - dt*( get_sigma(p)[] * MM[4] )

            # Update vectorial circulation
            G[1] += b*M[4]
            G[2] += b*M[5]
            G[3] += b*M[6]

            # Update cross-sectional area
            get_sigma(p)[] += b*M[8]

    end

    return nothing

end
