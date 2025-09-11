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
function _euler(pfield::ParticleField{R, <:ClassicVPM, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                                dt; relax::Bool=false) where {R, V}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    Threads.@threads for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue # skip static particles

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
function _euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                               dt::Real; relax::Bool=false) where {R, V, R2}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t) # can I get rid of this annotation without breaking ReverseDiff? @Eric

    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    Threads.@threads for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue # skip static particles

        C::R = get_C(p)[1]

        # Update position
        X = get_X(p)
        U = get_U(p)
        for i in 1:3
            X[i] += dt*(U[i] + Uinf[i])
        end

        # Store stretching S
        J = get_J(p)
        G = get_Gamma(p)
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM1 = (J[1]*G[1]+J[2]*G[2]+J[3]*G[3])
            MM2 = (J[4]*G[1]+J[5]*G[2]+J[6]*G[3])
            MM3 = (J[7]*G[1]+J[8]*G[2]+J[9]*G[3])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM1 = (J[1]*G[1]+J[4]*G[2]+J[7]*G[3])
            MM2 = (J[2]*G[1]+J[5]*G[2]+J[8]*G[3])
            MM3 = (J[3]*G[1]+J[6]*G[2]+J[9]*G[3])
        end

        # Store Z under MM4 with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        Gnorm2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]
        if Gnorm2 > zero(Gnorm2)
            MM4 = (f+g)/(1+3*f) * (MM1*G[1] + MM2*G[2] + MM3*G[3])
            MM4 -= f/(1+3*f) * (C*get_SFS1(p)*G[1] + C*get_SFS2(p)*G[2] + C*get_SFS3(p)*G[3]) * get_sigma(p)[]^3/zeta0
            MM4 /= G[1]^2 + G[2]^2 + G[3]^2
        else
            MM4 = zero(Gnorm2)
        end

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ - Cϵ)
        SFS = get_SFS(p)
        sigma3 = get_sigma(p)[]^3
        G[1] += dt * (MM1 - 3*MM4*G[1] - C*SFS[1]*sigma3/zeta0)
        G[2] += dt * (MM2 - 3*MM4*G[2] - C*SFS[2]*sigma3/zeta0)
        G[3] += dt * (MM3 - 3*MM4*G[3] - C*SFS[3]*sigma3/zeta0)

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        get_sigma(p)[] -= dt * ( get_sigma(p)[] * MM4 )

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
function rungekutta3(pfield::ParticleField{R, <:ClassicVPM, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                            dt::R3; relax::Bool=false, custom_UJ=nothing) where {R, V, R3}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7]

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    zeroR::R = zero(R)
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            if pfield.particles[STATIC_INDEX,i] == 0 
                pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
            end
        end
    else
        for i in 1:pfield.np
            if pfield.particles[STATIC_INDEX,i] == 0 
                pfield.particles[M_INDEX,i] .= zeroR # this is necessary to reset the particle's M storage memory
            end
        end
    end

    # Runge-Kutta inner steps
    for (a,b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))

        # Evaluate UJ, SFS, and C
        # NOTE: UJ evaluation is NO LONGER performed inside the SFS scheme
        pfield.SFS(pfield, BeforeUJ(); a=a, b=b)
        if isnothing(custom_UJ)
            pfield.UJ(pfield; reset_sfs=true, reset=true, sfs=isSFSenabled(pfield.SFS))
        else
            custom_UJ(pfield; reset_sfs=true, reset=true, sfs=isSFSenabled(pfield.SFS))
        end
        pfield.SFS(pfield, AfterUJ(); a=a, b=b)

        # Update the particle field: convection and stretching
        update_particle_states(pfield,a,b,dt,Uinf,f, g, zeta0)

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

        if pfield.np > MIN_MT_NP
            Threads.@threads for i in 1:pfield.np
                if pfield.particles[STATIC_INDEX,i] == 0
                    pfield.relaxation(pfield, i) # this is necessary to reset the particle's M storage memory
                end
            end
        else
            for i in 1:pfield.np
                if pfield.particles[STATIC_INDEX,i] == 0
                    pfield.relaxation(pfield, i) # this is necessary to reset the particle's M storage memory
                end
            end
        end
    end

    return nothing
end


function update_particle_states(pfield::ParticleField{R, <:ClassicVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}

    if Threads.nthreads() > 1
        update_particle_states_multithreaded(pfield,a,b,dt,Uinf,f, g, zeta0)
        return nothing
    end

    for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue # skip static particles

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

    return nothing

end

function update_particle_states_multithreaded(pfield::ParticleField{R, <:ClassicVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    assignments, n = thread_assignments(pfield.np, Threads.nthreads())

    Threads.@threads :static for i_assignment in eachindex(assignments)
        i_start = assignments[i_assignment]
        i_end = min(i_start + n - 1, pfield.np)

        for i in i_start:i_end
            p = get_particle(pfield, i)
            is_static(p) && continue # skip static particles

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
function rungekutta3(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                     dt::R3; relax::Bool=false, custom_UJ=nothing) where {R, V, R2, R3}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> get_M(p)[7],
    #                      qsmg <=> get_M(p)[8], Z <=> MM4, S <=> MM[1:3]

    # Calculate freestream
    Uinf = SVector{3,R}(pfield.Uinf(pfield.t)) # now infers its type from pfield. although tbh this isn't correct; a functor for U would be a cleaner implementation.
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g # formulation floating-point type may end up as Float64 even if AD is used. (double check this)
    zeta0::Float64 = pfield.kernel.zeta(0.0) # zeta0 should have the same type as 0.0, which is Float64.

    # Reset storage memory to zero
    if pfield.np > MIN_MT_NP
        Threads.@threads for i in 1:pfield.np
            if pfield.particles[STATIC_INDEX,i] == 0 
                for j=1:3
                    pfield.particles[M_INDEX[j],i] = zero(R) # this is necessary to reset the particle's M storage memory
                end
            end
        end
    else
        for i in 1:pfield.np
            if pfield.particles[STATIC_INDEX,i] == 0
                for j=1:3
                    pfield.particles[M_INDEX[j],i] = zero(R) # this is necessary to reset the particle's M storage memory
                end
            end
        end
    end
    #=
    if R <: ReverseDiff.TrackedReal
        tp = ReverseDiff.tape(pfield)
        zeroT = zero(eltype(pfield.particles[1].value))
        for p in iterator(pfield)
            M = get_M(p)
            for i=1:8
                M[i] = ReverseDiff.track(zeroT, tp)
            end
        end
    else
        for p in iterator(pfield); get_M(p) .= zeroR; end; # this line is not safe with ReverseDiff.
    end
    =#
    # Runge-Kutta inner steps
    for (a,b) in (((0.0, 1/3)), ((-5/9, 15/16)), ((-153/128, 8/15))) # doing type conversions on fixed floating-point numbers is redundant.

        # Evaluate UJ, SFS, and C
        pfield.SFS(pfield, BeforeUJ(); a=a, b=b)
        if isnothing(custom_UJ)
            pfield.UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
        else
            custom_UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
        end
        pfield.SFS(pfield, AfterUJ(); a=a, b=b)

        # Update the particle field: convection and stretching
        update_particle_states(pfield,a,b,dt,Uinf,f, g, zeta0)

        # Update the particle field: viscous diffusion
        viscousdiffusion(pfield, dt; aux1=a, aux2=b)
    end

    # something here breaks ForwardDiff # will need to re-enable and make sure this works now. @eric I removed the comments- want to test this?
    # Relaxation: Align vectorial circulation to local vorticity
    if relax

        # Resets U and J from previous step
        _reset_particles(pfield)

        # Calculates interactions between particles: U and J
        pfield.UJ(pfield)

        if pfield.np > MIN_MT_NP
            Threads.@threads for i in 1:pfield.np
                if pfield.particles[STATIC_INDEX,i] == 0 
                    pfield.relaxation(pfield, i) # this is necessary to reset the particle's M storage memory
                end
            end
        else
            for i in 1:pfield.np
                if pfield.particles[STATIC_INDEX,i] == 0 
                    pfield.relaxation(pfield, i) # this is necessary to reset the particle's M storage memory
                end
            end
        end
    end

    return nothing
end

function update_particle_states(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}

    if Threads.nthreads() > 1
        update_particle_states_multithreaded(pfield,a,b,dt,Uinf,f, g, zeta0)
        return nothing
    end

    for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue # skip static particles

        C::R = get_C(p)[1]

        # Low-storage RK step
        ## Velocity
        M = get_M(p); G = get_Gamma(p); J = get_J(p); S = get_SFS(p); sigma = get_sigma(p)[]; X = get_X(p); U = get_U(p)
        for i=1:3
            add!(M[i], (a-1)*M[i] + dt*(U[i] + Uinf[i]))
        end

        # Update position
        for i=1:3
            add!(X[i], b*M[i])
        end

        # Store stretching S under M[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM1 = J[1]*G[1]+J[2]*G[2]+J[3]*G[3]
            MM2 = J[4]*G[1]+J[5]*G[2]+J[6]*G[3]
            MM3 = J[7]*G[1]+J[8]*G[2]+J[9]*G[3]
        else
            # Classic scheme (Γ⋅∇)U
            MM1 = J[1]*G[1]+J[4]*G[2]+J[7]*G[3]
            MM2 = J[2]*G[1]+J[5]*G[2]+J[8]*G[3]
            MM3 = J[3]*G[1]+J[6]*G[2]+J[9]*G[3]
        end

        # Store Z under MM4 with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        Gnorm2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]
        if Gnorm2 > zero(Gnorm2)
            MM4 = (f+g)/(1+3*f) * (MM1*G[1] + MM2*G[2] + MM3*G[3])
            MM4 -= f/(1+3*f) * (C*S[1]*G[1] + C*S[2]*G[2] + C*S[3]*G[3]) * sigma^3/zeta0
            MM4 /= Gnorm2
        else
            MM4 = zero(Gnorm2)
        end

        # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
        # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
        add!(M[4], (a-1)*M[4] + dt*(MM1 - 3*MM4*G[1] - C*S[1]*sigma^3/zeta0))
        add!(M[5], (a-1)*M[5] + dt*(MM2 - 3*MM4*G[2] - C*S[2]*sigma^3/zeta0))
        add!(M[6], (a-1)*M[6] + dt*(MM3 - 3*MM4*G[3] - C*S[3]*sigma^3/zeta0))

        # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
        add!(M[8], (a-1)*M[8] - dt*(sigma * MM4))

        # Update vectorial circulation
        for i=1:3
            add!(G[i], b*M[i+3])
        end

        # Update cross-sectional area
        add!(sigma, b*M[8])

    end

    return nothing

end

function thread_assignments(np::Int, nthreads::Int)
    n_per_thread, rem = divrem(np, nthreads)
    n = n_per_thread + (rem > 0)
    assignments = 1:n:np
    return assignments, n
end

function update_particle_states_multithreaded(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    assignments, n = thread_assignments(pfield.np, Threads.nthreads())
    Threads.@threads :static for i_assignment in eachindex(assignments)
        i_start = assignments[i_assignment]
        i_end = min(i_start + n - 1, pfield.np)

        for i in i_start:i_end
            p = get_particle(pfield, i)
            is_static(p) && continue # skip static particles

            C::R = get_C(p)[1]

            # Low-storage RK step
            ## Velocity
            M = get_M(p); G = get_Gamma(p); J = get_J(p); S = get_SFS(p); sigma = get_sigma(p)[]; X = get_X(p); U = get_U(p)
            for i=1:3
                add!(M[i], (a-1)*M[i] + dt*(U[i] + Uinf[i]))
            end

            # Update position
            for i=1:3
                add!(X[i], b*M[i])
            end

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM1 = J[1]*G[1]+J[2]*G[2]+J[3]*G[3]
                MM2 = J[4]*G[1]+J[5]*G[2]+J[6]*G[3]
                MM3 = J[7]*G[1]+J[8]*G[2]+J[9]*G[3]
            else
                # Classic scheme (Γ⋅∇)U
                MM1 = J[1]*G[1]+J[4]*G[2]+J[7]*G[3]
                MM2 = J[2]*G[1]+J[5]*G[2]+J[8]*G[3]
                MM3 = J[3]*G[1]+J[6]*G[2]+J[9]*G[3]
            end

            # Store Z under MM4 with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
            Gnorm2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]
            if Gnorm2 > zero(Gnorm2)
                MM4 = (f+g)/(1+3*f) * (MM1*G[1] + MM2*G[2] + MM3*G[3])
                MM4 -= f/(1+3*f) * (C*S[1]*G[1] + C*S[2]*G[2] + C*S[3]*G[3]) * sigma^3/zeta0
                MM4 /= Gnorm2
            else
                MM4 = zero(Gnorm2)
            end

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
            add!(M[4], (a-1)*M[4] + dt*(MM1 - 3*MM4*G[1] - C*S[1]*sigma^3/zeta0))
            add!(M[5], (a-1)*M[5] + dt*(MM2 - 3*MM4*G[2] - C*S[2]*sigma^3/zeta0))
            add!(M[6], (a-1)*M[6] + dt*(MM3 - 3*MM4*G[3] - C*S[3]*sigma^3/zeta0))

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            add!(M[8], (a-1)*M[8] - dt*(sigma * MM4))

            # Update vectorial circulation
            for i=1:3
                add!(G[i], b*M[i+3])
            end

            # Update cross-sectional area
            add!(sigma, b*M[8])

        end
    end

    return nothing

end