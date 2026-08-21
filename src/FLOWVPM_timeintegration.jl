#=##############################################################################
# DESCRIPTION
    Time integration schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

"""
    _reset_M_storage!(pfield)

Zero the per-particle `M` storage rows (the RK3 low-storage state) of every
non-static particle. CPU (`Array`-backed particles): the original scalar loop,
unchanged. GPU (any other backing store): scalar indexing is disallowed on a
`CuArray`, so the (0/1) static-mask broadcast is used instead — same pattern
as `_reset_particles_broadcast!`.
"""
function _reset_M_storage!(pfield::ParticleField)
    zeroR = zero(eltype(pfield.particles))
    if pfield.particles isa Array
        for i in 1:pfield.np
            if pfield.particles[STATIC_INDEX, i] == 0
                pfield.particles[M_INDEX, i] .= zeroR
            end
        end
    else
        np = pfield.np
        is_static = view(pfield.particles, STATIC_INDEX:STATIC_INDEX, 1:np)
        view(pfield.particles, M_INDEX, 1:np) .*= is_static
    end
    return nothing
end

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

    # CPU (Array-backed particles): original zero-allocation scalar loop, unchanged.
    # GPU (CuArray, or any other non-Array backing store): broadcast-based implementation,
    # which works unchanged for CuArray but allocates and is not competitive on CPU (see
    # logs/2026-07-21-gpu-full.md for the benchmark that motivated this split).
    if pfield.particles isa Array
        _euler_cpu_classic!(pfield, dt, Uinf, zeta0)
    else
        _euler_broadcast_classic!(pfield, dt, Uinf, zeta0)
    end

    # Relaxation: Align vectorial circulation to local vorticity
    if relax
        if pfield.particles isa Array
            for i in 1:pfield.np
                pfield.particles[STATIC_INDEX, i] == 0 && pfield.relaxation(get_particle(pfield, i))
            end
        else
            relax_broadcast!(pfield.relaxation, pfield)
        end
    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

end

"CPU path for `_euler` (ClassicVPM): original per-particle scalar loop, unchanged from pre-Phase-1 FLOWVPM."
function _euler_cpu_classic!(pfield::ParticleField{R}, dt, Uinf, zeta0) where R
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
    end
    return nothing
end

"GPU-compatible path for `_euler` (ClassicVPM): broadcasts over row-slices, works unchanged for `CuArray`."
function _euler_broadcast_classic!(pfield::ParticleField, dt, Uinf, zeta0)

    # Static-particle mask: 0 for static, 1 for active
    active = 1.0 .- pfield.particles[STATIC_INDEX, :]

    # Update the particle field: convection and stretching
    # Position: X += dt*(U + Uinf)
    pfield.particles[X_INDEX, :] .+= (dt .* active') .* (pfield.particles[U_INDEX, :] .+ reshape(Uinf, 3, 1))

    # Vortex stretching contributions and SFS dissipation
    J = view(pfield.particles, J_INDEX, :)  # (9, np)
    G = view(pfield.particles, GAMMA_INDEX, :)  # (3, np), mutated below -- must be a view, not a copy
    C_all = view(pfield.particles, C_INDEX, :)  # (3, np) — C is stored as 3 components per particle
    SFS_all = view(pfield.particles, SFS_INDEX, :)  # (3, np)
    sigma3 = pfield.particles[SIGMA_INDEX, :] .^ 3  # (np,)

    if pfield.transposed
        # Transposed scheme (Γ⋅∇')U
        # Sequential read-after-write: G[1] updated, then used in computing G[2], etc.
        G[1, :] .+= dt .* active .* (J[1, :] .* G[1, :] .+ J[2, :] .* G[2, :] .+ J[3, :] .* G[3, :])
        G[2, :] .+= dt .* active .* (J[4, :] .* G[1, :] .+ J[5, :] .* G[2, :] .+ J[6, :] .* G[3, :])
        G[3, :] .+= dt .* active .* (J[7, :] .* G[1, :] .+ J[8, :] .* G[2, :] .+ J[9, :] .* G[3, :])
    else
        # Classic scheme (Γ⋅∇)U
        G[1, :] .+= dt .* active .* (J[1, :] .* G[1, :] .+ J[4, :] .* G[2, :] .+ J[7, :] .* G[3, :])
        G[2, :] .+= dt .* active .* (J[2, :] .* G[1, :] .+ J[5, :] .* G[2, :] .+ J[8, :] .* G[3, :])
        G[3, :] .+= dt .* active .* (J[3, :] .* G[1, :] .+ J[6, :] .* G[2, :] .+ J[9, :] .* G[3, :])
    end

    # Subfilter-scale contributions: -Cϵ where ϵ=(Eadv + Estr)/zeta_sgmp(0)
    # Note: C is stored as 3 components; use first component per original code
    # C_all[1,:] and sigma3 are (np,) -- transpose to (1,np) row vectors so they
    # broadcast against SFS_all's (3,np) shape along the particle axis, not the component axis
    G .-= dt .* active' .* (C_all[1, :]' .* SFS_all .* sigma3' ./ zeta0)

    return nothing
end







"""
Steps the field forward in time by dt in a first-order Euler integration scheme
using the VPM reformulation. See notebook 20210104.
"""
function _euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                               dt::Real; relax::Bool=false) where {R, V, R2}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # CPU (Array-backed particles): original zero-allocation scalar loop, unchanged.
    # GPU (CuArray, or any other non-Array backing store): broadcast-based implementation,
    # which works unchanged for CuArray but allocates and is not competitive on CPU (see
    # logs/2026-07-21-gpu-full.md for the benchmark that motivated this split).
    if pfield.particles isa Array
        _euler_cpu_reformulated!(pfield, dt, Uinf, f, g, zeta0)
    else
        _euler_broadcast_reformulated!(pfield, dt, Uinf, f, g, zeta0)
    end

    # Relaxation: Align vectorial circulation to local vorticity
    if relax
        if pfield.particles isa Array
            for i in 1:pfield.np
                pfield.particles[STATIC_INDEX, i] == 0 && pfield.relaxation(get_particle(pfield, i))
            end
        else
            relax_broadcast!(pfield.relaxation, pfield)
        end
    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

end

"CPU path for `_euler` (ReformulatedVPM): original per-particle scalar loop, unchanged from pre-Phase-1 FLOWVPM."
function _euler_cpu_reformulated!(pfield::ParticleField{R}, dt, Uinf, f::R2, g::R2, zeta0) where {R, R2}
    for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue # skip static particles

        C::R = get_C(p)[1]

        # Update position
        X = get_X(p)
        U = get_U(p)
        for k in 1:3
            X[k] += dt*(U[k] + Uinf[k])
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
    end
    return nothing
end

"GPU-compatible path for `_euler` (ReformulatedVPM): broadcasts over row-slices, works unchanged for `CuArray`."
function _euler_broadcast_reformulated!(pfield::ParticleField{R}, dt, Uinf, f::R2, g::R2, zeta0) where {R, R2}

    # Static-particle mask: 0 for static, 1 for active
    active = 1.0 .- pfield.particles[STATIC_INDEX, :]

    # Update the particle field: convection and stretching
    # Position: X += dt*(U + Uinf)
    pfield.particles[X_INDEX, :] .+= (dt .* active') .* (pfield.particles[U_INDEX, :] .+ reshape(Uinf, 3, 1))

    # Compute stretching S and Z for each particle
    J = view(pfield.particles, J_INDEX, :)  # (9, np)
    G = view(pfield.particles, GAMMA_INDEX, :)  # (3, np), mutated below -- must be a view, not a copy
    C_all = view(pfield.particles, C_INDEX, :)  # (3, np)
    SFS_all = view(pfield.particles, SFS_INDEX, :)  # (3, np)
    sigma = pfield.particles[SIGMA_INDEX, :]  # (np,) read-only snapshot, sigma updated separately below
    sigma3 = sigma .^ 3  # (np,)

    if pfield.transposed
        # Transposed scheme S = (Γ⋅∇')U
        MM1 = J[1, :] .* G[1, :] .+ J[2, :] .* G[2, :] .+ J[3, :] .* G[3, :]
        MM2 = J[4, :] .* G[1, :] .+ J[5, :] .* G[2, :] .+ J[6, :] .* G[3, :]
        MM3 = J[7, :] .* G[1, :] .+ J[8, :] .* G[2, :] .+ J[9, :] .* G[3, :]
    else
        # Classic scheme S = (Γ⋅∇)U
        MM1 = J[1, :] .* G[1, :] .+ J[4, :] .* G[2, :] .+ J[7, :] .* G[3, :]
        MM2 = J[2, :] .* G[1, :] .+ J[5, :] .* G[2, :] .+ J[8, :] .* G[3, :]
        MM3 = J[3, :] .* G[1, :] .+ J[6, :] .* G[2, :] .+ J[9, :] .* G[3, :]
    end

    # Compute Z: [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2
    Gnorm2 = G[1, :] .* G[1, :] .+ G[2, :] .* G[2, :] .+ G[3, :] .* G[3, :]
    S_dot_G = MM1 .* G[1, :] .+ MM2 .* G[2, :] .+ MM3 .* G[3, :]
    C_eps_dot_G = C_all[1, :] .* (SFS_all[1, :] .* G[1, :] .+ SFS_all[2, :] .* G[2, :] .+ SFS_all[3, :] .* G[3, :]) .* sigma3 ./ zeta0
    MM4 = ((f + g) / (1 + 3*f) .* S_dot_G .- f / (1 + 3*f) .* C_eps_dot_G) ./ max.(Gnorm2, eps(R))
    MM4 .= ifelse.(Gnorm2 .> zero(R), MM4, zero(R))

    # Update vectorial circulation: ΔΓ = Δt*(S - 3ZΓ - Cϵ)
    # Sequential read-after-write: G[1] updated first, then used in G[2], G[3]
    G[1, :] .+= dt .* active .* (MM1 .- 3 .* MM4 .* G[1, :] .- C_all[1, :] .* SFS_all[1, :] .* sigma3 ./ zeta0)
    G[2, :] .+= dt .* active .* (MM2 .- 3 .* MM4 .* G[2, :] .- C_all[1, :] .* SFS_all[2, :] .* sigma3 ./ zeta0)
    G[3, :] .+= dt .* active .* (MM3 .- 3 .* MM4 .* G[3, :] .- C_all[1, :] .* SFS_all[3, :] .* sigma3 ./ zeta0)

    # Update cross-sectional area of the tube σ = -Δt*σ*Z
    pfield.particles[SIGMA_INDEX, :] .-= dt .* active .* (sigma .* MM4)

    return nothing
end


"""
    euler_exp(pfield::ParticleField, dt::Real; relax::Bool=false, custom_UJ=nothing)

Frozen-gradient geometric integrator for the reformulated VPM with `f == 0`.

Holding the velocity gradient `L` fixed, first evolve `q' = L*q` exactly and
define `r = norm(q(dt))/norm(Γ(0))`.  The homogeneous rVPM geometry is then

    Γ(dt) = q(dt) * r^(-3g),
    σ(dt) = σ(0) * r^(-g).

For the live `g = 1/5` formulation this preserves the intended stretching
source: aligned strain amplifies Γ by `exp(2*dt*Z)`, rather than freezing the
initial stretching vector as a constant forcing.  Positive σ is preserved for
every finite gradient and timestep.  SFS forcing is applied afterward as an
explicit first-order Lie split.  `CoreSpreading` uses the step's effective
constant `Z = g*log(r)/dt` to integrate strain and diffusion together; other
viscous schemes retain their existing post-step update.
"""
function euler_exp(pfield::ParticleField, dt; relax::Bool=false, custom_UJ=nothing)

    # Evaluate UJ, SFS, and C
    pfield.SFS(pfield, BeforeUJ())
    if isnothing(custom_UJ)
        pfield.UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    else
        custom_UJ(pfield; reset_sfs=isSFSenabled(pfield.SFS), reset=true, sfs=isSFSenabled(pfield.SFS))
    end

    _euler_exp(pfield, dt; relax)

    return nothing
end

"""
Steps the field forward in time by dt with the exponential (exact-in-Z) local
update using the VPM reformulation. See `euler_exp` and `_euler`.
"""
function _euler_exp(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},
                               dt::Real; relax::Bool=false) where {R, V, R2}

    pfield.SFS(pfield, AfterUJ())

    # Calculate freestream
    Uinf = pfield.Uinf(pfield.t)

    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    f == zero(f) || throw(ArgumentError(
        "euler_exp geometric update currently requires ReformulatedVPM f == 0; got f=$f"))
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

        # Build the frozen stretching operator L such that S = L*Gamma.
        J = get_J(p)
        G = get_Gamma(p)
        if pfield.transposed
            L = @SMatrix [J[1] J[2] J[3];
                          J[4] J[5] J[6];
                          J[7] J[8] J[9]]
        else
            L = @SMatrix [J[1] J[4] J[7];
                          J[2] J[5] J[8];
                          J[3] J[6] J[9]]
        end

        G0 = SVector{3,R}(G[1], G[2], G[3])
        Gnorm2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]
        if Gnorm2 > zero(Gnorm2)
            q = exp(dt*L)*G0
            ratio = sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) / sqrt(Gnorm2)
            isfinite(ratio) && ratio > zero(ratio) || throw(DomainError(ratio,
                "non-finite frozen-gradient strength ratio in euler_exp"))
            gamma_scale = ratio^(-3*g)
            G[1] = q[1]*gamma_scale
            G[2] = q[2]*gamma_scale
            G[3] = q[3]*gamma_scale
            get_sigma(p)[] *= ratio^(-g)
            # M[9] is private scratch for the euler_exp/CoreSpreading
            # composition. It stores the constant rate with the same total
            # contraction over this step: sigma(dt)/sigma(0)=exp(-dt*Zeff).
            get_M(p)[9] = dt == zero(dt) ? zero(R) : g*log(ratio)/dt
        else
            get_M(p)[9] = zero(R)
        end

        # Existing SFS contribution, isolated as an explicit first-order Lie
        # split because it is an additive modeled forcing rather than part of
        # the homogeneous frozen-gradient geometry above.
        SFS = get_SFS(p)
        sigma3 = get_sigma(p)[]^3
        G[1] -= dt*C*SFS[1]*sigma3/zeta0
        G[2] -= dt*C*SFS[2]*sigma3/zeta0
        G[3] -= dt*C*SFS[3]*sigma3/zeta0

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
    _reset_M_storage!(pfield)

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

        if pfield.particles isa Array
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
        else
            relax_broadcast!(pfield.relaxation, pfield)
        end
    end

    return nothing
end


function update_particle_states(pfield::ParticleField{R, <:ClassicVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    # CPU (Array-backed particles): original zero-allocation scalar loop, unchanged.
    # GPU (CuArray, or any other non-Array backing store): broadcast-based implementation,
    # which works unchanged for CuArray but allocates and is not competitive on CPU (see
    # logs/2026-07-21-gpu-full.md for the benchmark that motivated this split).
    if pfield.particles isa Array
        update_particle_states_cpu_classic!(pfield,a,b,dt,Uinf,f,g,zeta0)
    else
        update_particle_states_broadcast_classic!(pfield,a,b,dt,Uinf,f,g,zeta0)
    end
    return nothing
end

"CPU path for RK3's `update_particle_states` (ClassicVPM): original per-particle scalar loop, unchanged from pre-Phase-1 FLOWVPM."
function update_particle_states_cpu_classic!(pfield::ParticleField{R, <:ClassicVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue

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
            M[4] = a*M[4] + dt*(J[1]*G[1]+J[2]*G[2]+J[3]*G[3] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
            M[5] = a*M[5] + dt*(J[4]*G[1]+J[5]*G[2]+J[6]*G[3] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
            M[6] = a*M[6] + dt*(J[7]*G[1]+J[8]*G[2]+J[9]*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)
        else
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

"GPU-compatible path for RK3's `update_particle_states` (ClassicVPM): broadcasts over row-slices with a preallocated scratch buffer, works unchanged for `CuArray`."
function update_particle_states_broadcast_classic!(pfield::ParticleField{R, <:ClassicVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}

    # All reads below are single-row *views* into pfield.particles/scratch (zero-copy);
    # every computed intermediate is written in-place into a persistent scratch row via
    # `.=` instead of allocating a fresh array each call.
    P = pfield.particles
    Sc = pfield.scratch

    static = view(P, STATIC_INDEX, :)
    active = view(Sc, 8, :); active .= 1.0 .- static  # row 8: free here (only rows 1-7 used below), no conflict with ReformulatedVPM's own row numbering since they never share a call
    isactive = active .> 0

    U1, U2, U3 = view(P, U_INDEX[1], :), view(P, U_INDEX[2], :), view(P, U_INDEX[3], :)
    J1,J2,J3,J4,J5,J6,J7,J8,J9 = (view(P, J_INDEX[k], :) for k in 1:9)
    C1 = view(P, C_INDEX[1], :)
    SFS1, SFS2, SFS3 = view(P, SFS_INDEX[1], :), view(P, SFS_INDEX[2], :), view(P, SFS_INDEX[3], :)
    sigma = view(P, SIGMA_INDEX, :)

    M1, M2, M3, M4, M5, M6 = (view(P, M_INDEX[k], :) for k in 1:6)
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)
    X1, X2, X3 = view(P, X_INDEX[1], :), view(P, X_INDEX[2], :), view(P, X_INDEX[3], :)

    sigma3 = view(Sc, 7, :); sigma3 .= sigma .^ 3
    M1_new, M2_new, M3_new = view(Sc, 1, :), view(Sc, 2, :), view(Sc, 3, :)
    M4_new, M5_new, M6_new = view(Sc, 4, :), view(Sc, 5, :), view(Sc, 6, :)

    ## Velocity
    M1_new .= a .* M1 .+ dt .* (U1 .+ Uinf[1])
    M2_new .= a .* M2 .+ dt .* (U2 .+ Uinf[2])
    M3_new .= a .* M3 .+ dt .* (U3 .+ Uinf[3])

    ## Stretching + SFS contributions
    if pfield.transposed
        M4_new .= a .* M4 .+ dt .* (J1.*G1 .+ J2.*G2 .+ J3.*G3 .- C1 .* SFS1 .* sigma3 ./ zeta0)
        M5_new .= a .* M5 .+ dt .* (J4.*G1 .+ J5.*G2 .+ J6.*G3 .- C1 .* SFS2 .* sigma3 ./ zeta0)
        M6_new .= a .* M6 .+ dt .* (J7.*G1 .+ J8.*G2 .+ J9.*G3 .- C1 .* SFS3 .* sigma3 ./ zeta0)
    else
        M4_new .= a .* M4 .+ dt .* (J1.*G1 .+ J4.*G2 .+ J7.*G3 .- C1 .* SFS1 .* sigma3 ./ zeta0)
        M5_new .= a .* M5 .+ dt .* (J2.*G1 .+ J5.*G2 .+ J8.*G3 .- C1 .* SFS2 .* sigma3 ./ zeta0)
        M6_new .= a .* M6 .+ dt .* (J3.*G1 .+ J6.*G2 .+ J9.*G3 .- C1 .* SFS3 .* sigma3 ./ zeta0)
    end

    # Position/circulation deltas are zero for static particles via the active mask
    X1 .+= active .* b .* M1_new
    X2 .+= active .* b .* M2_new
    X3 .+= active .* b .* M3_new

    G1 .+= active .* b .* M4_new
    G2 .+= active .* b .* M5_new
    G3 .+= active .* b .* M6_new

    # Static particles keep their previous M storage (frozen), others get the new RK stage value
    M1 .= ifelse.(isactive, M1_new, M1)
    M2 .= ifelse.(isactive, M2_new, M2)
    M3 .= ifelse.(isactive, M3_new, M3)
    M4 .= ifelse.(isactive, M4_new, M4)
    M5 .= ifelse.(isactive, M5_new, M5)
    M6 .= ifelse.(isactive, M6_new, M6)

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
    _reset_M_storage!(pfield)

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

        if pfield.particles isa Array
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
        else
            relax_broadcast!(pfield.relaxation, pfield)
        end
    end

    return nothing
end

function update_particle_states(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    # CPU (Array-backed particles): original zero-allocation scalar loop, unchanged.
    # GPU (CuArray, or any other non-Array backing store): broadcast-based implementation,
    # which works unchanged for CuArray but allocates and is not competitive on CPU (see
    # logs/2026-07-21-gpu-full.md for the benchmark that motivated this split).
    if pfield.particles isa Array
        update_particle_states_cpu_reformulated!(pfield,a,b,dt,Uinf,f,g,zeta0)
    else
        update_particle_states_broadcast_reformulated!(pfield,a,b,dt,Uinf,f,g,zeta0)
    end
    return nothing
end

"CPU path for RK3's `update_particle_states` (ReformulatedVPM): original per-particle scalar loop, unchanged from pre-Phase-1 FLOWVPM."
function update_particle_states_cpu_reformulated!(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}
    for i in 1:pfield.np
        p = get_particle(pfield, i)
        is_static(p) && continue

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
            MM1 = J[1]*G[1]+J[2]*G[2]+J[3]*G[3]
            MM2 = J[4]*G[1]+J[5]*G[2]+J[6]*G[3]
            MM3 = J[7]*G[1]+J[8]*G[2]+J[9]*G[3]
        else
            MM1 = J[1]*G[1]+J[4]*G[2]+J[7]*G[3]
            MM2 = J[2]*G[1]+J[5]*G[2]+J[8]*G[3]
            MM3 = J[3]*G[1]+J[6]*G[2]+J[9]*G[3]
        end

        # Store Z under MM4 with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        Gnorm2 = G[1]*G[1] + G[2]*G[2] + G[3]*G[3]
        if Gnorm2 > zero(Gnorm2)
            MM4 = (f+g)/(1+3*f) * (MM1*G[1] + MM2*G[2] + MM3*G[3])
            MM4 -= f/(1+3*f) * (C*get_SFS1(p)*G[1] + C*get_SFS2(p)*G[2] + C*get_SFS3(p)*G[3]) * get_sigma(p)[]^3/zeta0
            MM4 /= Gnorm2
        else
            MM4 = zero(Gnorm2)
        end

        # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
        # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )
        M[4] = a*M[4] + dt*(MM1 - 3*MM4*G[1] - C*get_SFS1(p)*get_sigma(p)[]^3/zeta0)
        M[5] = a*M[5] + dt*(MM2 - 3*MM4*G[2] - C*get_SFS2(p)*get_sigma(p)[]^3/zeta0)
        M[6] = a*M[6] + dt*(MM3 - 3*MM4*G[3] - C*get_SFS3(p)*get_sigma(p)[]^3/zeta0)

        # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
        M[8] = a*M[8] - dt*( get_sigma(p)[] * MM4 )

        # Update vectorial circulation
        G[1] += b*M[4]
        G[2] += b*M[5]
        G[3] += b*M[6]

        # Update cross-sectional area
        get_sigma(p)[] += b*M[8]
    end
    return nothing
end

"GPU-compatible path for RK3's `update_particle_states` (ReformulatedVPM): broadcasts over row-slices with a preallocated scratch buffer, works unchanged for `CuArray`."
function update_particle_states_broadcast_reformulated!(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any},a,b,dt::R3,Uinf,f,g,zeta0) where {R, R2, V, R3}

    # All reads below are single-row *views* into pfield.particles/scratch (zero-copy);
    # every computed intermediate is written in-place into a persistent scratch row via
    # `.=` instead of allocating a fresh array each call.
    #
    # Row reuse (11 rows total, not 16): MM1/MM2/MM3 are dead right after they're
    # consumed by M4_new/M5_new/M6_new, so M4_new/M5_new/M6_new are written directly
    # into MM1/MM2/MM3's own rows (`x .= f(x, ...)` -- safe because every formula here
    # is elementwise per-particle: output at index i depends only on inputs at index i).
    # Same for Gnorm2 -> MM4 (Gnorm2 dead once MM4 is computed from it) and
    # S_dot_G -> M8_new (S_dot_G dead once MM4 is computed). Ceps_dot_G has no later
    # reuse opportunity (nothing needs a fresh row after it dies) and keeps its own row.
    P = pfield.particles
    Sc = pfield.scratch

    static = view(P, STATIC_INDEX, :)
    active = view(Sc, 11, :); active .= 1.0 .- static
    isactive = active .> 0

    U1, U2, U3 = view(P, U_INDEX[1], :), view(P, U_INDEX[2], :), view(P, U_INDEX[3], :)
    J1,J2,J3,J4,J5,J6,J7,J8,J9 = (view(P, J_INDEX[k], :) for k in 1:9)
    C1 = view(P, C_INDEX[1], :)
    SFS1, SFS2, SFS3 = view(P, SFS_INDEX[1], :), view(P, SFS_INDEX[2], :), view(P, SFS_INDEX[3], :)
    sigma = view(P, SIGMA_INDEX, :)

    M1, M2, M3, M4, M5, M6, M8 = view(P, M_INDEX[1], :), view(P, M_INDEX[2], :), view(P, M_INDEX[3], :), view(P, M_INDEX[4], :), view(P, M_INDEX[5], :), view(P, M_INDEX[6], :), view(P, M_INDEX[8], :)
    G1, G2, G3 = view(P, GAMMA_INDEX[1], :), view(P, GAMMA_INDEX[2], :), view(P, GAMMA_INDEX[3], :)
    X1, X2, X3 = view(P, X_INDEX[1], :), view(P, X_INDEX[2], :), view(P, X_INDEX[3], :)
    sigma_v = view(P, SIGMA_INDEX, :)

    M1_new, M2_new, M3_new = view(Sc, 1, :), view(Sc, 2, :), view(Sc, 3, :)
    MM1, MM2, MM3 = view(Sc, 4, :), view(Sc, 5, :), view(Sc, 6, :)
    Gnorm2, S_dot_G, Ceps_dot_G = view(Sc, 7, :), view(Sc, 8, :), view(Sc, 9, :)
    sigma3 = view(Sc, 10, :); sigma3 .= sigma .^ 3

    ## Velocity
    M1_new .= a .* M1 .+ dt .* (U1 .+ Uinf[1])
    M2_new .= a .* M2 .+ dt .* (U2 .+ Uinf[2])
    M3_new .= a .* M3 .+ dt .* (U3 .+ Uinf[3])

    # Store stretching S
    if pfield.transposed
        MM1 .= J1.*G1 .+ J2.*G2 .+ J3.*G3
        MM2 .= J4.*G1 .+ J5.*G2 .+ J6.*G3
        MM3 .= J7.*G1 .+ J8.*G2 .+ J9.*G3
    else
        MM1 .= J1.*G1 .+ J4.*G2 .+ J7.*G3
        MM2 .= J2.*G1 .+ J5.*G2 .+ J8.*G3
        MM3 .= J3.*G1 .+ J6.*G2 .+ J9.*G3
    end

    # Store Z under MM4 (reuses Gnorm2's row -- Gnorm2 is only read here, on its own row, once)
    Gnorm2 .= G1.^2 .+ G2.^2 .+ G3.^2
    S_dot_G .= MM1 .* G1 .+ MM2 .* G2 .+ MM3 .* G3
    Ceps_dot_G .= C1 .* (SFS1.*G1 .+ SFS2.*G2 .+ SFS3.*G3) .* sigma3 ./ zeta0
    MM4 = Gnorm2  # alias: Gnorm2's row now holds MM4 from here on
    # NOTE: the Gnorm2>0 test and the division by Gnorm2 must be ONE fused broadcast
    # statement, not two -- Gnorm2 and MM4 are the same row, so a second statement
    # reading `Gnorm2` after MM4's first assignment would read the already-overwritten
    # MM4 value instead of the original squared-norm (this was a real bug, caught by
    # a direct loop-vs-broadcast numerical diff, not by `] test`).
    MM4 .= ifelse.(Gnorm2 .> zero(R), ((f+g)/(1+3*f) .* S_dot_G .- f/(1+3*f) .* Ceps_dot_G) ./ Gnorm2, zero(R))

    # Store qstr_i (M4_new/M5_new/M6_new reuse MM1/MM2/MM3's rows -- each MMk is read
    # here for the last time, then immediately overwritten by the corresponding Mk_new)
    M4_new, M5_new, M6_new = MM1, MM2, MM3
    M4_new .= a .* M4 .+ dt .* (MM1 .- 3 .* MM4 .* G1 .- C1 .* SFS1 .* sigma3 ./ zeta0)
    M5_new .= a .* M5 .+ dt .* (MM2 .- 3 .* MM4 .* G2 .- C1 .* SFS2 .* sigma3 ./ zeta0)
    M6_new .= a .* M6 .+ dt .* (MM3 .- 3 .* MM4 .* G3 .- C1 .* SFS3 .* sigma3 ./ zeta0)

    # Store qsgm_i (M8_new reuses S_dot_G's row -- S_dot_G is dead once MM4 was computed above)
    M8_new = S_dot_G
    M8_new .= a .* M8 .- dt .* (sigma .* MM4)

    # Position/circulation/sigma deltas are zero for static particles via the active mask
    X1 .+= active .* b .* M1_new
    X2 .+= active .* b .* M2_new
    X3 .+= active .* b .* M3_new

    G1 .+= active .* b .* M4_new
    G2 .+= active .* b .* M5_new
    G3 .+= active .* b .* M6_new

    sigma_v .+= active .* b .* M8_new

    # Static particles keep their previous M storage (frozen), others get the new RK stage value
    M1 .= ifelse.(isactive, M1_new, M1)
    M2 .= ifelse.(isactive, M2_new, M2)
    M3 .= ifelse.(isactive, M3_new, M3)
    M4 .= ifelse.(isactive, M4_new, M4)
    M5 .= ifelse.(isactive, M5_new, M5)
    M6 .= ifelse.(isactive, M6_new, M6)
    M8 .= ifelse.(isactive, M8_new, M8)

    return nothing

end
