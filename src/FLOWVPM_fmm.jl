################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

function upper_bound(σ, ω, ε)
    return ω / (8 * pi * ε * σ) * (sqrt(2/pi) + sqrt(2/(pi*σ*σ) + 16 * pi * ε / ω))
end

function residual(ρ_σ, σ, ω, ε)
    t1 = 4*pi*σ*σ*ε*ρ_σ*ρ_σ / ω
    t2 = erf(ρ_σ / sqrt(2))
    t3 = sqrt(2/pi) * ρ_σ * exp(-ρ_σ*ρ_σ*0.5)
    return t1 + t2 - t3 - 1.0
end

function solve_ρ_over_σ(σ, ω, ε)
    if ω < 10*eps()
        ω = zero(ω)
    end
    return Roots.find_zero((x) -> residual(x, σ, ω, ε), (0.0, upper_bound(σ, ω, ε)), Roots.Brent())
end

function solve_ρ_over_σ(σ, ω, ε::Nothing)
    return one(σ)
end

function fmm.source_system_to_buffer!(buffer, i_buffer, system::ParticleField, i_body)
    σ = system.particles[SIGMA_INDEX, i_body]
    Γx, Γy, Γz = view(system.particles, GAMMA_INDEX, i_body)
    Γ = sqrt(Γx*Γx + Γy*Γy + Γz*Γz)
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.ε_tol)
    buffer[1:3, i_buffer] .= view(system.particles, X_INDEX, i_body)
    buffer[4, i_buffer] = ρ_σ * σ
    buffer[5:7, i_buffer] .= view(system.particles, GAMMA_INDEX, i_body)
    buffer[8, i_buffer] = σ
end

function fmm.data_per_body(system::ParticleField)
    return 8
end
#--- getters ---#

function fmm.get_position(system::ParticleField, i)
    return SVector{3}(system.particles[j,i] for j in X_INDEX)
end

function fmm.strength_dims(system::ParticleField)
    return 3
end

fmm.get_n_bodies(system::ParticleField) = system.np

function fmm.body_to_multipole!(system::ParticleField, args...)
    return fmm.body_to_multipole!(fmm.Point{fmm.Vortex}, system, args...)
end

@inline function bound_induced_velocity_fmm(r1, r2, core_size=1e-3)
    # parameters
    nr1 = norm(r1)
    nr2 = norm(r2)
    nr1nr2 = nr1*nr2
    rcross = cross(r1, r2)
    rdot = dot(r1, r2)
    # check if evaluation point is colinear with the bound vortex
    #if norm(rcross) < epsilon # colinear if true
    if isapprox(rdot, -nr1nr2; atol=eps(rdot)*100) # at the midpoint, so return zero
        return zero(typeof(r1))
    else # coincident with the filament so use the finite core model
        r1s, r2s, εs = nr1^2, nr2^2, core_size*core_size
        f1 = rcross/(r1s*r2s - rdot^2 + εs*(r1s + r2s - 2*nr1nr2))
        f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)
        l = norm(r2 .- r1)
        σ = l * 0.1
        one_over_σ = 1 / σ
        f3 = 1 .- exp(-(get_distance(r1,r2)*one_over_σ)^3)
        Vhat = (f1*f2*f3)/(4*pi)
        return Vhat
    end
end

function fmm.direct!(target_buffer, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS}

    for i_source_particle in source_index

        # gamma_x, gamma_y, gamma_z = get_Gamma(source_particle)
        gamma_x, gamma_y, gamma_z = fmm.get_strength(source_buffer, source_system, i_source_particle)
        # source_x, source_y, source_z = get_X(source_particle)
        source_x, source_y, source_z = fmm.get_position(source_buffer, i_source_particle)
        # sigma = get_sigma(source_particle)[]
        sigma = source_buffer[8, i_source_particle]

        for j_target in target_index

            target_x, target_y, target_z = fmm.get_position(target_buffer, j_target)
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz

            if !iszero(r2)
                r = sqrt(r2)

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)

                # K × Γp
                r3inv = one(r) / (r2 * r)
                crss1 = -const4 * r3inv * ( dy*gamma_z - dz*gamma_y )
                crss2 = -const4 * r3inv * ( dz*gamma_x - dx*gamma_z )
                crss3 = -const4 * r3inv * ( dx*gamma_y - dy*gamma_x )

                if VS
                    # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                    Ux = g_sgm * crss1
                    Uy = g_sgm * crss2
                    Uz = g_sgm * crss3
                    # get_U(target_particle) .+= Ux, Uy, Uz
                    Ux0, Uy0, Uz0 = fmm.get_velocity(target_buffer, j_target)

                    val = SVector{3}(Ux, Uy, Uz)
                    fmm.set_velocity!(target_buffer, j_target, val)
                end

                if GS
                    # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                    # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                    aux = dg_sgmdr/(sigma*r) - 3*g_sgm / r2
                    # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                    # Adds the Kronecker delta term
                    aux2 = -const4 * g_sgm * r3inv
                    # j=1
                    du1x1 = aux * crss1 * dx
                    du2x1 = aux * crss2 * dx - aux2 * gamma_z
                    du3x1 = aux * crss3 * dx + aux2 * gamma_y
                    # j=2
                    du1x2 = aux * crss1 * dy + aux2 * gamma_z
                    du2x2 = aux * crss2 * dy
                    du3x2 = aux * crss3 * dy - aux2 * gamma_x
                    # j=3
                    du1x3 = aux * crss1 * dz - aux2 * gamma_y
                    du2x3 = aux * crss2 * dz + aux2 * gamma_x
                    du3x3 = aux * crss3 * dz
                    # @show aux, aux2, crss1, crss2, crss3, dx, dy, dz
                    # @show du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3

                    val = SMatrix{3,3}(du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3)
                    fmm.set_velocity_gradient!(target_buffer, j_target, val)
                end
            end
        end
    end

    return nothing
end

function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch, target_buffer, i_buffer)
    target_system.particles[U_INDEX, i_target] .+= fmm.get_velocity(target_buffer, i_buffer)
    j = fmm.get_velocity_gradient(target_buffer, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target] += j[i]
    end
end

Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where TF = TF
