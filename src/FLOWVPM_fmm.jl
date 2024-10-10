################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = get_X(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = get_sigma(particle_field, i)[]
#Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.VectorPotential) where R = SVector{3,R}(0.0,0.0,0.0) # If this breaks AD: replace with 'zeros(3,R)'
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.ScalarPotential) where R = zero(R)
Base.getindex(particle_field::ParticleField, i, ::fmm.Strength) = get_Gamma(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.Velocity) = get_U(particle_field, i)
Base.getindex(particle_field::ParticleField, i, ::fmm.VelocityGradient) = reshape(get_J(particle_field, i), (3, 3))
Base.getindex(particle_field::ParticleField, i, ::fmm.Body) = get_particle(particle_field, i)

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Body) = get_particle(particle_field, i) .= val

Base.setindex!(particle_field::ParticleField, val, i, ::fmm.ScalarPotential) = nothing
#Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VectorPotential) = nothing
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Velocity) = set_U(particle_field, i, val)
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VelocityGradient) = set_J(particle_field, i, vec(val))

fmm.get_n_bodies(particle_field::ParticleField) = get_np(particle_field)
Base.length(particle_field::ParticleField) = get_np(particle_field) # currently called internally by the version of the FMM I'm using. this will need to be changed to work with ImplicitAD, which probably just means getting the latest FMM version. that's on hold because there are a bunch of other breaking changes I'll need to deal with to get correct derivative again.

Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where TF = TF

fmm.buffer_element(system::ParticleField) = zeros(eltype(system.particles), size(system.particles, 1))

fmm.body_to_multipole!(system::ParticleField, args...) = fmm.body_to_multipole!(fmm.Point{fmm.Vortex}, system, args...)

@inline function vorticity_direct(target_system::ParticleField, target_index, source_system, source_index)
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target, fmm.POSITION]
        Wx = zero(eltype(target_system))
        Wy = zero(eltype(target_system))
        Wz = zero(eltype(target_system))
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = get_Gamma(source_system, i_source)
            source_x, source_y, source_z = get_X(source_system, i_source)
            sigma = get_sigma(source_system, i_source)[]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz # sqrt hahs an undefined derivative at r=0, so AD gets NaNs introduced without this check.
            if r2 > 0
                r = sqrt(r2)
                zeta = source_system.zeta(r/sigma)/(sigma*sigma*sigma)
                Wx += zeta * gamma_x
                Wy += zeta * gamma_y
                Wz += zeta * gamma_z
            end
        end
        get_vorticity(target_system, j_target) .+= Wx, Wy, Wz
    end
end

@inline function vorticity_direct(target_system, target_index, source_system, source_index)
    return nothing
end

@inline function Estr_direct(target_system::ParticleField, j_target, source_particle, r, zeta, transposed)
    Estr_direct(target_system[j_target, fmm.BODY], source_particle, r, zeta, transposed)
end

@inline function Estr_direct(target_system, j_target, source_particle, r, zeta, transposed)
    return nothing
end

function fmm.direct!(target_system, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_index) where {PS,VS,GS}
    if source_system.toggle_rbf
        vorticity_direct(target_system, target_index, source_system, source_index)
    else
        r = zero(eltype(source_system))

        for j_target in target_index
            target_x, target_y, target_z = target_system[j_target, fmm.POSITION]

            for source_particle in eachcol(view(source_system.particles, :, source_index))
                gamma_x, gamma_y, gamma_z = get_Gamma(source_particle)
                source_x, source_y, source_z = get_X(source_particle)
                sigma = get_sigma(source_particle)[]
                dx = target_x - source_x
                dy = target_y - source_y
                dz = target_z - source_z
                r2 = dx*dx + dy*dy + dz*dz
                if !iszero(r2)
                    r = sqrt(r2)
                    # Regularizing function and deriv
                    g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)

                    # K × Γp
                    crss1 = -const4 / r^3 * ( dy*gamma_z - dz*gamma_y )
                    crss2 = -const4 / r^3 * ( dz*gamma_x - dx*gamma_z )
                    crss3 = -const4 / r^3 * ( dx*gamma_y - dy*gamma_x )

                    # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                    Ux = g_sgm * crss1
                    Uy = g_sgm * crss2
                    Uz = g_sgm * crss3
                    # get_U(target_particle) .+= Ux, Uy, Uz
                    Ux0, Uy0, Uz0 = target_system[j_target, fmm.VELOCITY]
                    target_system[j_target, fmm.VELOCITY] = Ux+Ux0, Uy+Uy0, Uz+Uz0

                    # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                    # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                    aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r^2
                    # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                    # Adds the Kronecker delta term
                    aux2 = -const4 * g_sgm / r^3
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

                    du1x10, du2x10, du3x10, du1x20, du2x20, du3x20, du1x30, du2x30, du3x30 = target_system[j_target, fmm.VELOCITY_GRADIENT]
                    target_system[j_target, fmm.VELOCITY_GRADIENT] = SMatrix{3,3}(
                        du1x10 + du1x1,
                        du2x10 + du2x1,
                        du3x10 + du3x1,
                        du1x20 + du1x2,
                        du2x20 + du2x2,
                        du3x20 + du3x2,
                        du1x30 + du1x3,
                        du2x30 + du2x3,
                        du3x30 + du3x3
                    )
                end

                # include self-induced contribution to SFS
                if source_system.toggle_sfs
                    Estr_direct(target_system, j_target, source_particle, r, source_system.kernel.zeta, source_system.transposed)
                end
            end
        end
    end
end
