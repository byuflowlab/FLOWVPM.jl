################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = particle_field.particles[i].var[1:3]
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = particle_field.particles[i].var[7]
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.VectorPotential) where R = MVector{3,R}(0.0,0.0,0.0)
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.ScalarPotential) where R = zero(R)
Base.getindex(particle_field::ParticleField, i, ::fmm.VectorStrength) = particle_field.particles[i].var[4:6]
Base.getindex(particle_field::ParticleField, i, ::fmm.Velocity) = particle_field.particles[i].var[10:12]
Base.getindex(particle_field::ParticleField, i, ::fmm.VelocityGradient) = reshape(particle_field.particles[i].var[16:24], (3, 3))
Base.getindex(particle_field::ParticleField, i) = particle_field.particles[i]

Base.setindex!(particle_field::ParticleField, val, i) = particle_field.particles[i] = val
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.ScalarPotential) = nothing
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VectorPotential) = nothing
function Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Velocity)
    particle_field.particles[i].var[10:12] .= val[1:3]
end
Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VelocityGradient) = particle_field.particles[i].var[16:24] .= reshape(val, 9)

fmm.get_n_bodies(particle_field::ParticleField) = particle_field.np

Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where TF = TF

fmm.buffer_element(system::ParticleField) = deepcopy(system.particles[1])

fmm.B2M!(system::ParticleField, args...) = fmm.B2M!_vortexpoint(system, args...)

@inline function vorticity_direct(target_system::ParticleField, target_index, source_system, source_index)
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target, fmm.POSITION]
        Wx = zero(eltype(target_system))
        Wy = zero(eltype(target_system))
        Wz = zero(eltype(target_system))
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = source_system.particles[i_source].var[4:6]
            source_x, source_y, source_z = source_system.particles[i_source].var[1:3]
            sigma = source_system.particles[i_source].var[7]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r = sqrt(dx*dx + dy*dy + dz*dz)
            sigma = source_system.particles[i_source].var[7]
            zeta = source_system.zeta(r/sigma)/(sigma*sigma*sigma)
            Wx += zeta * gamma_x
            Wy += zeta * gamma_y
            Wz += zeta * gamma_z
        end
        target_system.particles[j_target].var[13:15] .+= Wx, Wy, Wz
    end
end

function fmm.direct!(target_system::ParticleField, target_index, source_system::ParticleField, source_index)
    if source_system.toggle_rbf
        vorticity_direct(target_system, target_index, source_system, source_index)
    else
        r = zero(eltype(source_system))
        for target_particle in view(target_system.particles,target_index)
            target_x, target_y, target_z = target_particle.var[1:3]
            for source_particle in view(source_system.particles,source_index)
                gamma_x, gamma_y, gamma_z = source_particle.var[4:6]
                source_x, source_y, source_z = source_particle.var[1:3]
                sigma = source_particle.var[7]
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
                    target_particle.var[10:12] .+= Ux, Uy, Uz

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

                    target_particle.var[16:24] .+= du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3
                end

                # include self-induced contribution to SFS
                if source_system.toggle_sfs
                    Estr_direct(target_particle::Particle, source_particle::Particle, r, source_system.kernel.zeta, source_system.transposed)
                end
            end
        end
    end
end

function fmm.direct!(target_system, target_index, source_system::ParticleField, source_index)
    for j_target in target_index
        du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3 = (zero(eltype(target_system)) for _ in 1:9)
        Ux, Uy, Uz = (zero(eltype(target_system)) for _ in 1:3)
        target_x, target_y, target_z = target_system[j_target,fmm.POSITION]
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = source_system.particles[i_source].var[4:6]
            source_x, source_y, source_z = source_system.particles[i_source].var[1:3]
            sigma = source_system.particles[i_source].var[8]
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
                Ux += g_sgm * crss1
                Uy += g_sgm * crss2
                Uz += g_sgm * crss3

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r^2
                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux2 = -const4 * g_sgm / r^3
                # j=1
                du1x1 += aux * crss1 * dx
                du2x1 += aux * crss2 * dx - aux2 * gamma_z
                du3x1 += aux * crss3 * dx + aux2 * gamma_y
                # j=2
                du1x2 += aux * crss1 * dy + aux2 * gamma_z
                du2x2 += aux * crss2 * dy
                du3x2 += aux * crss3 * dy - aux2 * gamma_x
                # j=3
                du1x3 += aux * crss1 * dz - aux2 * gamma_y
                du2x3 += aux * crss2 * dz + aux2 * gamma_x
                du3x3 += aux * crss3 * dz
            end
        end
        target_system[j_target,fmm.VELOCITY] += SVector{3,typeof(Ux)}(Ux, Uy, Uz)
        target_system[j_target,fmm.VELOCITY_GRADIENT] += SMatrix{3,3,typeof(du1x1),9}(du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3)
    end
end
