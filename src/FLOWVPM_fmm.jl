################################################################################
# FMM COMPATIBILITY FUNCTION
################################################################################

Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = particle_field.particles[i].X
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = particle_field.particles[i].sigma[1]
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.VectorPotential) where {R} = zeros(R,3)#MVector{3,R}(0.0,0.0,0.0)
Base.getindex(particle_field::ParticleField{R,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any}, i, ::fmm.ScalarPotential) where {R} = zero(R)
Base.getindex(particle_field::ParticleField, i, ::fmm.VectorStrength) = particle_field.particles[i].Gamma
Base.getindex(particle_field::ParticleField, i, ::fmm.Velocity) = particle_field.particles[i].U
Base.getindex(particle_field::ParticleField, i, ::fmm.VelocityGradient) = particle_field.particles[i].J
Base.getindex(particle_field::ParticleField, i) = particle_field.particles[i]
function Base.setindex!(particle_field::ParticleField, val, i)
    particle_field.particles[i] = val
end
function Base.setindex!(particle_field::ParticleField, val, i, ::fmm.ScalarPotential)
    nothing
end
function Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VectorPotential)
    nothing
end
function Base.setindex!(particle_field::ParticleField, val, i, ::fmm.Velocity)
    particle_field.particles[i].U .= val
end
function Base.setindex!(particle_field::ParticleField, val, i, ::fmm.VelocityGradient)
    particle_field.particles[i].J .= val
end
Base.length(particle_field::ParticleField) = particle_field.np
Base.eltype(::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any}) where {TF} = TF

function fmm.buffer_element(system::ParticleField)
    return deepcopy(system.particles[1])
end

fmm.B2M!(system::ParticleField, args...) = fmm.B2M!_vortexpoint(system, args...)

@inline function vorticity_direct(target_system::ParticleField, target_index, source_system, source_index)
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target, fmm.POSITION]
        Wx = zero(eltype(target_system))
        Wy = zero(eltype(target_system))
        Wz = zero(eltype(target_system))
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = source_system.particles[i_source].Gamma
            source_x, source_y, source_z = source_system.particles[i_source].X
            sigma = source_system.particles[i_source].sigma[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0
                r = sqrt(r2)
                sigma = source_system.particles[i_source].sigma[1]
                zeta = source_system.zeta(r/sigma)/(sigma*sigma*sigma)
                Wx += zeta * gamma_x
                Wy += zeta * gamma_y
                Wz += zeta * gamma_z
            end
        end
        target_system.particles[j_target].W .+= Wx, Wy, Wz
    end
end

using ReverseDiff
function fmm.direct!(target_system::ParticleField{R,F,V,S,Tkernel,TUJ,Tintegration}, target_index, source_system::ParticleField{R,F,V,S,Tkernel,TUJ,Tintegration}, source_index) where {R,F,V,S,Tkernel,TUJ,Tintegration}
    #@show typeof(target_system) typeof(source_system)
    if source_system.toggle_rbf
        vorticity_direct(target_system, target_index, source_system, source_index)
    else
        for target_particle in view(target_system.particles,target_index)
            J = reshape(target_particle.J,9)
            for source_particle in view(source_system.particles,source_index)
                sigma = source_particle.sigma[1]
                r2 = UJ_direct!(J,target_particle.X,source_particle.X,source_particle.Gamma, source_system.kernel, target_particle.U, sigma)

                # include self-induced contribution to SFS
                if source_system.toggle_sfs
                    source_system.SFS.model(target_particle::Particle, source_particle::Particle, r2 > 0 ? sqrt(r2) : 0.0, source_system.kernel.zeta, source_system.transposed)
                end
            end
        end
    end
end

function UJ_direct!(J,xyz_target, xyz_source, gamma_source, kernel_source, U_target, sigma)

    target_x, target_y, target_z = xyz_target
    source_x, source_y, source_z = xyz_source
    gamma_x, gamma_y, gamma_z = gamma_source
    dx = target_x - source_x
    dy = target_y - source_y
    dz = target_z - source_z
    r2 = dx*dx + dy*dy + dz*dz
    if r2 > 0 # AD safety check - don't do the sqrt until we're sure it's at a differentiable point.
        r = sqrt(r2)
        # Regularizing function and deriv
        g_sgm, dg_sgmdr = kernel_source.g_dgdr(r/sigma)
        # K × Γp
        crss1 = -const4 / r^3 * ( dy*gamma_z - dz*gamma_y )
        crss2 = -const4 / r^3 * ( dz*gamma_x - dx*gamma_z )
        crss3 = -const4 / r^3 * ( dx*gamma_y - dy*gamma_x )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        Ux = g_sgm * crss1
        Uy = g_sgm * crss2
        Uz = g_sgm * crss3
        U_target .+= Ux, Uy, Uz

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

        J .+= du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3
    end
    return r2

end

function fmm.direct!(target_system, target_index, source_system::ParticleField{R,F,V,S,Tkernel,TUJ,Tintegration}, source_index) where {R,F,V,S,Tkernel,TUJ,Tintegration}
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target,fmm.POSITION]
        velocity_gradient = reshape(target_system[j_target,fmm.VELOCITY_GRADIENT],9)
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = source_system.particles[i_source].Gamma
            source_x, source_y, source_z = source_system.particles[i_source].X
            sigma = source_system.particles[i_source].sigma[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            #if !iszero(r2)
            if r2 > 0
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
                target_system[j_target,fmm.VELOCITY] .+= Ux, Uy, Uz

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
                velocity_gradient .+= du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3
            end
        end
    end
end
