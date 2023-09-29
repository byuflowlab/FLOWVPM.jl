#=##############################################################################
# DESCRIPTION
    Fast-multipole parameters.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
=###############################################################################

################################################################################
# FMM STRUCT
################################################################################
"""
    `FMM(; p::Int=4, ncrit::Int=50, theta::Real=0.4, phi::Real=0.3)`

Parameters for FMM solver.

# Arguments
* `p`       : Order of multipole expansion (number of terms).
* `ncrit`   : Maximum number of particles per leaf.
* `theta`   : Neighborhood criterion. This criterion defines the distance
                where the far field starts. The criterion is that if θ*r < R1+R2
                the interaction between two cells is resolved through P2P, where
                r is the distance between cell centers, and R1 and R2 are each
                cell radius. This means that at θ=1, P2P is done only on cells
                that have overlap; at θ=0.5, P2P is done on cells that their
                distance is less than double R1+R2; at θ=0.25, P2P is done on
                cells that their distance is less than four times R1+R2; at
                θ=0, P2P is done on cells all cells.
* `phi`     : Regularizing neighborhood criterion. This criterion avoid
                approximating interactions with the singular-FMM between
                regularized particles that are sufficiently close to each other
                across cell boundaries. Used together with the θ-criterion, P2P
                is performed between two cells if φ < σ/dx, where σ is the
                average smoothing radius in between all particles in both cells,
                and dx is the distance between cell boundaries
                ( dx = r-(R1+R2) ). This means that at φ = 1, P2P is done on
                cells with boundaries closer than the average smoothing radius;
                at φ = 0.5, P2P is done on cells closer than two times the
                smoothing radius; at φ = 0.25, P2P is done on cells closer than
                four times the smoothing radius.
"""
mutable struct FMM
  # Optional user inputs
  p::Int32                        # Multipole expansion order
  ncrit::Int32                    # Max number of particles per leaf
  theta::RealFMM                  # Neighborhood criterion
  phi::RealFMM                    # Regularizing neighborhood criterion

  FMM(; p=4, ncrit=50, theta=0.4, phi=1/3) = new(p, ncrit, theta, phi)
end

#####
##### compatibility functions for use with FLOWFMM
#####
const zero_vec = SVector{3}(0.0,0.0,0.0)
Base.getindex(particle_field::ParticleField, i, ::fmm.Position) = particle_field.particles[i].X
Base.getindex(particle_field::ParticleField, i, ::fmm.Radius) = particle_field.particles[i].sigma[1]
Base.getindex(particle_field::ParticleField, i, ::fmm.VectorPotential) = zero_vec
Base.getindex(particle_field::ParticleField, i, ::fmm.ScalarPotential) = 0.0
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
Base.eltype(::ParticleField{TF}) where TF = TF

fmm.B2M!(system::ParticleField, args...) = fmm.B2M!_vortexpoint(system, args...)

function fmm.direct!(target_system, target_index, source_system::ParticleField, source_index)
    for j_target in target_index
        target_x, target_y, target_z = target_system[j_target,fmm.POSITION]
        for i_source in source_index
            gamma_x, gamma_y, gamma_z = source_system.particles[i_source].Gamma
            source_x, source_y, source_z = source_system.particles[i_source].X
            sigma = source_system.particles[i_source].sigma[1]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            if !iszero(r2)
                r = sqrt(r2)
                # Regularizing function and deriv
                g_sgm, dg_sgmdr = source_system.g_dgdr(r/Pj.sigma[1])

                # K × Γp
                crss1 = -const4 / r^3 * ( dX2*Pj.Gamma[3] - dX3*Pj.Gamma[2] )
                crss2 = -const4 / r^3 * ( dX3*Pj.Gamma[1] - dX1*Pj.Gamma[3] )
                crss3 = -const4 / r^3 * ( dX1*Pj.Gamma[2] - dX2*Pj.Gamma[1] )

                # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                Ux += g_sgm * crss1
                Uy += g_sgm * crss2
                Uz += g_sgm * crss3
                target_system[j_target,fmm.VELOCITY] .+= Ux, Uy, Uz

                # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                aux = dg_sgmdr/(sigma*r) - 3*g_sgm /r^2
                # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                # Adds the Kronecker delta term
                aux2 = -const4 * g_sgm / r^3
                # j=1
                du1x1 = aux * crss1 * dX1
                du2x1 = aux * crss2 * dX1 - aux2 * gamma_z
                du3x1 = aux * crss3 * dX1 + aux2 * gamma_y
                # j=2
                du1x2 = aux * crss1 * dX2 + aux2 * gamma_z
                du2x2 = aux * crss2 * dX2
                du3x2 = aux * crss3 * dX2 - aux2 * gamma_x
                # j=3
                du1x3 = aux * crss1 * dX3 - aux2 * gamma_y
                du2x3 = aux * crss2 * dX3 + aux2 * gamma_x
                du3x3 = aux * crss3 * dX3
                target_system[j_target,fmm.VELOCITY_GRADIENT] .+= du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3
            end
        end
    end
end
