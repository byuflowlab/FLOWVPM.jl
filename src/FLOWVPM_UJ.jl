#=##############################################################################
# DESCRIPTION
    Particle-to-particle interactions calculation.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################


"""
  `UJ_direct(pfield)`

Calculates the velocity and Jacobian that the field exerts on itself by direct
particle-to-particle interaction, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_direct(pfield::ParticleField)
  return UJ_direct(pfield, pfield)
end

"""
  `UJ_direct(source, target)`

Calculates the velocity and Jacobian that the field `source` exerts on every
particle of  field `target`, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_direct(source::ParticleField, target::ParticleField)
  return UJ_direct(iterator(source), iterator(target), source.kernel)
end

function UJ_direct(sources, targets, kernel::Kernel)
 return UJ_direct(sources, targets, kernel.g_dgdr)
end

function UJ_direct(sources, targets, g_dgdr::Function)

   for Pi in targets
     for Pj in sources

       dX1 = Pi.X[1] - Pj.X[1]
       dX2 = Pi.X[2] - Pj.X[2]
       dX3 = Pi.X[3] - Pj.X[3]
       r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

       if r!=0

         # Regularizing function and deriv
         g_sgm, dg_sgmdr = g_dgdr(r/Pj.sigma[1])

         # K × Γp
         crss1 = -const4 / r^3 * ( dX2*Pj.Gamma[3] - dX3*Pj.Gamma[2] )
         crss2 = -const4 / r^3 * ( dX3*Pj.Gamma[1] - dX1*Pj.Gamma[3] )
         crss3 = -const4 / r^3 * ( dX1*Pj.Gamma[2] - dX2*Pj.Gamma[1] )

         # U = ∑g_σ(x-xp) * K(x-xp) × Γp
         Pi.U[1] += g_sgm * crss1
         Pi.U[2] += g_sgm * crss2
         Pi.U[3] += g_sgm * crss3

         # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
         # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
         aux = dg_sgmdr/(Pj.sigma[1]*r) - 3*g_sgm /r^2
         # j=1
         Pi.J[1, 1] += aux * crss1 * dX1
         Pi.J[2, 1] += aux * crss2 * dX1
         Pi.J[3, 1] += aux * crss3 * dX1
         # j=2
         Pi.J[1, 2] += aux * crss1 * dX2
         Pi.J[2, 2] += aux * crss2 * dX2
         Pi.J[3, 2] += aux * crss3 * dX2
         # j=3
         Pi.J[1, 3] += aux * crss1 * dX3
         Pi.J[2, 3] += aux * crss2 * dX3
         Pi.J[3, 3] += aux * crss3 * dX3

         # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
         # Adds the Kronecker delta term
         aux = - const4 * g_sgm / r^3

         # j=1
         Pi.J[2, 1] -= aux * Pj.Gamma[3]
         Pi.J[3, 1] += aux * Pj.Gamma[2]
         # j=2
         Pi.J[1, 2] += aux * Pj.Gamma[3]
         Pi.J[3, 2] -= aux * Pj.Gamma[1]
         # j=3
         Pi.J[1, 3] -= aux * Pj.Gamma[2]
         Pi.J[2, 3] += aux * Pj.Gamma[1]

       end
     end
   end

  return nothing
end


"""
  `UJ_fmm(pfield)`

Calculates the velocity and Jacobian that the field exerts on itself through
a fast-multipole approximation, saving U and J on the particles.

NOTE: This method accumulates the calculation on the properties U and J of
every particle without previously emptying those properties.
"""
function UJ_fmm(pfield::ParticleField; verbose::Bool=false, rbf::Bool=false)

    # Calculate FMM of vector potential
    fmm.calculate(pfield.bodies,
                    Int32(get_np(pfield)),
                    Int32(pfield.fmm.p), Int32(pfield.fmm.ncrit),
                    RealFMM(pfield.fmm.theta), RealFMM(pfield.fmm.phi), verbose,
                    Int32(pfield.kernel.EXAFMM_P2P),
                    Int32(pfield.kernel.EXAFMM_L2P), rbf)

    # Sort particles according to index
    # sort!(iterator(pfield); by=P->P.index[1])

    aux1 = RealFMM(1/(4*pi))

    for P in iterator(pfield)
        # Velocity U = ∇ × ψ
        P.U[1] += aux1*(P.Jexa[2,3] - P.Jexa[3,2])
        P.U[2] += aux1*(P.Jexa[3,1] - P.Jexa[1,3])
        P.U[3] += aux1*(P.Jexa[1,2] - P.Jexa[2,1])

        # Jacobian
        # dU1 / dxj
        P.J[1, 1] += aux1*(P.dJdx1exa[2,3] - P.dJdx1exa[3,2])
        P.J[1, 2] += aux1*(P.dJdx2exa[2,3] - P.dJdx2exa[3,2])
        P.J[1, 3] += aux1*(P.dJdx3exa[2,3] - P.dJdx3exa[3,2])
        # dU2 / dxj
        P.J[2, 1] += aux1*(P.dJdx1exa[3,1] - P.dJdx1exa[1,3])
        P.J[2, 2] += aux1*(P.dJdx2exa[3,1] - P.dJdx2exa[1,3])
        P.J[2, 3] += aux1*(P.dJdx3exa[3,1] - P.dJdx3exa[1,3])
        # dU3 / dxj
        P.J[3, 1] += aux1*(P.dJdx1exa[1,2] - P.dJdx1exa[2,1])
        P.J[3, 2] += aux1*(P.dJdx2exa[1,2] - P.dJdx2exa[2,1])
        P.J[3, 3] += aux1*(P.dJdx3exa[1,2] - P.dJdx3exa[2,1])
    end

    return nothing
end
