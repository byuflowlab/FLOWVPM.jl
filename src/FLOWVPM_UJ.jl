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
function UJ_fmm(pfield::ParticleField; optargs...)

    # Calculate FMM of vector potential
    call_FLOWExaFMM(pfield; optargs...)

    aux1 = RealFMM(1/(4*pi))

    # Build velocity and velocity Jacobian from the FMM's vector potential
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

function call_FLOWExaFMM(pfield::ParticleField; verbose::Bool=false,
                            rbf::Bool=false, sgs::Bool=false, sgs_type::Int=-1,
                            transposed_sgs::Bool=true,
                            reset::Bool=true, reset_sgs::Bool=false,
                            sort::Bool=true)
    try
        fmm.calculate(pfield.bodies,
                        Int32(get_np(pfield)),
                        Int32(pfield.fmm.p), Int32(pfield.fmm.ncrit),
                        RealFMM(pfield.fmm.theta), RealFMM(pfield.fmm.phi), verbose,
                        Int32(pfield.kernel.EXAFMM_P2P),
                        Int32(pfield.kernel.EXAFMM_L2P),
                        Int32(sgs_type),
                        rbf, sgs, transposed_sgs,
                        reset, reset_sgs, sort)
    catch e
        error("ExaFMM unexpected error: $(e)")
    end
end

####################################################################################
# Eric Green's additions, pt. 1:
# Direct UJ calculations for vector inputs

# Inputs: Vector of source data, vector of target data, temp memory for targets, particle count, particle state size,
# temp state size, and kernel function
function UJ_direct_vectorinputs!(sources, targets, targets_temp_mem, np, psz, tsz, g_dgdr)

  for i=1:np
    for j=1:np
  #for Pi in targets
  #  for Pj in sources

      Pi = @view targets[psz*(i-1)+1:psz*(i-1)+9]
      Pj = @view sources[psz*(j-1)+1:psz*(j-1)+9]
      Pi_temp = @view targets_temp_mem[tsz*(i-1)+1:tsz*(i-1)+tsz]

      #dX1 = Pi.X[1] - Pj.X[1]
      #dX2 = Pi.X[2] - Pj.X[2]
      #dX3 = Pi.X[3] - Pj.X[3]
      dX1 = Pi[1] - Pj[1]
      dX2 = Pi[2] - Pj[2]
      dX3 = Pi[3] - Pj[3]
      r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

      #if r!=0
      if i != j
        # Regularizing function and deriv
        #g_sgm, dg_sgmdr = g_dgdr(r/Pj.sigma[1])
        g_sgm, dg_sgmdr = g_dgdr(r/Pj[7])

        # K × Γp
        #crss1 = -const4 / r^3 * ( dX2*Pj.Gamma[3] - dX3*Pj.Gamma[2] )
        #crss2 = -const4 / r^3 * ( dX3*Pj.Gamma[1] - dX1*Pj.Gamma[3] )
        #crss3 = -const4 / r^3 * ( dX1*Pj.Gamma[2] - dX2*Pj.Gamma[1] )
        crss1 = -const4 / r^3 * ( dX2*Pj[6] - dX3*Pj[5] )
        crss2 = -const4 / r^3 * ( dX3*Pj[4] - dX1*Pj[6] )
        crss3 = -const4 / r^3 * ( dX1*Pj[5] - dX2*Pj[4] )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        #Pi.U[1] += g_sgm * crss1
        #Pi.U[2] += g_sgm * crss2
        #Pi.U[3] += g_sgm * crss3

        Pi_temp[1] += g_sgm * crss1
        Pi_temp[2] += g_sgm * crss2
        Pi_temp[3] += g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        #aux = dg_sgmdr/(Pj.sigma[1]*r) - 3*g_sgm /r^2
        aux = dg_sgmdr/(Pj[7]*r) - 3*g_sgm /r^2
        # j=1
        #Pi.J[1, 1] += aux * crss1 * dX1
        #Pi.J[2, 1] += aux * crss2 * dX1
        #Pi.J[3, 1] += aux * crss3 * dX1
        # j=2
        #Pi.J[1, 2] += aux * crss1 * dX2
        #Pi.J[2, 2] += aux * crss2 * dX2
        #Pi.J[3, 2] += aux * crss3 * dX2
        # j=3
        #Pi.J[1, 3] += aux * crss1 * dX3
        #Pi.J[2, 3] += aux * crss2 * dX3
        #Pi.J[3, 3] += aux * crss3 * dX3

        # j=1
        Pi_temp[4] += aux * crss1 * dX1
        Pi_temp[5] += aux * crss2 * dX1
        Pi_temp[6] += aux * crss3 * dX1
        # j=2
        Pi_temp[7] += aux * crss1 * dX2
        Pi_temp[8] += aux * crss2 * dX2
        Pi_temp[9] += aux * crss3 * dX2
        # j=3
        Pi_temp[10] += aux * crss1 * dX3
        Pi_temp[11] += aux * crss2 * dX3
        Pi_temp[12] += aux * crss3 * dX3

        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux = - const4 * g_sgm / r^3

        # j=1
        #Pi.J[2, 1] -= aux * Pj.Gamma[3]
        #Pi.J[3, 1] += aux * Pj.Gamma[2]
        # j=2
        #Pi.J[1, 2] += aux * Pj.Gamma[3]
        #Pi.J[3, 2] -= aux * Pj.Gamma[1]
        # j=3
        #Pi.J[1, 3] -= aux * Pj.Gamma[2]
        #Pi.J[2, 3] += aux * Pj.Gamma[1]

        # j=1
        Pi_temp[5] -= aux * Pj[6]
        Pi_temp[6] += aux * Pj[5]
        # j=2
        Pi_temp[7] += aux * Pj[6]
        Pi_temp[9] -= aux * Pj[4]
        # j=3
        Pi_temp[10] -= aux * Pj[5]
        Pi_temp[11] += aux * Pj[4]

      end
    end
  end

 return nothing
end

function UJ_direct_2(sources, targets, g_dgdr::Function)

  #for Pi in targets
  #  for Pj in sources

  np = 180

  for i=1:np
    for j=1:np

      if (typeof(sources[1]) <: Particle)
        Pi = targets[i]
        Pj = sources[j]
      else
        Pi = @view targets[(i-1)*size(Particle)+1:(i-1)*size(Particle)+size(Particle)]
        Pj = @view sources[(j-1)*size(Particle)+1:(j-1)*size(Particle)+size(Particle)]
      end

      dX1 = Pi[1] - Pj[1]
      dX2 = Pi[2] - Pj[2]
      dX3 = Pi[3] - Pj[3]
      r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

      if r!=0

        # Regularizing function and deriv
        #g_sgm, dg_sgmdr = g_dgdr(r/Pj.sigma[1])
        g_sgm, dg_sgmdr = g_dgdr(r/Pj[7])

        # K × Γp
        crss1 = -const4 / r^3 * ( dX2*Pj[6] - dX3*Pj[5] )
        crss2 = -const4 / r^3 * ( dX3*Pj[4] - dX1*Pj[6] )
        crss3 = -const4 / r^3 * ( dX1*Pj[5] - dX2*Pj[4] )

        # U = ∑g_σ(x-xp) * K(x-xp) × Γp
        #Pi.U[1] += g_sgm * crss1
        #Pi.U[2] += g_sgm * crss2
        #Pi.U[3] += g_sgm * crss3
        Pi[10] += g_sgm * crss1
        Pi[11] += g_sgm * crss2
        Pi[12] += g_sgm * crss3

        # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
        # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
        #aux = dg_sgmdr/(Pj.sigma[1]*r) - 3*g_sgm /r^2
        aux = dg_sgmdr/(Pj[7]*r) - 3*g_sgm /r^2
        # j=1
        Pi[13] += aux * crss1 * dX1
        Pi[14] += aux * crss2 * dX1
        Pi[15] += aux * crss3 * dX1
        # j=2
        Pi[16] += aux * crss1 * dX2
        Pi[17] += aux * crss2 * dX2
        Pi[18] += aux * crss3 * dX2
        # j=3
        Pi[19] += aux * crss1 * dX3
        Pi[20] += aux * crss2 * dX3
        Pi[21] += aux * crss3 * dX3

        # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
        # Adds the Kronecker delta term
        aux = - const4 * g_sgm / r^3

        # j=1
        Pi[14] -= aux * Pj[6]
        Pi[15] += aux * Pj[5]
        # j=2
        Pi[16] += aux * Pj[6]
        Pi[18] -= aux * Pj[4]
        # j=3
        Pi[19] -= aux * Pj[5]
        Pi[20] += aux * Pj[4]

      end
    end
  end

 return nothing
end

function UJ_direct_3!(d_targets, sources, targets, g_dgdr::Function, settings)

  #active_np_i = Int(get_np(settings))
  #active_np_j = active_np_i
  const4 = 1/(4*pi) # defining this here avoids a segfault. For some reason ReverseDiff does not like global variables in other files, so if const4 is pulled from FLOWVPM.jl the sensitivity analysis breaks.
  i0 = 0
  j0 = 0
  dX1 = zero(targets[1])
  dX2 = zero(targets[1])
  dX3 = zero(targets[1])
  rs = zero(targets[1])
  r = zero(targets[1])
  g_sgm = zero(targets[1])
  dg_sgmdr = zero(targets[1])
  aux = zero(targets[1])
  np = Int(get_np(settings))
  #np = 180

  for i=1:np
    for j=1:np
        # i -> targets
        # j -> sources

        # indices for the correct starting location for each particle
        i0 = (i-1)*length(Particle)
        j0 = (j-1)*length(Particle)

        #if sum(isnan.(targets[i0+1:i0+7])) > 0 || sum(isnan.(sources[i0+1:i0+7])) > 0
        #  error("")
        #end
        #if targets[i0+7] <= 0 || sources[j0+7] <= 0
          #println("$i, $j")
          #error("$(targets[i0+7]), $(sources[j0+7])")
        #end

        dX1 = targets[i0+1] - sources[j0+1]
        dX2 = targets[i0+2] - sources[j0+2]
        dX3 = targets[i0+3] - sources[j0+3]
        rs = dX1*dX1 + dX2*dX2 + dX3*dX3 # taking the square root causes issues because the derivative of square root at 0 is undefined.
        if rs!=zero(rs)
          r = sqrt(rs)
          # Regularizing function and deriv
          g_sgm, dg_sgmdr = g_dgdr(r/sources[j0+7]) # some performance stuff to figure out later - this line slows things down a lot.
          # K × Γp          
          crss1 = -const4 / r^3 * ( dX2*sources[j0+6] - dX3*sources[j0+5] )
          crss2 = -const4 / r^3 * ( dX3*sources[j0+4] - dX1*sources[j0+6] )
          crss3 = -const4 / r^3 * ( dX1*sources[j0+5] - dX2*sources[j0+4] )
          
          # U = ∑g_σ(x-xp) * K(x-xp) × Γp
          d_targets[i0+10] += g_sgm * crss1
          d_targets[i0+11] += g_sgm * crss2
          d_targets[i0+12] += g_sgm * crss3

          # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
          # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
          aux = dg_sgmdr/(sources[j0+7]*r) - 3*g_sgm /r^2
          # j=1
          d_targets[i0+13] += aux * crss1 * dX1
          d_targets[i0+14] += aux * crss2 * dX1
          d_targets[i0+15] += aux * crss3 * dX1
          # j=2
          d_targets[i0+16] += aux * crss1 * dX2
          d_targets[i0+17] += aux * crss2 * dX2
          d_targets[i0+18] += aux * crss3 * dX2
          # j=3
          d_targets[i0+19] += aux * crss1 * dX3
          d_targets[i0+20] += aux * crss2 * dX3
          d_targets[i0+21] += aux * crss3 * dX3

          # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
          # Adds the Kronecker delta term
          aux = - const4 * g_sgm / r^3

          # j=1
          d_targets[i0+14] -= aux * sources[j0+6]
          d_targets[i0+15] += aux * sources[j0+5]
          # j=2
          d_targets[i0+16] += aux * sources[j0+6]
          d_targets[i0+18] -= aux * sources[j0+4]
          # j=3
          d_targets[i0+19] -= aux * sources[j0+5]
          d_targets[i0+20] += aux * sources[j0+4]
          
        end
    end
  end

  return nothing

end

function UJ_direct_2(sources, targets, kernel::Kernel)
  return UJ_direct_2(sources, targets, kernel.g_dgdr)
end

function UJ_direct_3!(d_targets, sources, targets, kernel::Kernel, settings)
  return UJ_direct_3!(d_targets, sources, targets, kernel.g_dgdr, settings)
end

#TODO:
# clean up UJ_direct_3! # done
#    remove dev code output # done