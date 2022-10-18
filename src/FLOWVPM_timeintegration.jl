#=##############################################################################
# DESCRIPTION
    Time integration schemes.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

"""
Steps the field forward in time by dt in a first-order Euler integration scheme.
"""
function euler(pfield::ParticleField{R, <:ClassicVPM, V},
                                dt::Real; relax::Bool=false) where {R, V}

    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sgs(pfield)
    pfield.sgsmodel(pfield)

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        scl::R = pfield.sgsscaling(p, pfield)

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Update vectorial circulation
        ## Vortex stretching contributions
        if pfield.transposed
            # Transposed scheme (Γ⋅∇')U
            p.Gamma[1] += dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
            p.Gamma[2] += dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
            p.Gamma[3] += dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        else
            # Classic scheme (Γ⋅∇)U
            p.Gamma[1] += dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
            p.Gamma[2] += dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
            p.Gamma[3] += dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        end

        ## Subgrid-scale contributions
        p.Gamma[1] += dt*scl*get_SGS1(p)*(p.sigma[1]^3/zeta0)
        p.Gamma[2] += dt*scl*get_SGS2(p)*(p.sigma[1]^3/zeta0)
        p.Gamma[3] += dt*scl*get_SGS3(p)*(p.sigma[1]^3/zeta0)


        # Relaxation: Align vectorial circulation to local vorticity
        if relax
            pfield.relaxation(pfield.rlxf, p)
        end

    end
    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end









"""
Steps the field forward in time by dt in a first-order Euler integration scheme
using the VPM reformulation. See notebook 20210104.
"""
function euler(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V},
                              dt::Real; relax::Bool=false ) where {R, V, R2}

    # Reset U and J to zero
    _reset_particles(pfield)

    # Calculate interactions between particles: U and J
    pfield.UJ(pfield)

    # Calculate subgrid-scale contributions
    _reset_particles_sgs(pfield)
    pfield.sgsmodel(pfield)

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Update the particle field: convection and stretching
    for p in iterator(pfield)

        scl::R = pfield.sgsscaling(p, pfield)

        # Update position
        p.X[1] += dt*(p.U[1] + Uinf[1])
        p.X[2] += dt*(p.U[2] + Uinf[2])
        p.X[3] += dt*(p.U[3] + Uinf[3])

        # Store stretching S under MM[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
            MM[2] = (p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
            MM[3] = (p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        else
            # Classic scheme S = (Γ⋅∇)U
            MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
            MM[2] = (p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
            MM[3] = (p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ + f/(1+3f) * M3⋅Γ ] / mag(Γ)^2, and M3=(M2+E)/zeta_sgmp(0)
        MM[4] = (f+g)/(1+3*f) * (MM[1]*p.Gamma[1] + MM[2]*p.Gamma[2] + MM[3]*p.Gamma[3])
        MM[4] += f/(1+3*f) * (scl*get_SGS1(p)*p.Gamma[1]
                                + scl*get_SGS2(p)*p.Gamma[2]
                                + scl*get_SGS3(p)*p.Gamma[3])*(p.sigma[1]^3/zeta0)
        MM[4] /= p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2

        # Update vectorial circulation ΔΓ = Δt*(S - 3ZΓ + M3)
        p.Gamma[1] += dt * (MM[1] - 3*MM[4]*p.Gamma[1] + scl*get_SGS1(p)*(p.sigma[1]^3/zeta0))
        p.Gamma[2] += dt * (MM[2] - 3*MM[4]*p.Gamma[2] + scl*get_SGS2(p)*(p.sigma[1]^3/zeta0))
        p.Gamma[3] += dt * (MM[3] - 3*MM[4]*p.Gamma[3] + scl*get_SGS3(p)*(p.sigma[1]^3/zeta0))

        # Update cross-sectional area of the tube σ = -Δt*σ*Z
        p.sigma[1] -= dt * ( p.sigma[1] * MM[4] )

        # Relaxation: Alig vectorial circulation to local vorticity
        if relax
            pfield.relaxation(pfield.rlxf, p)
        end

    end

    # Update the particle field: viscous diffusion
    viscousdiffusion(pfield, dt)

    return nothing
end












"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme. See Notebook entry 20180105.
"""
function rungekutta3(pfield::ParticleField{R, <:ClassicVPM, V},
                            dt::Real; relax::Bool=false) where {R, V}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    for p in iterator(pfield); p.M .= zero(R); end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Reset U and J from previous step
        _reset_particles(pfield)

        # Calculate interactions between particles: U and J
        pfield.UJ(pfield)

        # Calculate subgrid-scale contributions
        _reset_particles_sgs(pfield)
        pfield.sgsmodel(pfield)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            scl::R = pfield.sgsscaling(p, pfield)

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            # Update position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            ## Stretching + SGS contributions
            if pfield.transposed
                # Transposed scheme (Γ⋅∇')U
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3] + scl*get_SGS1(p)*(p.sigma[1]^3/zeta0))
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3] + scl*get_SGS2(p)*(p.sigma[1]^3/zeta0))
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3] + scl*get_SGS3(p)*(p.sigma[1]^3/zeta0))
            else
                # Classic scheme (Γ⋅∇)U
                p.M[1, 2] = a*p.M[1, 2] + dt*(p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3] + scl*get_SGS1(p)*(p.sigma[1]^3/zeta0))
                p.M[2, 2] = a*p.M[2, 2] + dt*(p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3] + scl*get_SGS2(p)*(p.sigma[1]^3/zeta0))
                p.M[3, 2] = a*p.M[3, 2] + dt*(p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3] + scl*get_SGS3(p)*(p.sigma[1]^3/zeta0))
            end

            # Update vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

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
            pfield.relaxation(pfield.rlxf, p)
        end
    end

    return nothing
end












"""
Steps the field forward in time by dt in a third-order low-storage Runge-Kutta
integration scheme using the VPM reformulation. See Notebook entry 20180105
(RK integration) and notebook 20210104 (reformulation).
"""
function rungekutta3(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V},
                     dt::Real; relax::Bool=false ) where {R, V, R2}

    # Storage terms: qU <=> p.M[:, 1], qstr <=> p.M[:, 2], qsmg2 <=> p.M[1, 3],
    #                      qsmg <=> p.M[2, 3], Z <=> MM[4], S <=> MM[1:3]

    # Calculate freestream
    Uinf::Array{<:Real, 1} = pfield.Uinf(pfield.t)

    MM::Array{<:Real, 1} = pfield.M
    f::R2, g::R2 = pfield.formulation.f, pfield.formulation.g
    zeta0::R = pfield.kernel.zeta(0)

    # Reset storage memory to zero
    for p in iterator(pfield); p.M .= zero(R); end;

    # Runge-Kutta inner steps
    for (a,b) in (R.((0, 1/3)), R.((-5/9, 15/16)), R.((-153/128, 8/15)))

        # Reset U and J from previous step
        _reset_particles(pfield)

        # Calculate interactions between particles: U and J
        pfield.UJ(pfield)

        # Calculate subgrid-scale contributions
        _reset_particles_sgs(pfield)
        pfield.sgsmodel(pfield)

        # Update the particle field: convection and stretching
        for p in iterator(pfield)

            scl::R = pfield.sgsscaling(p, pfield)

            # Low-storage RK step
            ## Velocity
            p.M[1, 1] = a*p.M[1, 1] + dt*(p.U[1] + Uinf[1])
            p.M[2, 1] = a*p.M[2, 1] + dt*(p.U[2] + Uinf[2])
            p.M[3, 1] = a*p.M[3, 1] + dt*(p.U[3] + Uinf[3])

            # Update position
            p.X[1] += b*p.M[1, 1]
            p.X[2] += b*p.M[2, 1]
            p.X[3] += b*p.M[3, 1]

            # Store stretching S under M[1:3]
            if pfield.transposed
                # Transposed scheme S = (Γ⋅∇')U
                MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[2,1]*p.Gamma[2]+p.J[3,1]*p.Gamma[3])
                MM[2] = (p.J[1,2]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[3,2]*p.Gamma[3])
                MM[3] = (p.J[1,3]*p.Gamma[1]+p.J[2,3]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            else
                # Classic scheme (Γ⋅∇)U
                MM[1] = (p.J[1,1]*p.Gamma[1]+p.J[1,2]*p.Gamma[2]+p.J[1,3]*p.Gamma[3])
                MM[2] = (p.J[2,1]*p.Gamma[1]+p.J[2,2]*p.Gamma[2]+p.J[2,3]*p.Gamma[3])
                MM[3] = (p.J[3,1]*p.Gamma[1]+p.J[3,2]*p.Gamma[2]+p.J[3,3]*p.Gamma[3])
            end

            # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ + f/(1+3f) * M3⋅Γ ] / mag(Γ)^2 and M3=(M2+E)/zeta_sgmp(0)
            MM[4] = (f+g)/(1+3*f) * (MM[1]*p.Gamma[1] + MM[2]*p.Gamma[2] + MM[3]*p.Gamma[3])
            MM[4] += f/(1+3*f) * (scl*get_SGS1(p)*p.Gamma[1]
                                    + scl*get_SGS2(p)*p.Gamma[2]
                                    + scl*get_SGS3(p)*p.Gamma[3])*(p.sigma[1]^3/zeta0)
            MM[4] /= p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2

            # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
            # with ΔΓ = Δt*( S - 3ZΓ + M3 )
            p.M[1, 2] = a*p.M[1, 2] + dt*(MM[1] - 3*MM[4]*p.Gamma[1] + scl*get_SGS1(p)*(p.sigma[1]^3/zeta0))
            p.M[2, 2] = a*p.M[2, 2] + dt*(MM[2] - 3*MM[4]*p.Gamma[2] + scl*get_SGS2(p)*(p.sigma[1]^3/zeta0))
            p.M[3, 2] = a*p.M[3, 2] + dt*(MM[3] - 3*MM[4]*p.Gamma[3] + scl*get_SGS3(p)*(p.sigma[1]^3/zeta0))

            # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
            p.M[2, 3] = a*p.M[2, 3] - dt*( p.sigma[1] * MM[4] )

            # Update vectorial circulation
            p.Gamma[1] += b*p.M[1, 2]
            p.Gamma[2] += b*p.M[2, 2]
            p.Gamma[3] += b*p.M[3, 2]

            # Update cross-sectional area
            p.sigma[1] += b*p.M[2, 3]

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
            pfield.relaxation(pfield.rlxf, p)
        end
    end

    return nothing
end






"""
    `relaxation_Pedrizzetti(rlxf::Real, p::Particle)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
"""
function relaxation_pedrizzetti(rlxf::Real, p::Particle)

    nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                    (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                    (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
    nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

    p.Gamma[1] = (1-rlxf)*p.Gamma[1] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.Gamma[2] = (1-rlxf)*p.Gamma[2] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.Gamma[3] = (1-rlxf)*p.Gamma[3] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

    return nothing
end


"""
    `relaxation_correctedPedrizzetti(rlxf::Real, p::Particle)`

Relaxation scheme where the vortex strength is aligned with the local vorticity.
This version fixes the error in Pedrizzetti's relaxation that made the strength
to continually decrease over time. See notebook 20200921 for derivation.
"""
function relaxation_correctedpedrizzetti(rlxf::Real, p::Particle)

    nrmw = sqrt( (p.J[3,2]-p.J[2,3])*(p.J[3,2]-p.J[2,3]) +
                    (p.J[1,3]-p.J[3,1])*(p.J[1,3]-p.J[3,1]) +
                    (p.J[2,1]-p.J[1,2])*(p.J[2,1]-p.J[1,2]))
    nrmGamma = sqrt(p.Gamma[1]^2 + p.Gamma[2]^2 + p.Gamma[3]^2)

    b2 =  1 - 2*(1-rlxf)*rlxf*(1 - (
                                    p.Gamma[1]*(p.J[3,2]-p.J[2,3]) +
                                    p.Gamma[2]*(p.J[1,3]-p.J[3,1]) +
                                    p.Gamma[3]*(p.J[2,1]-p.J[1,2])
                                   ) / (nrmGamma*nrmw))

    p.Gamma[1] = (1-rlxf)*p.Gamma[1] + rlxf*nrmGamma*(p.J[3,2]-p.J[2,3])/nrmw
    p.Gamma[2] = (1-rlxf)*p.Gamma[2] + rlxf*nrmGamma*(p.J[1,3]-p.J[3,1])/nrmw
    p.Gamma[3] = (1-rlxf)*p.Gamma[3] + rlxf*nrmGamma*(p.J[2,1]-p.J[1,2])/nrmw

    # Normalize the direction of the new vector to maintain the same strength
    p.Gamma ./= sqrt(b2)

    return nothing
end

################################################################################
################################################################################
################################################################################

# Eric Green's additions:

"""
This provides a function of the form f(dx,x,p) for interfacing with DifferentialEquations. Extra methods were defined for adding two ParticleFields
    and for multiplying a number by a ParticleField (i.e. n*ParticleField). It should be (mostly) in-place.
"""

# function DiffEQ_derivative_function!(dpfield::ParticleField{R,F,V}, pfield::ParticleField{R,F,V}, param,t) where {R,F,V}
function DiffEQ_derivative_function!(dpfield,pfield,settings,t)
    ## The format for computations was taken from Euler step implementation.
    #println(get_np(settings))
    np = Int(get_np(settings))
    #np = 180
    dpfield .= zero(eltype(dpfield))

    # No need to run any more computations if there are no active particles. This might also prevent weird behavior in the UJ function if no particles are active.
    if np < 1
        #println("No active particles!")
        return nothing
    end

    # Gets the verbosity level for console output.
    verbose = get_verbose(settings)
    # Checks if the output is transposed.
    transposed = get_transposed(settings)
    # Gets the viscous formulation
    viscous = get_viscous(settings)
    #nu = get_nu(viscous)
    nu = viscous.nu
    # Gets the particle interaction function. The function signature should be f(d_sources,sources,targets,kernel,settings).
    UJ_function! = get_UJ(settings)

    kernel = get_kernel(settings)
    # Call the UJ function. Edits dpfield in-place.
    UJ_function!(dpfield,pfield,pfield,kernel,settings) # 20% of allocations occur here (and practically all of the ones for this ODE function)... but only in the reverse pass
    #UJ_direct_3!(dpfield,pfield,pfield,gaussianerf,settings)
    # Calculate subgrid-scale contributions - currently disabled; will need to be updated for compatibility with current calculation approaches
    #_reset_particles_sgs(pfield)
    # pfield.sgsmodel(pfield)

    # Calculate freestream
    Uinf = get_Uinf(settings)(t) # evaluates Uinf here to only call it once per iteration.
    # Currently disabled sgs calculations.
    #zeta0::R = pfield.kernel.zeta(0)
    #zeta0 = pfield.kernel.zeta(0)
    ##
    # The range from here to the end accounts for only 0.25% of the (total) allocations on the reverse pass. However, it accounts for most of the allocations in this function on the forward pass.
    # Update the particle field: convection and stretching
    for i=1:np
        i0 = (i-1)*size(Particle) # stores 1 less than the index associated with the beginning of particle i.
        # Currently disabled sgs stuff
        #scl = pfield.sgsscaling(p, pfield)

        # Update position:
        # dx .= U .+ Uinf
        dpfield[i0+1] = dpfield[i0+10] + Uinf[1]
        dpfield[i0+2] = dpfield[i0+11] + Uinf[2]
        dpfield[i0+3] = dpfield[i0+12] + Uinf[3]

        # Update vectorial circulation
        ## Vortex stretching contributions
        if transposed
            # Transposed scheme (Γ⋅∇')U
            dpfield[i0+4] = dpfield[i0+13]*pfield[i0+4] + dpfield[i0+14]*pfield[i0+5] + dpfield[i0+15]*pfield[i0+6]
            dpfield[i0+5] = dpfield[i0+16]*pfield[i0+4] + dpfield[i0+17]*pfield[i0+5] + dpfield[i0+18]*pfield[i0+6]
            dpfield[i0+6] = dpfield[i0+19]*pfield[i0+4] + dpfield[i0+20]*pfield[i0+5] + dpfield[i0+21]*pfield[i0+6]
        else
            # Classic scheme (Γ⋅∇)U
            dpfield[i0+4] = dpfield[i0+13]*pfield[i0+4] + dpfield[i0+16]*pfield[i0+5] + dpfield[i0+19]*pfield[i0+6]
            dpfield[i0+5] = dpfield[i0+14]*pfield[i0+4] + dpfield[i0+17]*pfield[i0+5] + dpfield[i0+20]*pfield[i0+6]
            dpfield[i0+6] = dpfield[i0+15]*pfield[i0+4] + dpfield[i0+18]*pfield[i0+5] + dpfield[i0+21]*pfield[i0+6]
        end
    
        # Currently disabled LES calculations:
        #dp[4] += scl*get_SGS1(p)*(p.sigma[1]^3/zeta0)
        #dp[5] += scl*get_SGS2(p)*(p.sigma[1]^3/zeta0)
        #dp[6] += scl*get_SGS3(p)*(p.sigma[1]^3/zeta0)

        # Since the differential part of the core spreading is only three lines, I moved it to the main derivative calculation function.
        # Core size resets now occur through callbacks. ## core size reset currently disabled; I'll probably want to look into those details at some point.
        if iscorespreadingmodified(viscous)
            # This should be nonzero, but if it isn't it will fill everything with NaN values. The error might be turned into a warning if solvers cause states full of zeros to be passed in.
            if pfield[i0+7] > 0 
                # This line could use some theoretical checks
                dpfield[i0+7] = nu/pfield[i0+7]
            else
                #error("sigma is $(pfield[i0+7]) for particle $(i) at time $(t)!")
            end
        elseif isinviscid(viscous)
            nothing
        else
            error("viscous scheme not identified")
        end

        #if sum(isnan.(dpfield[i0+1:i0+7])) > 0
        #    error("NaN value at particle $(i) at time $(t)!\t Particle data: $(pfield[i0+1:i0+7])\tDerivative data: $(dpfield[i0+1:i0+size(Particle)])")
        #end
    end
    if 0 < verbose < 15
        if typeof(t) <: Number
            println(t)
        else
            println(t.value)
        end
    elseif verbose  >= 15
        println(t)
    end
    if verbose >= 10
        println(np)
    end
end
# TODO (mostly for this file):
# Code clean up:
#    clean up commented out stuff - delete or put in verbosity blocks # done
# Interface update:
#    Verbosity: pass in, check it, etc. It should be a hidden variable associated with the settings struct.
#    UJ function: pass in through settings struct, again as a hidden variable.
#    Uinf function: make sure its passed in through the settings struct correctly. This might already be done.
# SGS/LES:
#    currently commented out because the sgs models need to be rewritten
# Make sure the reformulated VPM is being used rather than the non-reformulated one. I think this is the case but I need to double check.
# Compatibility with Ryan's surface interactions: add a function to add to the velocity at a particle. This will probably be a lot
#    like the Uinf function but might need some other parameters like particle number or location
