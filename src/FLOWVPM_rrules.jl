

# functions to provide ChainRules pullbacks for:

# direct!
# vorticity_direct!

# probably euler and rungekutta3

# run_vpm! might be covered by ImplicitAD, but we'll see what's easier to implement. 

# idea: use CatViews if ReverseDiff is used so that the vectors in ParticleFields can be treated like one big array

using ChainRulesCore
using ReverseDiff
using CatViews

# Levi-Civita tensor contractions for convenience. This shows up in cross products.
# ϵ with two vectors -> vector, so need one scalar index
# ϵ with one vector -> matrix, so need two scalar indices
# ϵ with one matrix -> vector, so need one scalar index
ϵ(a,x::Vector,y::Vector) = (a == 1) ? (x[2]*y[3] - x[3]*y[2]) : ((a == 2) ? (x[3]*y[1] - x[1]*y[3]) : ((a == 3) ? (x[1]*y[2] - x[2]*y[1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")))
ϵ(a,b::Number,y::Vector) = (a == b) ? zero(eltype(y)) : ((mod(b-a,3) == 1) ? y[mod(b,3)+1] : ((mod(a-b,3) == 1) ? -y[mod(b-2,3)+1] : error("attempted to evaluate Levi-Civita symbol at out-of-bounds indices $(a) and $(b)!")))
ϵ(a,x::Vector,c::Number) = -1 .*ϵ(a,c,x)
ϵ(a,x::TM) where {TM <: AbstractArray} = (a == 1) ? (x[2,3] - x[3,2]) : (a == 2) ? (x[3,1]-x[1,3]) : (a == 3) ? (x[1,2]-x[2,1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")
ϵ(a,b::Number, c::Number) = (a == b || b == c || c == a) ? 0 : (mod(b-a,3) == 1 ? 1 : -1) # no error checks in this implementation, since that would significantly increase the cost of it

function fmm.direct!(target_system::ParticleField{R,F,V,S,Tkernel,TUJ,Tintegration}, target_index, source_system::ParticleField{R,F,V,S,Tkernel,TUJ,Tintegration}, source_index) where {R<:ReverseDiff.TrackedReal,F,V,S,Tkernel,TUJ,Tintegration}
    # need: target xyz vectors, target J matrices, source gamma vectors, source xyz vectors, source sigma vectors, source kernel, target U vectors
    # also, for the SFS self-interactions, I need target J matrices, source J matrices, source gamma vectors, source sigma vectors, and target S vectors.

    #l = length(ReverseDiff.tape(target_system.particles[1].X[1]))
    xyz_target = cat(map(i->target_system.particles[i].X, target_index)...;dims=1)
    J_target = cat(map(i->reshape(target_system.particles[i].J,9), target_index)...;dims=1)
    gamma_source = cat(map(i->source_system.particles[i].Gamma, source_index)...;dims=1)
    xyz_source = cat(map(i->source_system.particles[i].X, source_index)...;dims=1)
    sigma_source = cat(map(i->source_system.particles[i].sigma, source_index)...;dims=1)
    kernel_source = source_system.kernel
    U_target = cat(map(i->target_system.particles[i].U, target_index)...;dims=1)

    J_source = cat(map(i->reshape(source_system.particles[i].J,9), source_index)...;dims=1)
    S_target = cat(map(i->target_system.particles[i].S, target_index)...;dims=1)

    if source_system.toggle_rbf
        error("vorticity_direct not yet compatible with reversediff! Please set toggle_rbf to false.")
    end

    #UJS = CatView(U_target, J_target, S_target)
    #UJS = [U_target...,J_target...,S_target...]
    #UJS = zeros(length(U_target) + length(J_target) + length(S_target))
    #UJS = cat(U_target..., J_target..., S_target...;dims=1)

    # problem: the inputs are currently vectors of vectors rather than plain vectors. So either I need to change the input types or I need to change the input to the macro. Changing macro inputs to match these types is probably the first thing to try.
    #@show typeof(xyz_target) typeof(J_target) typeof(gamma_source) typeof(xyz_source) typeof(sigma_source) typeof(kernel_source) typeof(U_target) typeof(J_source) typeof(S_target)
    #println("preprocessing tape entries: $(length(ReverseDiff.tape(target_system.particles[1].X[1])) - l)")
    #l = length(ReverseDiff.tape(target_system.particles[1].X[1]))
    UJS = fmm.direct!(xyz_target, J_target, gamma_source, xyz_source, sigma_source, kernel_source, U_target, J_source, S_target, length(target_index), length(source_index),source_system.toggle_sfs)
    #println("direct! tape entries: $(length(ReverseDiff.tape(target_system.particles[1].X[1])) - l)")
    #l = length(ReverseDiff.tape(target_system.particles[1].X[1]))
    #@show size(UJS) size(J_target) size(U_target)
    for i=1:length(target_index)
        target_system.particles[target_index[i]].U .= UJS[3*(i-1)+1:3*(i-1)+3]
        target_system.particles[target_index[i]].J .= reshape(UJS[3*length(target_index) + 9*(i-1)+1:3*length(target_index) + 9*(i-1)+9],(3,3))
        target_system.particles[target_index[i]].S .= UJS[12*length(target_index) + 3*(i-1)+1:12*length(target_index) + 3*(i-1)+3]
    end
    #println("postprocessing tape entries: $(length(ReverseDiff.tape(target_system.particles[1].X[1])) - l)")
    #l = length(ReverseDiff.tape(target_system.particles[1].X[1]))
    #U_target .= UJS[1:3*length(target_index)]
    #J_target .= UJS[3*length(target_index)+1:12*length(target_index)]
    #S_target .= UJS[12*length(target_index)+1:15*length(target_index)]

end

function fmm.direct!(xyz_target, J_target, gamma_source, xyz_source, sigma_source, kernel_source, U_target, J_source, S_target,target_index_count,source_index_count,toggle_sfs)

    r = zero(eltype(xyz_target[1]))
    for ti = 1:target_index_count
        tidx = 3*(ti-1)
        target_x, target_y, target_z = view(xyz_target,tidx+1:tidx+3)
        J_target_mat = reshape(view(J_target,9*(ti-1)+1:9*(ti-1)+9),(3,3))
        for si = 1:source_index_count
            sidx = 3*(si-1)
            gamma_x, gamma_y, gamma_z = view(gamma_source,sidx+1:sidx+3)
            source_x, source_y, source_z = view(xyz_source,sidx+1:sidx+3)
            sigma = sigma_source[si]
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            #if !iszero(r2)
            if r2 > 0
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
                view(U_target,tidx+1:tidx+3) .+= Ux, Uy, Uz

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

                J_target_mat[1:3,1] .+= du1x1, du2x1, du3x1
                J_target_mat[1:3,2] .+= du1x2, du2x2, du3x2
                J_target_mat[1:3,3] .+= du1x3, du2x3, du3x3
            end

            # include self-induced contribution to SFS
            if toggle_sfs && r2 > 0
                @show toggle_sfs
                error("SFS pullback not implemented yet!")
                #source_system.SFS.model(target_particle::Particle, source_particle::Particle, r2 > 0 ? sqrt(r2) : 0.0, source_system.kernel.zeta, source_system.transposed)
                # Transposed scheme (Γq⋅∇')(Up - Uq)
                S1 = (J_target_mat[1,1] - J_source[si][1,1])*gamma_source[si][1]+(J_target_mat[2,1] - J_source[si][2,1])*gamma_source[si][2]+(J_target_mat[3,1] - J_source[si][3,1])*gamma_source[si][3]
                S2 = (J_target_mat[1,2] - J_source[si][1,2])*gamma_source[si][1]+(J_target_mat[2,2] - J_source[si][2,2])*gamma_source[si][2]+(J_target_mat[3,2] - J_source[si][3,2])*gamma_source[si][3]
                S3 = (J_target_mat[1,3] - J_source[si][1,3])*gamma_source[si][1]+(J_target_mat[2,3] - J_source[si][2,3])*gamma_source[si][2]+(J_target_mat[3,3] - J_source[si][3,3])*gamma_source[si][3]
            
                zeta_sgm = sqrt(r2)/sigma^4
            
                # Add ζ_σ (Γq⋅∇)(Up - Uq)
                S_target[tidx+1] += zeta_sgm*S1
                S_target[tidx+2] += zeta_sgm*S2
                S_target[tidx+3] += zeta_sgm*S3
            end
        end
    end

    return [U_target...,J_target..., S_target...]

end

function ChainRulesCore.rrule(::typeof(FLOWFMM.direct!), xyz_target, J_target, gamma_source, xyz_source, sigma_source, kernel_source, U_target, J_source, S_target,target_index_count,source_index_count,toggle_sfs)

    UJS = fmm.direct!(xyz_target, copy(J_target), gamma_source, xyz_source, sigma_source, kernel_source, copy(U_target), J_source, copy(S_target) ,target_index_count,source_index_count,toggle_sfs)

    function UJS_pullback(UJSbar) # three sets of cotagents mashed together. Not pretty, but doing them separately is really inefficient. #note: S part currently disabled
        # split UJSbar into parts (using views to avoid allocations) TODO: figure out a nice scheme to pack and unpack these values
        
        #Ū = view(UJSbar,1:3*target_index_count)
        #J̄ = view(UJSbar,3*target_index_count+1:12*target_index_count)
        Ū = UJSbar[1:3*target_index_count] # inefficient allocations, but avoids some errors for now. It might be breaking because of tests that run in ChainRulesTestUtils?
        J̄ = UJSbar[3*target_index_count+1:12*target_index_count]
        S̄ = UJSbar[12*target_index_count+1:15*target_index_count]

        #S̄ = view(UJSbar,12*lenTargets+1:15*target_index_count)
        #U = view(UJS,1:lenU*lenTargets)
        #J = view(UJS,lenU*lenTargets+1:(lenU+lenJ)*lenTargets)
        #S = view(UJS,(lenU+lenJ)*lenTargets:(lenU+lenJ+lenS)*lenTargets)
        c4 = 1/(4*pi)

        xyz_target_bar = zeros(length(xyz_target))
        xyz_source_bar = zeros(length(xyz_source))
        sigma_source_bar = zeros(length(sigma_source))
        gamma_source_bar = zeros(length(gamma_source))
        U_target_bar = zeros(size(U_target))
        U_target_bar .= Ū # passes tests, but this one is pretty trivial.
        c4 = 1/(4*pi)
        J_target_bar = zeros(size(J_target))
        J_source_bar = zeros(size(J_source))
        S_target_bar = zeros(size(S_target))
        S_target_bar .= S̄

        # Contributions from Ū:
        # U is affected by xyz_target, gamma_source, xyz_source, sigma_source, and U_target

        dx = zeros(3)
        for j=1:source_index_count
            for i=1:target_index_count
                iidx = 3*(i-1) # the vectors of vectors are concatenated, so there's some index conversion that needs to happen.
                jidx = 3*(j-1)
                for η=1:3
                    dx[η] = xyz_target[iidx+η] - xyz_source[jidx+η]
                end
                rij = sqrt(sum(dx.^2))
                if rij > 0
                    g,dg = kernel_source.g_dgdr(rij/sigma_source[j])
                    A = g/rij^3
                    B = dg/(rij^4*sigma_source[j]) - 3*g/rij^5
                    for η=1:3
                        x_term = 0.0
                        x_term -= A*ϵ(η,Ū[iidx+1:iidx+3],gamma_source[jidx+1:jidx+3])
                        for a=1:3
                            x_term += B*dx[η]*ϵ(a,dx,gamma_source[jidx+1:jidx+3]*Ū[iidx+a])
                        end
                        x_term *= -c4
                        xyz_target_bar[iidx+η] += x_term # really close but not quite right. maybe an algebra mistake?
                        xyz_source_bar[jidx+η] -= x_term
                    end
                    for a=1:3
                        sigma_source_bar[j] += c4*dg/(rij^2*sigma_source[j]^2)*ϵ(a,dx,gamma_source[jidx+1:jidx+3])*Ū[iidx+a] # passes tests... but I think it's just going to zero.
                        gamma_source_bar[jidx+a] += c4*g/rij^3*ϵ(a,dx,Ū[iidx+1:iidx+3]) # actually passes tests.
                    end
                    
                end
            end
        end

        # Contributions from J̄
        # J is affected by xyz_target, gamma_source, xyz_source, sigma_source, and J_target

        J_target_bar .= J̄

        # not used: U_target, J_source, S_target

        dx = zeros(3)
        crss = zeros(3)
        xyz_term = 0.0
        gamma_term = 0.0
        sigma_term = 0.0
        J̄_mat = zeros(3,3) # remove once testing is done
        for i=1:target_index_count # yes six nested for loops.
            iidx = 3*(i-1)
            for j=1:source_index_count
                jidx = 3*(j-1)
                #J = reshape(view(J_target,iidx+1:iidx+9),(3,3)) # disabled for testing-related reasons.
                J̄_mat .= reshape(J̄[9*(i-1)+1:9*(i-1)+9],(3,3))
                for η=1:3
                    dx[η] = xyz_target[iidx+η] - xyz_source[jidx+η]
                end
                rij = sqrt(sum(dx.^2))
                if rij > 0.0
                    g,dg = kernel_source.g_dgdr(rij/sigma_source[j])
                    ddg = ForwardDiff.derivative(kernel_source.dgdr,rij/sigma_source[j])
                    α = dg/(sigma_source[j]*rij) - 3*g/rij^2
                    β = -c4*g/rij^3
                    #println("α: $α\tβ: $β\ti: $i\tj: $j\trij: $rij")
                    #println("xyz_target: $xyz_target\txyz_source: $xyz_source")
                    for a=1:3
                        crss[a] = -c4/rij^3*ϵ(a,dx,gamma_source[jidx+1:jidx+3])
                        #=for c=1:3
                            for d=1:3
                                crss[a] += ϵ(a,c,d)*dx[c]*gamma_source[jidx+d]
                            end
                        end
                        crss[a] *= -c4/rij^3=#
                    end
                    #println("crss: $crss")
                    for a = 1:3
                        for b=1:3
                            sigma_term = 0.0
                            for c=1:3
                                sigma_term += c4*dg*ϵ(a,b,c)*gamma_source[jidx+c]/rij^2
                                #gamma_term = 0.0
                                gamma_term = -β*ϵ(a,b,c)*J̄_mat[c,b]
                                xyz_term = 0.0
                                for d=1:3
                                    xyz_term += (dx[b]*ϵ(c,a,d) + dx[a]*ϵ(c,b,d))*gamma_source[jidx+d]
                                    gamma_term += α*c4/rij^3*ϵ(a,c,d)*dx[c]*dx[b]*J̄_mat[d,b]
                                end
                                xyz_term *= -α*c4/rij
                                xyz_term += (ddg/sigma_source[j]^2 - 7*dg/(rij*sigma_source[j]) + 15*g/rij^2)*crss[c]*dx[b]*dx[a]
                                xyz_term *= J̄_mat[c,b]/rij^2
                                gamma_source_bar[jidx+a] += gamma_term
                                xyz_target_bar[iidx+a] += xyz_term
                                xyz_source_bar[jidx+a] -= xyz_term
                                #println("gamma_term: $gamma_term\txyz_term: $xyz_term")
                            end
                            sigma_term += (-ddg/sigma_source[j] + 2*dg/rij)*crss[a]*dx[b]
                            sigma_source_bar[j] += J̄_mat[a,b]/sigma_source[j]^2*sigma_term
                            xyz_target_bar[iidx+a] += J̄_mat[b,a]*α*crss[b]
                            xyz_source_bar[jidx+a] -= J̄_mat[b,a]*α*crss[b]
                            #println("sigma_term: $sigma_term")
                        end
                    end
                end
            end
        end

        # Contributions from S̄

        # Return the whole list of input cotangents

        function_bar = NoTangent() # not a closure
        kernel_source_bar = NoTangent() # kernel is a function
        target_index_count_bar = NoTangent() # indices
        source_index_count_bar = NoTangent() # more indices
        return function_bar, xyz_target_bar, J_target_bar, gamma_source_bar, xyz_source_bar, sigma_source_bar, kernel_source_bar, U_target_bar, J_source_bar, S_target_bar, target_index_count_bar, source_index_count_bar, NoTangent()

    end

    return UJS, UJS_pullback

end
ReverseDiff.@grad_from_chainrules fmm.direct!(xyz_target::Vector{<:ReverseDiff.TrackedReal},
                                                  J_target::Vector{<:ReverseDiff.TrackedReal},
                                                  gamma_source::Vector{<:ReverseDiff.TrackedReal},
                                                  xyz_source::Vector{<:ReverseDiff.TrackedReal},
                                                  sigma_source::Vector{<:ReverseDiff.TrackedReal},
                                                  kernel_source,
                                                  U_target::Vector{<:ReverseDiff.TrackedReal},
                                                  J_source::Vector{<:ReverseDiff.TrackedReal},
                                                  S_target::Vector{<:ReverseDiff.TrackedReal},
                                                  target_index_count,
                                                  source_index_count,
                                                  toggle_sfs)

function update_particle_states(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:SubFilterScale, <:Any, <:Any, <:Any},MM,a,b,dt::R3,Uinf,f,g,zeta0) where {R <: ReverseDiff.TrackedReal, R2, V, R3}

    if pfield.transposed == false
        error("Time step pullback for non-transposed scheme not implemented yet! Please set transposed to true.")
    end
    # reformat inputs into vectors
    np = pfield.np
    M1 = cat(map(i->pfield.particles[i].M[:,1], 1:np)...;dims=1)
    X = cat(map(i->pfield.particles[i].X, 1:np)...;dims=1)
    U = cat(map(i->pfield.particles[i].U, 1:np)...;dims=1)
    M2 = cat(map(i->pfield.particles[i].M[:,2], 1:np)...;dims=1)
    M23 = map(i->pfield.particles[i].M[2,3], 1:np)
    J = cat(map(i->reshape(pfield.particles[i].J,9), 1:np)...;dims=1)
    sigma = cat(map(i->pfield.particles[i].sigma, 1:np)...;dims=1)
    Gamma = cat(map(i->pfield.particles[i].Gamma, 1:np)...;dims=1)
    C = cat(map(i->pfield.particles[i].C, 1:np)...;dims=1)
    S = cat(map(i->pfield.particles[i].S, 1:np)...;dims=1)
    
    # get output vector
    states = _update_particle_states(M1,X,U,Uinf,M2,M23,J,sigma,Gamma,C,S,MM,a,b,dt,f,g,zeta0)
    # write output vector to output states
    for i=1:pfield.np
        iidx = 3*(i-1)
        itr = 0
        pfield.particles[i].M[:,1] .= states[itr + iidx + 1:itr + iidx + 3]; itr = 3*np
        pfield.particles[i].X .= states[itr + iidx + 1:itr + iidx + 3]; itr = 6*np
        pfield.particles[i].M[:,2] .= states[itr + iidx + 1:itr + iidx + 3]; itr = 9*np
        pfield.particles[i].M[2,3] = states[itr+i]; itr = 10*np
        pfield.particles[i].Gamma .= states[itr + iidx + 1:itr + iidx + 3]; itr = 13*np
        pfield.particles[i].sigma .= states[itr + i]
    end
    return nothing

end

function _update_particle_states(M1,X,U,Uinf,M2,M23,J,sigma,Gamma,C,S,MM,a,b,dt,f,g,zeta0)

    np = length(sigma)
    J_mat = zeros(eltype(J),(3,3))
    for i=1:np
        iidx = 3*(i-1)
        MM .= zero(MM)
        Γ2 = 0.0
        for η=1:3
            J_mat .= reshape(J[9*(i-1)+1:9*(i-1)+9],(3,3))
            M1[iidx+η] = a*M1[iidx+η] + dt*(U[iidx+η]+Uinf[η])
            X[iidx+η] = X[iidx+η] + b*M1[iidx+η]
            for ξ=1:3
                MM[η] += J_mat[ξ,η]*Gamma[iidx+ξ]
            end
            MM[4] += (f+g)/(1+3*f)*MM[η]*Gamma[iidx+η] - f/(1+3*f)*C[i]*S[iidx+η]*Gamma[iidx+η]*sigma[i]^3/zeta0
            Γ2 += Gamma[iidx+η]^2
        end
        MM[4] /= Γ2
        M23[i] = a*M23[i] - dt*sigma[i]*MM[4]
        sigma[i] += b*M23[i]
        for η=1:3
            M2[iidx+η] = a*M2[iidx+η] + dt*(MM[η] - 3*MM[4]*Gamma[iidx+η] - C[i]*S[iidx+η]*sigma[i]^3/zeta0)
            Gamma[iidx+η] += b*M2[iidx+η]
        end
    end
    return cat(M1,X,M2,M23,Gamma,sigma;dims=1)
    #return [M1...,X...,M2...,M23...,Gamma...,sigma...]

end

function ChainRulesCore.rrule(::typeof(_update_particle_states),M1,X,U,Uinf,M2,M23,J,sigma,Gamma,C,S,MM,a,b,dt,f,g,zeta0)

    states = _update_particle_states(M1,X,U,Uinf,M2,M23,J,sigma,Gamma,C,S,MM,a,b,dt,f,g,zeta0)

    # Properly documenting the mathematics of this pullback requires markdown or Latex (due to a profusion of super/subscripts, special symbols, and implied summations). 
    function state_pullback(state_bar)
        A = (f+g)/(1+3*f)
        B = f/(1+3*f)
        B2 = B/zeta0
        MM = zeros(4)
        dMM4dJ = zeros(3,3)
        # unpack state_bar
        # state_bar contents: M1 (3*np) x (3*np) M2 (3*np) M23 (np) Gamma (3*np) sigma (np)
        np = length(sigma)
        M1outbar = view(state_bar,1:3*np)
        xoutbar = view(state_bar,3*np+1:6*np)
        M2outbar = view(state_bar,6*np+1:9*np)
        M23outbar = view(state_bar,9*np+1:10*np)
        Gammaoutbar = view(state_bar,10*np+1:13*np)
        sigmaoutbar = view(state_bar,13*np+1:14*np)

        M1inbar = zeros(size(M1))
        xinbar = zeros(size(X))
        Ubar = zeros(size(U))
        Uinfbar = zeros(size(Uinf))
        M2inbar = zeros(size(M2))
        M23inbar = zeros(size(M23))
        Jbar = zeros(size(J))
        sigmainbar = zeros(size(sigma))
        Gammainbar = zeros(size(Gamma))
        Cbar = zeros(size(C))
        Sbar = zeros(size(S))
        J_mat = zeros(3,3)
        Jbar_mat = zeros(3,3)

        xinbar .= xoutbar
        for i=1:np
            J_mat .= reshape(J[9*(i-1)+1:9*(i-1)+9],(3,3))
            Jbar_mat .= reshape(Jbar[9*(i-1)+1:9*(i-1)+9],(3,3))
            MM .= 0.0
            Γ2 = 0.0
            # define J_mat
            iidx = 3*(i-1)
            for η=1:3
                Γ2 += Gamma[iidx+η]^2 # Γ2 = |Γⁱ|
                for ξ=1:3
                    MM[η] += J_mat[ξ,η]*Gamma[iidx+ξ] # MM_η = J_ξη*Γ_ξ (summed over ξ). Or, for the non-transposed scheme MM_η = J_ηξ*Γ_ξ. For now I'm just dealing with the transposed scheme.
                end
            end
            for η=1:3
                MM[4] += (A*MM[η]*Gamma[iidx+η] - B2*C[i]*S[iidx+η]*Gamma[iidx+η]*sigma[i]^3)/Γ2
            end
            dMM4dC = 0.0
            dMM4dsigma = 0.0
            sigmabar_term = M23outbar[i] + b*sigmaoutbar[i]
            M23inbar[i] = a*sigmabar_term
            for η = 1:3
                dMM4dGamma_η = -B2*C[i]*S[iidx+η]*sigma[i]^3 - 2*MM[4]*Gamma[iidx+η] + A*MM[η]
                for ξ = 1:3
                    dMM4dJ[ξ,η] = 1/Γ2*A*Gamma[iidx+ξ]*Gamma[iidx+η]
                    dMM4dGamma_η += A*J_mat[η,ξ]*Gamma[iidx+ξ]
                end
                dMM4dGamma_η /= Γ2
                dMM4dC += -B2*S[iidx+η]*Gamma[iidx+η]*sigma[i]^3/Γ2
                dMM4dS_η = -B2*C[i]*Gamma[iidx+η]*sigma[i]^3/Γ2
                dMM4dsigma += -3*B2*C[i]*S[iidx+η]*Gamma[iidx+η]*sigma[i]^2/Γ2

                xbar_term_η = M1outbar[iidx+η]+b*xoutbar[iidx+η]
                M1inbar[iidx+η] = a*xbar_term_η
                Ubar[iidx+η] = dt*xbar_term_η
                Uinfbar[η] += dt*xbar_term_η
                # xinbar already taken care of: xinbar[iidx+η] = xbar[iidx+η]

                gammabar_term_η = M2outbar[iidx+η] + b*Gammaoutbar[iidx+η]
                M2inbar[iidx+η] = a*gammabar_term_η

                for ξ=1:3
                    Jbar_mat[ξ,η] = dt*(Gamma[iidx+ξ] - 3*Gamma[iidx+η]*dMM4dJ[ξ,η])*gammabar_term_η - dt*sigma[i]*dMM4dJ[ξ,η]*sigmabar_term
                end
                sigmainbar[i] -= 3*dt*(Gamma[iidx+η]*dMM4dsigma + C[i]*S[iidx+η]*sigma[i]^2/zeta0)*gammabar_term_η
                Gammainbar[iidx+η] = Gammaoutbar[iidx+η] - 3* dt*(Gamma[iidx+η]*dMM4dGamma_η + MM[4])*gammabar_term_η - dt*sigma[i]*dMM4dGamma_η*sigmabar_term
                for ξ=1:3
                    Gammainbar[iidx+η] += dt*J_mat[η,ξ]*(M2outbar[ξ] + b*Gammaoutbar[iidx+ξ])
                end
                Sbar[iidx+η] = -dt*(3*Gamma[iidx+η]*dMM4dS_η + C[i]*sigma[i]^3/zeta0)*gammabar_term_η - dt*sigma[i]*dMM4dS_η*sigmabar_term
            end
            sigmainbar[i] = sigmaoutbar[i] - dt*(MM[4] + sigma[i]*dMM4dsigma)*sigmabar_term
            Cbar[i] = -dt*sigma[i]*dMM4dC*sigmabar_term
            for η=1:3
                gammabar_term_η = M2outbar[iidx+η] + b*Gammaoutbar[iidx+η]
                Cbar[i] -= dt*(3*Gamma[iidx+η]*dMM4dC + S[iidx+η]*sigma[i]^3/zeta0)*gammabar_term_η
            end
        end
        return NoTangent(), M1inbar, xinbar, Ubar, Uinfbar, M2inbar, M23inbar, Jbar, sigmainbar, Gammainbar, Cbar, Sbar, zeros(eltype(MM),4), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return states, state_pullback

end

ReverseDiff.ReverseDiff.@grad_from_chainrules _update_particle_states(M1::Vector{<:ReverseDiff.TrackedReal},
                                                                      X::Vector{<:ReverseDiff.TrackedReal},
                                                                      U::Vector{<:ReverseDiff.TrackedReal},
                                                                      Uinf::Vector{<:ReverseDiff.TrackedReal},
                                                                      M2::Vector{<:ReverseDiff.TrackedReal},
                                                                      M23::ReverseDiff.TrackedArray,
                                                                      J::Vector{<:ReverseDiff.TrackedReal},
                                                                      sigma::Vector{<:ReverseDiff.TrackedReal},
                                                                      Gamma::Vector{<:ReverseDiff.TrackedReal},
                                                                      C::Vector{<:ReverseDiff.TrackedReal},
                                                                      S::Vector{<:ReverseDiff.TrackedReal},
                                                                      MM::Vector{<:ReverseDiff.TrackedReal},
                                                                      a,b,dt,f,g,zeta0)