# Levi-Civita tensor contractions for convenience. This shows up in cross products.
# ϵ with two vectors -> vector, so need one scalar index
# ϵ with one vector -> matrix, so need two scalar indices
# ϵ with one matrix -> vector, so need one scalar index
ϵ(a::Int,x::Vector,y::Vector) = (a == 1) ? (x[2]*y[3] - x[3]*y[2]) : ((a == 2) ? (x[3]*y[1] - x[1]*y[3]) : ((a == 3) ? (x[1]*y[2] - x[2]*y[1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")))
ϵ(a::Int,b::Int,y::Vector) = (a == b) ? zero(eltype(y)) : ((mod(b-a,3) == 1) ? y[mod(b,3)+1] : ((mod(a-b,3) == 1) ? -y[mod(b-2,3)+1] : error("attempted to evaluate Levi-Civita symbol at out-of-bounds indices $(a) and $(b)!")))
ϵ(a::Int,x::Vector,c::Int) = -1 .*ϵ(a,c,x)
ϵ(a::Int,x::TM) where {TM <: AbstractArray} = (a == 1) ? (x[2,3] - x[3,2]) : (a == 2) ? (x[3,1]-x[1,3]) : (a == 3) ? (x[1,2]-x[2,1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")
ϵ(a::Int,b::Int, c::Int) = (a == b || b == c || c == a) ? 0 : (mod(b-a,3) == 1 ? 1 : -1) # no error checks in this implementation, since that would significantly increase the cost of it
δ(a, b) = a == b ? 1 : 0 # don't need an Int constraint here, since we're just checking equality.

using ChainRulesCore

using ForwardDiff
const c4 = 1/(4*pi)

function fmm.direct!(target_buffer::AbstractArray{<:ReverseDiff.TrackedReal{V, D, O}}, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS,V,D,O}
    target_buffer_val = ReverseDiff.value.(target_buffer)
    target_buffer_val_star = deepcopy(target_buffer_val) # since this is an in-place function, we need to save the overwritten input.
    #source_system_val = ReverseDiff.value(source_system) # TODO: just pass in the particle field directly, since the actual math is done with the buffers anyway.
    source_buffer_val = ReverseDiff.value.(source_buffer)
    tp = ReverseDiff.tape(source_system)
    fmm.direct!(target_buffer_val, target_index, derivatives_switch, source_system, source_buffer_val, source_index)
    for idx in CartesianIndices(target_buffer[:, target_index])
        target_buffer[idx].value = target_buffer_val[idx]
    end
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.direct!,
                        (target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index),
                        target_buffer,
                        (target_buffer_val_star,PS,VS,GS))
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.direct!)})
    
    target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index = instruction.input
    target_buffer_val_star, PS,VS,GS = instruction.cache

    ReverseDiff.value!.(target_buffer, target_buffer_val_star) # map original value back
    
    T = eltype(ReverseDiff.value(target_buffer[1]))
    Γ = zeros(T,3)
    Γbar = zeros(T,3) # Γbar_a^j
    x_source = zeros(T,3)
    #x_source_bar = zeros(T,3)
    x_target = zeros(T,3)
    #x_target_bar = zeros(T,3)
    dx = zeros(T,3)
    γ = zeros(T,3) # called crss in the orignal code
    #σbar = zero(T) # σbar_j
    Ubar = zeros(T,3) # Ubar_a^i
    Jbar = zeros(T,3,3) # Jbar_ab^i
    dxbar = zeros(T, 3)
    γbar = zeros(T,3) # γbar_a^ij

    for i in source_index

        for a=1:3
            Γ[a] = source_buffer[a+4, i].value
        end
        for a=1:3
            x_source[a] = source_buffer[a,i].value
        end
        #x_source_bar .= zero(T)
        σ = source_buffer[8, i].value
        for j in target_index
            # calculate r, dx, and check if particles actually interact
            for a=1:3
                x_target[a] = target_buffer[a,j].value
            end
            #x_target_bar .= zero(T)
            for a=1:3
                dx[a] = x_target[a] - x_source[a]
                dxbar[a] = zero(T)
            end
            Γbar .= zero(T)
            σbar = zero(T)
            r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[3]*dx[3]
            if r2 > 0
                r = sqrt(r2)
                g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/σ)
                ddg_sgmdr = ForwardDiff.derivative(source_system.kernel.dgdr,r/σ) # derivative of g' at r/sigma

                α = dg_sgmdr/(σ*r) - 3*g_sgm/r^2
                β = -const4*g_sgm/r^3
                for a=1:3
                    γ[a] = -const4/r^3 * ϵ(a,dx,Γ)
                end
                # reset containers
                rbar = zero(T)
                αbar = zero(T)
                βbar = zero(T)
                for a=1:3
                    γbar[a] = zero(T)
                end
                if VS
                    @views Ubar_target_buffer = fmm.get_gradient(target_buffer, derivatives_switch, j)
                    for a=1:3
                        Ubar[a] = Ubar_target_buffer[a].deriv
                    end

                    for a=1:3
                        rbar += Ubar[a]*dg_sgmdr/σ*γ[a]
                        σbar -= Ubar[a]*dg_sgmdr*r/σ^2*γ[a]
                        γbar[a] += Ubar[a]*g_sgm
                    end
                    
                end
                if GS
                    @views Jbar_target_buffer = fmm.get_hessian(target_buffer, derivatives_switch, j)
                    for a=1:3
                        for b=1:3
                            Jbar[a,b] = Jbar_target_buffer[a, b].deriv # may need to transpose this?
                            αbar += Jbar[a, b] * γ[a] * dx[b]
                            γbar[a] += Jbar[a, b] * α * dx[b]
                            dxbar[b] += Jbar[a, b] * α * γ[a]

                            for c=1:3
                                βbar += Jbar[a, b]*ϵ(a, b, c)*Γ[c]
                                Γbar[c] += Jbar[a, b]*β*ϵ(a, b, c)
                            end
                        end
                    end
                    
                end

                rbar += αbar*(ddg_sgmdr/(σ^2*r) - 4*dg_sgmdr/(σ*r^2) + 6*g_sgm/r^3)
                σbar += αbar*(-ddg_sgmdr/σ^3 + 2*dg_sgmdr/(σ^2*r))

                rbar += βbar*(-c4*dg_sgmdr/(σ*r^3) + 3*c4*g_sgm/(r^4))
                σbar += βbar*c4*dg_sgmdr/(σ^2*r^2)
                
                for a=1:3
                    for b=1:3
                        for c=1:3
                            rbar += 3*γbar[a]*c4*r^-4*ϵ(a, b, c)*dx[b]*Γ[c]
                            dxbar[b] -= γbar[a]*c4*ϵ(a, b, c)*r^-3*Γ[c]
                            Γbar[c] -= c4*γbar[a]*r^-3*ϵ(a, b, c)*dx[b]
                        end
                    end
                end

                for a=1:3
                    dxbar[a] += rbar*dx[a]/r
                end

            end
            for a=1:3
                ReverseDiff._add_to_deriv!(target_buffer[a, j], dxbar[a])
                ReverseDiff._add_to_deriv!(source_buffer[a+4, i], Γbar[a])
                ReverseDiff._add_to_deriv!(source_buffer[a, i], -dxbar[a])
            end
            ReverseDiff._add_to_deriv!(source_buffer[8, i], σbar)

        end
        #for a=1:3
            #ReverseDiff._add_to_deriv!(source_buffer[a+4, i], Γbar[a])
            #ReverseDiff._add_to_deriv!(source_buffer[a, i], -dxbar[a])
        #end
        #ReverseDiff._add_to_deriv!(source_buffer[8, i], σbar)

    end

    # unseed outputs
    # skipped - since we accumulate into U and J (instead of overwriting them), we do not reset their derivatives.

    #=for j in target_index
        for a=1:3
            target_buffer[a+4, j].deriv = 0.0
        end
        for a=1:3
            for b=1:3
                target_buffer[7 + 3*(b-1) + a, j].deriv = 0.0
            end
        end
    end=#

    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.direct!)})
    
    target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index = instruction.input
    target_buffer_val_star, PS, VS, GS = instruction.cache
    
    target_buffer_val = ReverseDiff.value.(target_buffer)
    source_buffer_val = ReverseDiff.value.(source_buffer)

    for idx in CartesianIndices(target_buffer_val_star)
        target_buffer_val_star[idx] = target_buffer_val[idx]
    end

    fmm.direct!(target_buffer_val, target_index, derivatives_switch, source_system, source_buffer_val, source_index)
    for idx in CartesianIndices(target_buffer[:, target_index])
        target_buffer[idx].value = target_buffer_val[idx]
    end

    return nothing

end

function fmm.source_system_to_buffer!(buffer::AbstractArray{<:ReverseDiff.TrackedReal}, i_buffer, system::ParticleField, i_body)

    buffer_star = deepcopy(ReverseDiff.value.(buffer[1:8, i_buffer]))
    tp = ReverseDiff.tape(buffer, system)

    σ = system.particles[SIGMA_INDEX, i_body].value
    Γx, Γy, Γz = view(system.particles, GAMMA_INDEX, i_body)
    #Γ = sqrt(Γx.value*Γx.value + Γy.value*Γy.value + Γz.value*Γz.value)
    Γ2 = Γx.value*Γx.value + Γy.value*Γy.value + Γz.value*Γz.value
    Γ = Γ2 > 0 ? sqrt(Γ2) : zero(typeof(Γ2))
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)
    for i=1:3
        buffer[i, i_buffer].value = system.particles[X_INDEX[i], i_body].value
    end
    buffer[4, i_buffer].value = ρ_σ * σ
    for i=1:3
        buffer[i+4, i_buffer].value = system.particles[GAMMA_INDEX[i], i_body].value
    end
    buffer[8, i_buffer].value = σ

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.source_system_to_buffer!,
                        (buffer, i_buffer, system, i_body),
                        nothing,
                        (buffer_star, Γ, ρ_σ))
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.source_system_to_buffer!)})

    buffer, i_buffer, system, i_body = instruction.input
    buffer_star, Γ, ρ_σ = instruction.cache

    for idx in 1:8
        buffer[idx, i_buffer].value = buffer_star[idx]
    end
    
    σ = system.particles[SIGMA_INDEX, i_body].value
    #Γx = system.particles[GAMMA_INDEX[1], i_body].value
    #Γy = system.particles[GAMMA_INDEX[2], i_body].value
    #Γz = system.particles[GAMMA_INDEX[3], i_body].value
    #Γ = sqrt(Γx^2 + Γy^2 + Γz^2)
    #ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)
    for i=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[i], i_body], buffer[i, i_buffer].deriv)
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[i], i_body], buffer[i+4, i_buffer].deriv)
    end
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[8, i_buffer].deriv)

    dρ_σ_dσ = ForwardDiff.derivative(_σ->(solve_ρ_over_σ(_σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)), σ)
    dρ_σ_dΓ = ForwardDiff.derivative(_Γ->(solve_ρ_over_σ(σ, _Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)), Γ)

    for j=1:3
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[j], i_body], buffer[4, i_buffer].deriv * dρ_σ_dΓ * σ * system.particles[GAMMA_INDEX[j], i_body].value / Γ)
    end
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[4, i_buffer].deriv * (dρ_σ_dσ * σ + ρ_σ))

    T = eltype(buffer[1].deriv)
    
    # manual unseed - we do not want to keep derivatives in the buffer after we map derivatives back to the particle field.
    for idx in 1:8
        buffer[idx, i_buffer].deriv = zero(T)
    end
    
    return nothing

end

# not rigorously tested, but hopefully works. todo: test.
function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.source_system_to_buffer!)})

    buffer, i_buffer, system, i_body = instruction.input

    # store original buffer value, using existing cache
    for idx in CartesianIndices(instruction.cache)
        instruction.cache[idx] = ReverseDiff.value(buffer[idx])
    end

    σ = system.particles[SIGMA_INDEX, i_body].value
    Γx, Γy, Γz = view(system.particles, GAMMA_INDEX, i_body)
    Γ2 = Γx.value*Γx.value + Γy.value*Γy.value + Γz.value*Γz.value
    Γ = Γ2 > 0 ? sqrt(Γ2) : zero(typeof(Γ2))
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)
    for i=1:3
        buffer[i, i_buffer].value = system.particles[X_INDEX[i], i_body].value
    end
    buffer[4, i_buffer].value = ρ_σ * σ
    for i=1:3
        buffer[i+4, i_buffer].value = system.particles[GAMMA_INDEX[i], i_body].value
    end
    buffer[8, i_buffer].value = σ

    return nothing

end

function fmm.get_position_pullback!(system::ParticleField, i, buffer)
    for j=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[j],i], buffer[j].deriv)
    end

    return nothing

end

check_derivs(x; label=nothing) = x
check_derivs_trackedreal() = nothing
check_derivs_trackedarray() = nothing
check_derivs_array_of_trackedreals() = nothing

function check_derivs(x::ReverseDiff.TrackedReal; label=nothing)
    label === nothing ? println("ready to check derivs of TrackedReal") : println("ready to check derivs of TrackedReal $label")
    tp = ReverseDiff.tape(x)

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_trackedreal,
                        (x,),
                        x,
                        label)
    return x
end

function check_derivs(x::ReverseDiff.TrackedArray; label=nothing)

    label === nothing ? println("ready to check derivs of TrackedArray") : println("ready to check derivs of TrackedArray $label")
    tp = ReverseDiff.tape(x)
    #println("tape ID: $(pointer(tp))")

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_trackedarray,
                        (x,),
                        x,
                        label)
    return x

end

function check_derivs(x::AbstractArray{<:ReverseDiff.TrackedReal}; label=nothing)

    label === nothing ? println("ready to check derivs of array of TrackedReals") : println("ready to check derivs of array of TrackedReals $label")
    tp = ReverseDiff.tape(x)
    #println("tape ID: $(pointer(tp))")

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_array_of_trackedreals,
                        (x,),
                        x,
                        label)
    return x

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedreal)})
    label = instruction.cache
    label === nothing ? println("derivative: $(ReverseDiff.deriv(instruction.input[1]))") : println("derivative of $label: $(ReverseDiff.deriv(instruction.input[1]))")
    return nothing

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedarray)})
    label = instruction.cache
    label === nothing ? println("sum of derivatives: $(sum(ReverseDiff.deriv(instruction.input[1])))") : println("sum of derivatives of $label: $(sum(ReverseDiff.deriv(instruction.input[1])))")
    
    tp = ReverseDiff.tape(instruction.input[1])
    #println("tape ID: $(pointer(tp))")
    return nothing

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_array_of_trackedreals)})
    label = instruction.cache
    label === nothing ? println("sum of derivatives: $(sum(ReverseDiff.deriv.(instruction.input[1])))") : println("sum of derivatives of $label: $(sum(ReverseDiff.deriv.(instruction.input[1])))")
    
    tp = ReverseDiff.tape(instruction.input[1])
    #println("tape ID: $(pointer(tp))")
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedreal)})
    return nothing
end
@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedarray)})
    return nothing
end
@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_array_of_trackedreals)})
    return nothing
end

check_deriv_allocation(x; label=nothing) = x
check_deriv_allocation_trackedarray() = error() # dummy function
check_deriv_allocation_array_of_trackedreals() = error() # dummy function
function check_deriv_allocation(x::ReverseDiff.TrackedArray; label=nothing)

    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(x.value)
    one_x_val = one(eltype(x.value))
    for xi in x.value
        xi += one_x_val
    end
    s2 = sum(x.value)
    for xi in x.value
        xi -= one_x_val
    end
    s3 = sum(x.value)
    if abs(s-s3) > ϵ ; error("Initial sum of values $s is not equal to final sum of values $(s3)!"); end
    if abs(s2-s - length(x.value)) > ϵ; error("Perturbation check failed! Initial sum of values is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - length(x.value))"); end
    label === nothing ? println("value of TrackedArray is properly allocated!") : println("value of TrackedArray $label is properly allocated!")

    tp = ReverseDiff.tape(x)

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_deriv_allocation_trackedarray,
                        (x),
                        x,
                        label)
    return x

end

function check_deriv_allocation(x::AbstractArray{<:ReverseDiff.TrackedReal}; label=nothing)

    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(ReverseDiff.value.(x))
    one_x_val = one(eltype(x[1].value))
    for xi in x
        xi.value += one_x_val
    end
    s2 = sum(ReverseDiff.value.(x))
    for xi in x
        xi.value -= one_x_val
    end
    s3 = sum(ReverseDiff.value.(x))
    if abs(s-s3 > ϵ); error("Initial sum of values $s is not equal to final sum of values $(s3)!"); end
    if abs(s2-s - length(x)) > ϵ ; error("Perturbation check failed! Initial sum of values is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - s - length(x))"); end
    label === nothing ? println("value of array of TrackedReals is properly allocated!") : println("value of array of TrackedReals $label is properly allocated!")

    tp = ReverseDiff.tape(x)

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_deriv_allocation_array_of_trackedreals,
                        (x),
                        x,
                        label)
    return x

end
@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_trackedarray)})

    
    return nothing

end
@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_array_of_trackedreals)})

    x = instruction.input
    label = instruction.cache
    ϵ = 1e-6
    tp = ReverseDiff.tape(x)
    s = sum(ReverseDiff.deriv.(x))
    #@show s
    one_x_deriv = one(eltype(x[1].deriv))
    for xi in x
        xi.deriv += one_x_deriv
    end
    s2 = sum(ReverseDiff.deriv.(x))
    for xi in x
        xi.deriv -= one_x_deriv
    end
    s3 = sum(ReverseDiff.deriv.(x))
    if abs(s-s3 > ϵ); error("Initial sum of derivs $s is not equal to final sum of derivs $(s3)!"); end
    if abs(s2-s - length(x)) > ϵ ; error("Perturbation check failed! Initial sum of derivs is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - s)"); end
    label === nothing ? println("derivative of array of TrackedReals is properly allocated!") : println("derivative of array of TrackedReals $label is properly allocated!")

    if length(tp) == 0
        label === nothing ? error("tape has length zero!") : error("tape of $label has length zero!")
    end
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_deriv_allocation_trackedarray)})
    return nothing
end

# buffer[1:3, i_body] .= get_position(system, i_sorted)
function fmm.position_to_buffer__value!(buffer, i_body, system::ParticleField, i_sorted, buffer_star)
    for a=1:3
        buffer_star[a, i_body] = buffer[a, i_body].value
        buffer[a, i_body,].value = system.particles[X_INDEX[a], i_sorted].value
    end
    return nothing
end

#=function fmm.metadata_to_buffer!(buffer, switch, i_buffer, system::ParticleField, i_body)
    previous_potential = zero(eltype(system))
    gx, gy, gz = get_U(system, i_body)
    G = gx*gx + gy*gy + gz*gz
    previous_gradient = G > 0 ? sqrt(G) : zero(eltype(G))
    buffer[fmm.metadata_index(switch, 1), i_buffer] = previous_potential
    buffer[fmm.metadata_index(switch, 2), i_buffer] = previous_gradient
end=#
function fmm.metadata_to_buffer__value!(buffer, switch, i_buffer, system::ParticleField, i_body, buffer_star)
    previous_potential = zero(ReverseDiff.valtype(eltype(system)))#zero(eltype(system))
    U = get_U(system, i_body)
    G2 = U[1].value*U[1].value + U[2].value*U[2].value + U[3].value*U[3].value
    previous_gradient = sqrt(G2)
    buffer_star[4, i_buffer] = buffer[fmm.metadata_index(switch, 1), i_buffer].value
    buffer_star[5, i_buffer] = buffer[fmm.metadata_index(switch, 2), i_buffer].value
    buffer[fmm.metadata_index(switch, 1), i_buffer].value = previous_potential
    buffer[fmm.metadata_index(switch, 2), i_buffer].value = previous_gradient
    return nothing
end

# buffer[1:3, i_body] .= get_position(system, i_sorted)
function fmm.position_to_buffer__pullback!(buffer, i_body, system::ParticleField, i_sorted, buffer_star)
    for a=1:3
        buffer[a, i_body].value = buffer_star[a, i_body] # reset to original value, which is the buffer value at the end of the previous timestep.
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[a], i_sorted], buffer[a, i_body].deriv)
        ReverseDiff.unseed!(buffer[a, i_body])
    end
    return nothing
end

#=function fmm.metadata_to_buffer!(buffer, switch, i_buffer, system::ParticleField, i_body)
    previous_potential = zero(eltype(system))
    gx, gy, gz = get_U(system, i_body)
    G = gx*gx + gy*gy + gz*gz
    previous_gradient = G > 0 ? sqrt(G) : zero(eltype(G))
    buffer[fmm.metadata_index(switch, 1), i_buffer] = previous_potential
    buffer[fmm.metadata_index(switch, 2), i_buffer] = previous_gradient
end=#
function fmm.metadata_to_buffer__pullback!(buffer, switch, i_buffer, system::ParticleField, i_body, buffer_star)

    # Ubar[a] = Gbar*dGdU[a] = Gbar * U[a]/sqrt(G) (if G > 0)
    #previous_potential = zero(eltype(system)) # the only thing needed to handle this is to unseed the relevant buffer entry
    buffer[fmm.metadata_index(switch, 1), i_buffer].value = buffer_star[4, i_body]
    buffer[fmm.metadata_index(switch, 2), i_buffer].value = buffer_star[5, i_body]
    U = get_U(system, i_body)
    G2 = U[1].value*U[1].value + U[2].value*U[2].value + U[3].value*U[3].value
    if G2 > 0 # if zero gradient, then the pullback doesn't matter - the cotangent blows up the infinity, but the existence of a real, deterministic answer ensures that we always have a removeable singularity.
        G = sqrt(G2)
        for a=1:3
            ReverseDiff._add_to_deriv!(U[a], buffer[fmm.metadata_index(switch, 2), i_buffer].deriv * U[a].value/G)
        end
    end
    ReverseDiff.unseed!(buffer[fmm.metadata_index(switch, 1), i_buffer])
    ReverseDiff.unseed!(buffer[fmm.metadata_index(switch, 2), i_buffer])

end

#=
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, target_buffer, i_buffer) where {PS,VS,GS}
    # switch-aware getters -- see the comment on set_gradient! in direct! above. The
    # bare 2-arg forms read hardcoded rows 5:7 / 8:16, which no longer match the
    # buffer layout now that ParticleField declares metadata_per_body = 2.
    if VS
        @views target_system.particles[U_INDEX, i_target] .+= fmm.get_gradient(target_buffer, derivatives_switch, i_buffer)
    end
    if GS
        j = fmm.get_hessian(target_buffer, derivatives_switch, i_buffer)
        for i = 1:9
            target_system.particles[J_INDEX[i], i_target] += j[i]
        end
    end
end
=#

# In-place function that breaks without an explicit rule
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, target_buffer::AbstractArray{<:ReverseDiff.TrackedReal}, i_buffer) where {PS,VS,GS}
    
    tp = ReverseDiff.tape(target_system, target_buffer)
    if VS
        u = fmm.get_gradient(target_buffer, derivatives_switch, i_buffer)
        for i=1:3
            target_system.particles[U_INDEX[i], i_target].value += u[i].value
        end
    end
    if GS
        j = fmm.get_hessian(target_buffer, derivatives_switch, i_buffer)
        for i = 1:9
            target_system.particles[J_INDEX[i], i_target].value += j[i].value
        end
    end
    
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.buffer_to_target_system!,
                        (target_system, i_target, derivatives_switch, target_buffer, i_buffer, VS, GS),
                        nothing)
    return nothing
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer, VS, GS = instruction.input
    if VS
        u = fmm.get_gradient(target_buffer, derivatives_switch, i_buffer)
        for i=1:3
            target_system.particles[U_INDEX[i], i_target].value -= u[i].value
            ReverseDiff._add_to_deriv!(u[i], target_system.particles[U_INDEX[i], i_target].deriv)
        end
    end
    if GS
        j = fmm.get_hessian(target_buffer, derivatives_switch, i_buffer)
        for i=1:9
            target_system.particles[J_INDEX[i], i_target].value -= j[i].value
            ReverseDiff._add_to_deriv!(j[i], target_system.particles[J_INDEX[i], i_target].deriv)
        end
    end
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer, VS, GS = instruction.input
    if VS
        u = fmm.get_gradient(target_buffer, i_buffer)
        for i=1:3
            target_system.particles[U_INDEX[i], i_target].value += u[i].value
        end
    end
    if GS
        j = fmm.get_hessian(target_buffer, i_buffer)
        for i = 1:9
            target_system.particles[J_INDEX[i], i_target].value += j[i].value
        end
    end

    return nothing

end

add!(A,B) = A += B
add!_trackedreal() = error("dummy function")
function add!(A::ReverseDiff.TrackedReal, B::ReverseDiff.TrackedReal)
    Astar = A.value
    A.value += B.value
    tp = ReverseDiff.tape(A, B)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        add!_trackedreal,
                        (A, B),
                        A)

    return A
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(add!_trackedreal)})

    A, B = instruction.input
    A.value -= B.value
    B.deriv += A.deriv
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(add!_trackedreal)})

    A, B = instruction.input
    A.value += B.value
    return nothing

end

# this is needed because += is not overloadable. In theory, I guess I could also check the implementation of assignment...
add!(A::AbstractArray, B::AbstractArray) = A .+= B
function add!(A::AbstractArray{<:ReverseDiff.TrackedReal}, B::AbstractArray{<:ReverseDiff.TrackedReal})
    Astar = deepcopy(ReverseDiff.value.(A))
    for i=1:length(A)
        A[i] += B[i]
    end
    tp = ReverseDiff.tape(A, B)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        add!_mat,
                        (A, B),
                        A,
                        Astar)

    return A
end
add!_mat() = error("dummy function")

#=
check_deriv_allocation(x;label=nothing) = x

function check_deriv_allocation(x::AbstractArray{<:ReverseDiff.TrackedReal}; label=nothing)

    s = sum(x)
    for idx in x
        x[idx].value += 1.0
    end
    s2 = sum(x)
    if abs((s + length(x)) - s2) > 1e-12
        error("Perturbed sum $s2 does not match original sum $s for array $(label === nothing ? nothing : label) of length $(length(x))")
    end

    record!(tp,
            ReverseDiff.SpecialInstruction,
            check_deriv_allocation,
            (x,),
            x)

end

=#

function fmm.get_previous_influence_pullback!(system::ParticleField, i, buffer)
    #=prev_potential = zero(eltype(system))
    gx, gy, gz = get_U(system, i)
    return prev_potential, sqrt(gx*gx + gy*gy + gz*gz)=#

    gx, gy, gz = get_U(system, i)
    G2 = ReverseDiff.value(gx*gx + gy*gy + gz*gz)
    G2 > 0 ? G = sqrt(G2) : return nothing
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[1],i], buffer[2].deriv*gx.value/G)
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[2],i], buffer[2].deriv*gy.value/G)
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[3],i], buffer[2].deriv*gz.value/G)
    return nothing

end

# reverse pass - in the same format as a direct! call.
#=function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.direct!)})
    
    target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index = instruction.input
    target_buffer_val_star, PS,VS,GS = instruction.cache

    ReverseDiff.value!.(target_buffer, target_buffer_val_star) # map original value back
    
    T = eltype(ReverseDiff.value(target_buffer[1]))
    Γ = zeros(T,3)
    Γbar = zeros(T,3) # Γbar_a^j
    x_source = zeros(T,3)
    x_target = zeros(T,3)
    dx = zeros(T,3)
    Ubar = zeros(T,3) # Ubar_a^i
    Jbar = zeros(T,3,3) # Jbar_ab^i
    dxbar = zeros(T, 3)

    gradr_m1 = zeros(T, 3)
    grad2r_m1 = zeros(T, 3, 3)
    grad3r_m1 = zeros(T, 3, 3, 3)

    for i in source_index

        Γ_buffer = fmm.get_strength(source_buffer, source_system, i)
        for a=1:3
            Γ[a] = Γ_buffer[a].value
        end
        x_source_buffer = fmm.get_position(source_buffer, i)
        for a=1:3
            x_source[a] = x_source_buffer[a].value
        end
        σ = source_buffer[8, i].value
        for j in target_index

            # calculate r, dx, and check if particles actually interact
            x_target_buffer = fmm.get_position(target_buffer, j)
            for a=1:3
                x_target[a] = x_target_buffer[a].value
            end
            for a=1:3
                dx[a] = x_target[a] - x_source[a]
                dxbar[a] = zero(T)
                Γbar[a] = zero(T)
            end
            r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[3]*dx[3]
            if r2 > 0

                r = sqrt(r2)
                # explicitly calculate gradients of r^-1 here; in the actual FMM implementation we have this available.
                for a=1:3
                    gradr_m1[a] = -dx[a]/r^3
                    for b=1:3
                        grad2r_m1[a, b] = -3*dx[a]*dx[b]/r^5 + δ(a, b)/r^3
                        for c=1:3
                            grad3r_m1[a, b, c] = 15*dx[a]*dx[b]*dx[c]/r^7 - 3/r^5*(δ(a, b)*dx[c] + δ(a, c)*dx[b] + δ(b, c)*dx[a])
                        end
                    end
                end

                if VS
                    @views Ubar_target_buffer = fmm.get_gradient(target_buffer, derivatives_switch, j)
                    for a=1:3
                        Ubar[a] = Ubar_target_buffer[a].deriv
                    end

                    for a=1:3
                        for b=1:3
                            for c=1:3
                                for d=1:3
                                    dxbar[a] -= const4*Ubar[b]*ϵ(b, c, d)*grad2r_m1[a, c]*Γ[d]
                                end
                                Γbar[a] -= const4*ϵ(a, b, c)*gradr_m1[b]*Ubar[c]
                            end
                        end
                    end
                    
                end
                if GS
                    @views Jbar_target_buffer = fmm.get_hessian(target_buffer, derivatives_switch, j)
                    for a=1:3
                        for b=1:3
                            Jbar[a,b] = Jbar_target_buffer[a, b].deriv
                        end
                    end
                    for a=1:3
                        for b=1:3
                            for c=1:3
                                for e=1:3
                                    for d=1:3
                                        dxbar[a] -= const4*Jbar[e, b]*ϵ(c, d, e)*grad3r_m1[a, b, c]*Γ[d]
                                    end
                                    Γbar[a] += const4*Jbar[e, b]*ϵ(a, c, e)*grad2r_m1[b, c]
                                end
                            end
                        end
                    end
                    
                end

            end
            for a=1:3
                ReverseDiff._add_to_deriv!(x_target_buffer[a], dxbar[a])
                ReverseDiff._add_to_deriv!(Γ_buffer[a], Γbar[a])
                ReverseDiff._add_to_deriv!(x_source_buffer[a], -dxbar[a])
            end
            
        end
    end


    return nothing

end=#

struct Xbar_Target{T}

    xbar_target::T
    x::T
    gamma::T
    ubar::T
    jbar::T

end
Base.eltype(x::Xbar_Target) = eltype(x.xbar_target)

struct Xbar_Source{TA}

    xbar_source::TA
    x::TA
    gamma::TA
    ubar::TA
    jbar::TA

end
Base.eltype(x::Xbar_Source) = eltype(x.xbar_source)

struct Gammabar_Source{TA}

    gammabar::TA
    x::TA
    gamma::TA
    ubar::TA
    jbar::TA

end
Base.eltype(x::Gammabar_Source) = eltype(x.gammabar)

# fmm.fmm! call, specialized for ::ParticleField{<:ReverseDiff.TrackedReal}.
# First, we extract the primal values from the particle field. This is actually pretty inefficient, since it involves allocating an entire new particle field.
#    Ideally, we replace the particle array in the new particle field with a view into the original particle array (with an additional call to ReverseDiff.value for each entry).
#    However, the particle array is an array of TrackedReals, so a simple call to view() does not work the way we want.
#    We could change the access rules for ParticleField{<:ReverseDiff.TrackedReal} to access just the value part.
#    We could also add a struct that contains an array and a toggle for accessing the value or derivative.
#    For now, I just allocate a new particle field.
# Second, we save the original values of U and J. We need to revert the value of the particle field in the reverse pass;
#    we can either run the inverse of the primal function or save the state before the function call for further use.
#    Here, we could do either. Saving and reloading the value is faster if we compile the function and is also much easier to implement,
#    so I opted for saving the state of U and J.
#    Saving the original value of arrays that are updated in-place is also the reason we have to write our our function for the forward pass.
# Third, we grab the tape from the particle field. Later this is the tape we will record the fmm.fmm! call to.
# Fourth, we run the primal function.
# Fifth, we record the function call to the tape. We write a 'SpecialInstruction' to the tape 'tp'; a 'SpecialInstruction' is the kind of container used for any custom pullback.
#    We record which function was called, and this is used to determine which reverse pass function to call later.
#    We record all inputs to fmm.fmm! we care about: (pfield, optargs)
#    We record all outputs from fmm.fmm! we care about: args
#    We also record any other data we want later in a cache: (ustar, jstar)
# Finally, we return the output of the primal function call. Because 'args' is not a differentiable object, we can return it with no further changes.
function fmm.fmm!(pfield::ParticleField{<:ReverseDiff.TrackedReal}; optargs...)
    
    pfield_val = ReverseDiff.value(pfield) # allocates a copy of pfield
    u_star = deepcopy(pfield_val.particles[U_INDEX, :]) # unavoidable allocations
    j_star = deepcopy(pfield_val.particles[J_INDEX, :]) # unavoidable allocations
    tp = ReverseDiff.tape(pfield)
    args = fmm.fmm!(pfield_val; optargs...) # unpacking components of tracked reals seems to be best accomplished in the compatiblity overloads for FastMultipole.
    
    # For now, the extracted values do not share memory with the original particle array.
    # We need to manually map the altered states to the original pfield.
    for i=1:pfield.np
        for j=1:length(U_INDEX)
            pfield.particles[U_INDEX[j], i].value = pfield_val.particles[U_INDEX[j], i]
        end
        for j=1:length(J_INDEX)
            pfield.particles[J_INDEX[j], i].value = pfield_val.particles[J_INDEX[j], i]
        end
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.fmm!,
                        (pfield, optargs),
                        args,
                        (pfield_val, u_star, j_star))
    return args

end

# The reverse pass for fmm.fmm!, specialized for ::ParticleField{<:ReverseDiff.TrackedReal}.
# First, unpack the input, output, and cache from the tape instruction.
# Second, revert any changes to the primal values. As we run from the end of the program to the beginning,
#    we need primal values to always match their original states.
# Third, we run the pullback for fmm.fmm!. We can accomplish this by running fmm.fmm! calls on three new objects.
#    We build three wrapper structs that reference particle locations, particle strengths, cotangents of particle velocities,
#    and cotangents of particle velocity gradients. Each container also sees the cotangents of the field that they are used
#    to calculate pullbacks for.
#    Each container contains purely real-valued arrays, since we unpack the particle field into real and cotangent components.
#    As a result, all fmm calls on the containers are also purely real-valued; we never pass AD directly through the FMM.
#    of the original pfield in this function call.
# Finally, we return nothing - the reverse pass always updates the value and contangent in-place.
function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.fmm!)})
    
    pfield, optargs = instruction.input
    args = instruction.output # may be useful if we can re-use interaction lists
    pfield_val, u_star, j_star = instruction.cache

    # revert u and j values to original states
    for i=1:pfield.np
        for j=1:length(U_INDEX)
            pfield.particles[U_INDEX[j], i].value = u_star[j, i]
        end
        for j=1:length(J_INDEX)
            pfield.particles[J_INDEX[j], i].value = j_star[j, i]
        end
    end

    pfield_deriv = ReverseDiff.deriv(pfield) # very inefficient
    # (source location, source strength, target location, ubar/jbar) -> (xbar_source, xbar_target, gammabar_source)                 
    # we need to make new containers that have different direct interactions
    # these containers are non-allocating, since they share memory with the original particle array.
    xbar_target = Xbar_Target(
                            view(pfield_deriv.particles, X_INDEX, :),
                            view(pfield_val.particles, X_INDEX, :),
                            view(pfield_val.particles, GAMMA_INDEX, :),
                            view(pfield_deriv.particles, U_INDEX, :),
                            view(pfield_deriv.particles, J_INDEX, :)
                            )
    #=xbar_target = Xbar_Target(
                            pfield_deriv.particles[X_INDEX, :],
                            pfield_val.particles[X_INDEX, :],
                            pfield_val.particles[GAMMA_INDEX, :],
                            pfield_deriv.particles[U_INDEX, :],
                            pfield_deriv.particles[J_INDEX, :]
                            )                   =#
    xbar_source = Xbar_Source(
                            view(pfield_deriv.particles, X_INDEX, :),
                            view(pfield_val.particles, X_INDEX, :),
                            view(pfield_val.particles, GAMMA_INDEX, :),
                            view(pfield_deriv.particles, U_INDEX, :),
                            view(pfield_deriv.particles, J_INDEX, :)
                            )                     
    gammabar_source = Gammabar_Source(
                            view(pfield_deriv.particles, GAMMA_INDEX, :),
                            view(pfield_val.particles, X_INDEX, :),
                            view(pfield_val.particles, GAMMA_INDEX, :),
                            view(pfield_deriv.particles, U_INDEX, :),
                            view(pfield_deriv.particles, J_INDEX, :)
                            )            
    args = fake_fmm!(xbar_target; 
                        optargs...)
    args = fake_fmm!(xbar_source; 
                        optargs...)
    args = fake_fmm!(gammabar_source; 
                        optargs...)
    for i=1:pfield.np
        for j=1:length(X_INDEX)
            pfield.particles[X_INDEX[j], i].deriv = xbar_target.xbar_target[j, i]
        end
        for j=1:length(GAMMA_INDEX)
            pfield.particles[GAMMA_INDEX[j], i].deriv = gammabar_source.gammabar[j, i]
        end
    end

    return nothing

end

# The forward pass when tape is compiled. It runs the forward pass, but we already have inputs and caches pre-allocated.
# We just update the cache (stored primal values for the reverse pass) and run the primal fmm call.
# This call should be non-allocating (aside from any allocations in the fmm call itself).
function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.fmm!)})
    pfield, optargs = instruction.input
    pfield_val, u_star, j_star = instruction.cache

    for i=1:pfield.np
        for j=1:length(U_INDEX)
            u_star[U_UNDEX[j], i] = pfield.particles[U_INDEX[j], i].value
        end
        for j=1:length(J_INDEX)
            j_star[U_UNDEX[j], j] = pfield.particles[U_INDEX[j], j].value
        end
    end
    args = fmm.fmm!(pfield_val; optargs...)
    instruction.output = args
    for i=1:pfield.np
        for j=1:length(U_INDEX)
            pfield.particles[U_INDEX[j], i].value = pfield_val.particles[U_INDEX[j], i]
        end
        for j=1:length(J_INDEX)
            pfield.particles[J_INDEX[j], i].value = pfield_val.particles[J_INDEX[j], i]
        end
    end

    return nothing
end

# equations this implements:
# for all a, b, c, d, i, j:
# xbar_targetⁱ[a] += -const4*Ubarⁱ[b]*ϵ(b,c,d) *∇ᵢ∇ᵢr⁻¹[a,c]ⁱʲ*Γʲ[d]
# for all a, b, c, d, e, i, j:
# xbar_targetⁱ[a] += -const4*Jbarⁱ[e,b]*ϵ(c,d,e)*∇ᵢ∇ᵢ∇ᵢr⁻¹[a,b,c]ⁱʲ*Γʲ[d]
function fake_fmm!(system::Xbar_Target; optargs...)
    T = eltype(system)
    dx = zeros(T, 3)
    #gradr_m1 = zeros(T, 3)
    grad2r_m1 = zeros(T, 3, 3)
    grad3r_m1 = zeros(T, 3, 3, 3)
    ntargets = size(system.x)[2]
    nsources = size(system.x)[2]
    for i=1:ntargets
        for j=1:nsources
            r2 = zero(T)
            for a=1:3
                dx[a] = system.x[a, i] - system.x[a, j]
                r2 += dx[a]^2
            end
            if r2 > 0
                r = sqrt(r2)
                # without actually running the FMM, we have to manually calculate gradients of 1/r:
                for a=1:3
                    #gradr_m1[a] = -dx[a]/r^3
                    for b=1:3
                        grad2r_m1[a, b] = -3*dx[a]*dx[b]/r^5 + δ(a, b)/r^3
                        for c=1:3
                            grad3r_m1[a, b, c] = 15*dx[a]*dx[b]*dx[c]/r^7 - 3/r^5*(δ(a, b)*dx[c] + δ(a, c)*dx[b] + δ(b, c)*dx[a])
                        end
                    end
                end
                # for all a, b, c, d, i, j:
                # xbar_targetⁱ[a] += -const4*Ubarⁱ[b]*ϵ(b,c,d) *∇ᵢ∇ᵢr⁻¹[a,c]ⁱʲ*Γʲ[d]
                # for all a, b, c, d, e, i, j:
                # xbar_targetⁱ[a] += -const4*Jbarⁱ[e,b]*ϵ(c,d,e)*∇ᵢ∇ᵢ∇ᵢr⁻¹[a,b,c]ⁱʲ*Γʲ[d]
                for a=1:3, b=1:3, c=1:3, d=1:3
                    system.xbar_target[a, i] += -const4*system.ubar[b, i]*ϵ(b,c,d)*grad2r_m1[a,c]*system.gamma[d, j]
                    for e=1:3
                        #system.xbar_target[a, i] += -const4*system.jbar[e, b, i]*ϵ(c,d,e)*grad3r_m1[a,b,c]*system.gamma[d, j]
                        system.xbar_target[a, i] += -const4*system.jbar[3*(e-1) + b, i]*ϵ(c,d,e)*grad3r_m1[a,b,c]*system.gamma[d, j]
                    end
                end
            end
        end
    end
    return nothing
end

# equations this implements:
# for all a, b, c, d, i, j:
# xbar_sourceʲ[a] += const4*Ubarⁱ[b]*ϵ(b,c,d)*∇ⱼ∇ⱼr⁻¹[a,c]ⁱʲΓʲ[d]
# for all a, b, c, d, e, i, j:
# xbar_sourceʲ[a] += -const4*Jbarⁱ[e,b]*ϵ(c,d,e)*∇ⱼ∇ⱼ∇ⱼr⁻¹[a,b,c]ⁱʲΓʲ[d]
function fake_fmm!(system::Xbar_Source; optargs...)
    T = eltype(system)
    dx = zeros(T, 3)
    #gradr_m1 = zeros(T, 3)
    grad2r_m1 = zeros(T, 3, 3)
    grad3r_m1 = zeros(T, 3, 3, 3)
    ntargets = size(system.x)[2]
    nsources = size(system.x)[2]    
    for i=1:ntargets
        for j=1:nsources
            r2 = zero(T)
            for a=1:3
                dx[a] = system.x[a, i] - system.x[a, j]
                r2 += dx[a]^2
            end
            
            if r2 > 0
                r = sqrt(r2)
                # without actually running the FMM, we have to manually calculate gradients of 1/r:
                for a=1:3
                    #gradr_m1[a] = -dx[a]/r^3
                    for b=1:3
                        grad2r_m1[a, b] = -3*dx[a]*dx[b]/r^5 + δ(a, b)/r^3
                        for c=1:3
                            grad3r_m1[a, b, c] = 15*dx[a]*dx[b]*dx[c]/r^7 - 3/r^5*(δ(a, b)*dx[c] + δ(a, c)*dx[b] + δ(b, c)*dx[a])
                        end
                    end
                end
                # for all a, b, c, d, i, j:
                # xbar_sourceʲ[a] += const4*Ubarⁱ[b]*ϵ(b,c,d)*∇ⱼ∇ⱼr⁻¹[a,c]ⁱʲΓʲ[d]
                # for all a, b, c, d, e, i, j:
                # xbar_sourceʲ[a] += const4*Jbarⁱ[e,b]*ϵ(c,d,e)*∇ⱼ∇ⱼ∇ⱼr⁻¹[a,b,c]ⁱʲΓʲ[d]
                for a=1:3, b=1:3, c=1:3, d=1:3
                    system.xbar_source[a, j] += const4*system.ubar[b, i]*ϵ(b,c,d)*grad2r_m1[a,c]*system.gamma[d, j]
                    for e=1:3
                        #system.xbar_source[a, j] += const4*system.jbar[e, b, i]*ϵ(c,d,e)*grad3r_m1[a,b,c]*system.gamma[d, j]
                        system.xbar_source[a, j] += const4*system.jbar[3*(e-1) + b, i]*ϵ(c,d,e)*grad3r_m1[a,b,c]*system.gamma[d, j]
                    end
                end
            end
        end
    end
    return nothing
end

# equations this implements:
# for all a, b, c, i, j:
# Γbarʲ[a] += -const4*ϵ(a,b,c)*∇ⱼr⁻¹[b]ⁱʲUbarⁱ[c]
# for all a, b, c, e, i, j:
# Γbarʲ[a] += const4*Jbarⁱ[e,b]*ϵ(a,c,e)*∇ⱼ∇ⱼr⁻¹[b,c]ⁱʲ
# note that we do not loop over d in these equations.
function fake_fmm!(system::Gammabar_Source; optargs...)
    T = eltype(system)
    dx = zeros(T, 3)
    gradr_m1 = zeros(T, 3)
    grad2r_m1 = zeros(T, 3, 3)
    #grad3r_m1 = zeros(T, 3, 3, 3)
    ntargets = size(system.x)[2]
    nsources = size(system.x)[2] 
    for i=1:ntargets
        for j=1:nsources
            r2 = zero(T)
            for a=1:3
                dx[a] = system.x[a, i] - system.x[a, j]
                r2 += dx[a]^2
            end
            if r2 > 0
                r = sqrt(r2)
                # without actually running the FMM, we have to manually calculate gradients of 1/r:
                for a=1:3
                    gradr_m1[a] = -dx[a]/r^3
                    for b=1:3
                        grad2r_m1[a, b] = -3*dx[a]*dx[b]/r^5 + δ(a, b)/r^3
                        #for c=1:3
                        #    grad3r_m1[a, b, c] = 15*dx[a]*dx[b]*dx[c]/r^7 - 3/r^5*(δ(a, b)*dx[c] + δ(a, c)*dx[b] + δ(b, c)*dx[a])
                        #end
                    end
                end
                # for all a, b, c, i, j:
                # Γbarʲ[a] += -const4*ϵ(a,b,c)*∇ⱼr⁻¹[b]ⁱʲUbarⁱ[c]
                # for all a, b, c, e, i, j:
                # Γbarʲ[a] += -const4*Jbarⁱ[e,b]*ϵ(a,c,e)*∇ⱼ∇ⱼr⁻¹[b,c]ⁱʲ
                for a=1:3, b=1:3, c=1:3
                    system.gammabar[a, j] += -const4*ϵ(a,b,c)*gradr_m1[b]*system.ubar[c, i]
                    for e=1:3
                        #system.gammabar[a, j] += const4*system.jbar[e, b, i]*ϵ(a,c,e)*grad2r_m1[b,c]
                        system.gammabar[a, j] += const4*system.jbar[3*(e-1) + b, i]*ϵ(a,c,e)*grad2r_m1[b,c]
                    end
                end
            end
        end
    end
    return nothing
end