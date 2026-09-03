# Levi-Civita tensor contractions for convenience. This shows up in cross products.
# ϵ with two vectors -> vector, so need one scalar index
# ϵ with one vector -> matrix, so need two scalar indices
# ϵ with one matrix -> vector, so need one scalar index
ϵ(a,x::Vector,y::Vector) = (a == 1) ? (x[2]*y[3] - x[3]*y[2]) : ((a == 2) ? (x[3]*y[1] - x[1]*y[3]) : ((a == 3) ? (x[1]*y[2] - x[2]*y[1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")))
ϵ(a,b::Number,y::Vector) = (a == b) ? zero(eltype(y)) : ((mod(b-a,3) == 1) ? y[mod(b,3)+1] : ((mod(a-b,3) == 1) ? -y[mod(b-2,3)+1] : error("attempted to evaluate Levi-Civita symbol at out-of-bounds indices $(a) and $(b)!")))
ϵ(a,x::Vector,c::Number) = -1 .*ϵ(a,c,x)
ϵ(a,x::TM) where {TM <: AbstractArray} = (a == 1) ? (x[2,3] - x[3,2]) : (a == 2) ? (x[3,1]-x[1,3]) : (a == 3) ? (x[1,2]-x[2,1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")
ϵ(a,b::Number, c::Number) = (a == b || b == c || c == a) ? 0 : (mod(b-a,3) == 1 ? 1 : -1) # no error checks in this implementation, since that would significantly increase the cost of it

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
                    for a=1:3
                        Ubar[a] = target_buffer[a+4, j].deriv
                    end

                    for a=1:3
                        rbar += Ubar[a]*dg_sgmdr/σ*γ[a]
                        σbar -= Ubar[a]*dg_sgmdr*r/σ^2*γ[a]
                        γbar[a] += Ubar[a]*g_sgm
                    end
                    
                end
                if GS
                    for a=1:3
                        for b=1:3
                            Jbar[a,b] = target_buffer[7 + 3*(b-1) + a, j].deriv # may need to transpose this?
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
    
    for j in target_index
        for a=1:3
            target_buffer[a+4, j].deriv = 0.0
        end
        for a=1:3
            for b=1:3
                target_buffer[7 + 3*(b-1) + a, j].deriv = 0.0
            end
        end
    end

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
    Γ = sqrt(Γx.value*Γx.value + Γy.value*Γy.value + Γz.value*Γz.value)
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
    Γ = sqrt(Γx.value*Γx.value + Γy.value*Γy.value + Γz.value*Γz.value)
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

# In-place function that breaks without an explicit rule
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch, target_buffer::AbstractArray{<:ReverseDiff.TrackedReal}, i_buffer)
    
    tp = ReverseDiff.tape(target_system, target_buffer)
    #ustar = ReverseDiff.value.(target_system.particles[U_INDEX, i_target])
    #jstar = ReverseDiff.value.(target_system.particles[J_INDEX, i_target])
    u = fmm.get_gradient(target_buffer, i_buffer)
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value += u[i].value
        #target_system.particles[U_INDEX[i], i_target] = ReverseDiff.track(target_system.particles[U_INDEX[i], i_target] + u[i], tp)
    end
    j = fmm.get_hessian(target_buffer, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target].value += j[i].value
        #target_system.particles[J_INDEX[i], i_target] = ReverseDiff.track(target_system.particles[J_INDEX[i], i_target] + j[i], tp)
    end
    
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.buffer_to_target_system!,
                        (target_system, i_target, derivatives_switch, target_buffer, i_buffer),
                        nothing)
                        #(ustar, jstar))
    return nothing
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer = instruction.input
    #ustar, jstar = instruction.cache
    u = fmm.get_gradient(target_buffer, i_buffer)
    j = fmm.get_hessian(target_buffer, i_buffer)
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value -= u[i].value
        #target_buffer[i+4, i_buffer].deriv += target_system.particles[U_INDEX[i], i_target].deriv
        ReverseDiff._add_to_deriv!(target_buffer[i+4, i_buffer], target_system.particles[U_INDEX[i], i_target].deriv)
        #target_system.particles[U_INDEX[i], i_target].deriv = 0.0
    end
    for i=1:9
        target_system.particles[J_INDEX[i], i_target].value -= j[i].value
        #target_buffer[i+7, i_buffer].deriv += target_system.particles[J_INDEX[i], i_target].deriv
        ReverseDiff._add_to_deriv!(target_buffer[i+7, i_buffer], target_system.particles[J_INDEX[i], i_target].deriv)
        #target_system.particles[J_INDEX[i], i_target].deriv = 0.0
    end
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer = instruction.input
    ustar, jstar = instruction.cache
    ustar .= ReverseDiff.value.(target_system.particles[U_INDEX, i_target])
    jstar .= ReverseDiff.value.(target_system.particles[J_INDEX, i_target])
    u = fmm.get_gradient(target_buffer, i_buffer)
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value += u[i].value
    end
    j = fmm.get_hessian(target_buffer, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target].value += j[i].value
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
    G = ReverseDiff.value(sqrt(gx*gx + gy*gy + gz*gz))
    if G == 0.0
        return nothing
    end
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[1],i], buffer[2].deriv*gx.value/G)
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[2],i], buffer[2].deriv*gy.value/G)
    ReverseDiff._add_to_deriv!(system.particles[U_INDEX[3],i], buffer[2].deriv*gz.value/G)
    return nothing

end
