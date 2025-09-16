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

ReverseDiff.tape(pfield::ParticleField) = ReverseDiff.tape(pfield.particles)

using ForwardDiff
const c4 = 1/(4*pi)

function fmm.direct!(target_buffer::AbstractArray{<:ReverseDiff.TrackedReal{V, D, O}}, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS,V,D,O}
    
    target_buffer_val = ReverseDiff.value.(target_buffer)
    target_buffer_val_star = deepcopy(target_buffer_val) # since this is an in-place function, we need to save the overwritten input.
    source_system_val = ReverseDiff.value(source_system) # TODO: just pass in the particle field directly, since the actual math is done with the buffers anyway.
    source_buffer_val = ReverseDiff.value.(source_buffer)
    tp = ReverseDiff.tape(source_system)
    fmm.direct!(target_buffer_val, target_index, derivatives_switch, source_system_val, source_buffer_val, source_index)
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
    Gamma = zeros(T,3)
    Gamma_i_bar = zeros(T,3)
    x_i = zeros(T,3)
    x_i_bar = zeros(T,3)
    x_j = zeros(T,3)
    x_j_bar = zeros(T,3)
    dx = zeros(T,3)
    crss = zeros(T,3)
    sigma_i_bar = zero(T)
    u_j_bar = zeros(T,3)
    du_j_bar = zeros(T,3,3)

    for i in source_index

        for a=1:3
            Gamma[a] = source_buffer[a+4, i].value
        end
        Gamma_i_bar .= zero(T)
        for a=1:3
            x_i[a] = source_buffer[a,i].value
        end
        x_i_bar .= zero(T)
        sigma = source_buffer[8, i].value
        sigma_i_bar = zero(T)
        for j in target_index
            # calculate r, dx, and check if particles actually interact
            for a=1:3
                x_j[a] = target_buffer[a,j].value
            end
            x_j_bar .= zero(T)
            for a=1:3
                dx[a] = x_j[a] - x_i[a]
            end
            r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[3]*dx[3]
            if r2 > 0
                r = sqrt(r2)
                r3inv = 1/r^3
                g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)
                ddg_sgmdr = ForwardDiff.derivative(source_system.kernel.dgdr,r/sigma) # derivative of g' at r/sigma

                for a=1:3
                    crss[a] = -const4 * r3inv * ϵ(a,dx,Gamma)
                end
                VS = true
                GS = true
                if VS
                    
                    for a=1:3
                        u_j_bar[a] = target_buffer[a+4, j].deriv
                    end
                    
                    A = -const4*g_sgm/r^3
                    B = const4*(dg_sgmdr/(r^4*sigma) - 3*g_sgm/r^5)
                    C = const4*dg_sgmdr/(r2*sigma^2)
                    for a=1:3
                        for b=1:3
                            for c=1:3
                                for d=1:3
                                    x_j_bar[a] -= B*dx[a]*ϵ(b,c,d)*dx[c]*Gamma[d]*u_j_bar[b]
                                    x_i_bar[a] += B*dx[a]*ϵ(b,c,d)*dx[c]*Gamma[d]*u_j_bar[b]
                                end
                                x_j_bar[a] -= A*ϵ(a,b,c)*u_j_bar[b]*Gamma[c]
                                x_i_bar[a] += A*ϵ(a,b,c)*u_j_bar[b]*Gamma[c]
                                sigma_i_bar += C*ϵ(a,b,c)*dx[b]*Gamma[c]*u_j_bar[a]
                                Gamma_i_bar[a] -= A*ϵ(a,b,c)*dx[b]*u_j_bar[c]
                            end
                        end
                    end
                end
                if GS
                    # calculate assorted coefficients that only depend on i and j
                    for a=1:3
                        for b=1:3
                            du_j_bar[b,a] = target_buffer[7 + 3*(b-1) + a, j].deriv
                        end
                    end
                    
                    α = dg_sgmdr/(sigma*r) - 3*g_sgm/r^2
                    β = -const4*g_sgm/r^3
                    γ = (ddg_sgmdr/sigma^2 - 7*dg_sgmdr/(r*sigma) + 15*g_sgm/r^2)
                    for a = 1:3
                        for b=1:3
                            sigma_temp = 0.0
                            for c=1:3
                                sigma_temp += const4*dg_sgmdr*ϵ(a,b,c)*Gamma[c]/r^2
                                gamma_temp = -β*ϵ(a,b,c)*du_j_bar[b, c]
                                xyz_temp = 0.0
                                for d=1:3
                                    xyz_temp += (dx[b]*ϵ(c,a,d) + dx[a]*ϵ(c,b,d))*Gamma[d]
                                    gamma_temp += α*const4/r^3*ϵ(a,c,d)*dx[c]*dx[b]*du_j_bar[b, d]
                                end
                                xyz_temp *= -α*const4/r
                                xyz_temp += γ*crss[c]*dx[b]*dx[a]
                                xyz_temp *= du_j_bar[b, c]/r^2
                                Gamma_i_bar[a] += gamma_temp

                                x_j_bar[a] += xyz_temp
                                x_i_bar[a] -= xyz_temp
                            end
                            sigma_temp += (-ddg_sgmdr/sigma + 2*dg_sgmdr/r)*crss[a]*dx[b]
                            sigma_i_bar += du_j_bar[b, a]/sigma^2*sigma_temp
                            x_j_bar[a] += du_j_bar[a, b]*α*crss[b]
                            x_i_bar[a] -= du_j_bar[a, b]*α*crss[b]
                        end
                    end
                    
                end
            end
            for a=1:3
                ReverseDiff._add_to_deriv!(target_buffer[a, j], x_j_bar[a])
            end

        end
        
        for a=1:3
            ReverseDiff._add_to_deriv!(source_buffer[a+4, i], Gamma_i_bar[a])
            ReverseDiff._add_to_deriv!(source_buffer[a, i], x_i_bar[a])
        end
        ReverseDiff._add_to_deriv!(source_buffer[8, i], sigma_i_bar)

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
                        buffer_star)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.source_system_to_buffer!)})

    buffer, i_buffer, system, i_body = instruction.input
    buffer_star = instruction.cache

    for idx in 1:8
        buffer[idx, i_buffer].value = buffer_star[idx]
    end
    
    σ = system.particles[SIGMA_INDEX, i_body].value
    Γx = system.particles[GAMMA_INDEX[1], i_body].value
    Γy = system.particles[GAMMA_INDEX[2], i_body].value
    Γz = system.particles[GAMMA_INDEX[3], i_body].value
    Γ = sqrt(Γx^2 + Γy^2 + Γz^2)
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)
    for i=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[i], i_body], buffer[i, i_buffer].deriv)
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[i], i_body], buffer[i+4, i_buffer].deriv)
    end
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[8, i_buffer].deriv)

    dρ_σ_dσ = ForwardDiff.derivative(_σ->(solve_ρ_over_σ(_σ, Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)), σ)
    dρ_σ_dΓ = ForwardDiff.derivative(_Γ->(solve_ρ_over_σ(σ, _Γ, system.fmm.relative_tolerance, system.fmm.absolute_tolerance, system.fmm.autotune_reg_error, system.fmm.default_rho_over_sigma)), Γ)

    for j=1:3
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[j], i_body], buffer[4, i_buffer].deriv * dρ_σ_dΓ * system.particles[GAMMA_INDEX[j], i_body].value / Γ)
    end
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[4, i_buffer].deriv * (dρ_σ_dσ + ρ_σ))

    T = eltype(buffer[1].deriv)
    
    # manual unseed - we do not want to keep derivatives in the buffer after we map derivatives back to the particle field.
    for idx in 1:8
        buffer[idx, i_buffer].deriv = zero(T)
    end
    
    return nothing

end

function ρ_σ_residual(ρ_σ, σ, ω, ε_rel, ε_abs, autotune_reg_error, default_rho_over_sigma)

    if false#autotune_reg_error
        if ω < 10*eps()
            return zero(eltype(ω))
        end

        if ε_rel<=eps(10.0)
            return ε_abs <= eps(10.0) ? zero(eltype(σ)) : residual_abs(ρ_σ, σ, ω, ε_abs)
        end
        if ε_abs<=eps(10.0)
            return residual_rel(ρ_σ, ε_rel)
        end

        ρ_over_σ_rel = ε_rel<=eps(10.0) ? one(σ) : Roots.find_zero((x) -> residual_rel(x, ε_rel), (0.0, upper_bound_rel()), Roots.Brent())
        ρ_over_σ_abs = ε_abs<=eps(10.0) ? one(σ) : Roots.find_zero((x) -> residual_abs(x, σ, ω, ε_abs), (0.0, upper_bound_abs(σ, ω, ε_abs)), Roots.Brent())
        return (ρ_over_σ_rel < ρ_over_σ_abs) ? residual_rel(ρ_σ, ε_rel) : residual_abs(ρ_σ, σ, ω, ε_abs)

    else
        return zero(eltype(default_rho_over_sigma))
    end

end

function fmm.source_system_to_buffer_pullback!(buffer, i_buffer, system::ParticleField, i_body)

    σ = system.particles[SIGMA_INDEX, i_body].value
    ε = system.fmm.ε_tol
    Γx = system.particles[GAMMA_INDEX[1], i_body].value
    Γy = system.particles[GAMMA_INDEX[2], i_body].value
    Γz = system.particles[GAMMA_INDEX[3], i_body].value
    Γ = sqrt(Γx^2 + Γy^2 + Γz^2)
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.ε_tol)
    
    for i=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[i], i_body], buffer[i, i_buffer].deriv)
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[i], i_body], buffer[i+4, i_buffer].deriv)
    end
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[8, i_buffer].deriv)

    dr_dρ_σ = ForwardDiff.derivative((_ρ_σ)->residual(_ρ_σ, σ, Γ, ε), ρ_σ)
    dr_dσ = ForwardDiff.derivative((_σ)->residual(ρ_σ, _σ, Γ, ε), σ)
    dr_dω = ForwardDiff.derivative((_Γ)->residual(ρ_σ, σ, _Γ, ε), Γ)
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[1], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γx/Γ)
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[2], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γy/Γ)
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[3], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γz/Γ)
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[4, i_buffer].deriv*(-1/dr_dρ_σ * dr_dσ + ρ_σ))
    T = eltype(buffer.deriv)
    buffer.deriv[:, i_buffer] .= zero(T) # manual unseed - we do not want to keep derivatives in the buffer after we map derivatives back to the particle field.
    
    return nothing

end

function fmm.get_position_pullback!(system::ParticleField, i, buffer)
    for j=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[j],i], buffer[j].deriv)
    end

    return nothing

end

check_derivs(x; label=nothing) = x
check_derivs_trackedarray() = nothing
check_derivs_array_of_trackedreals() = nothing
function check_derivs(x::ReverseDiff.TrackedArray; label=nothing)

    label === nothing ? println("ready to check derivs of TrackedArray") : println("ready to check derivs of TrackedArray $label")
    tp = ReverseDiff.tape(x)

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

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        check_derivs_array_of_trackedreals,
                        (x,),
                        x,
                        label)
    return x

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedarray)})
    label = instruction.cache
    label === nothing ? println("sum of derivatives: $(sum(ReverseDiff.deriv(instruction.input[1])))") : println("sum of derivatives of $label: $(sum(ReverseDiff.deriv(instruction.input[1])))")
    return nothing

end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_array_of_trackedreals)})
    label = instruction.cache
    label === nothing ? println("sum of derivatives: $(sum(ReverseDiff.deriv.(instruction.input[1])))") : println("sum of derivatives of $label: $(sum(ReverseDiff.deriv.(instruction.input[1])))")
    return nothing

end

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_trackedarray)})
    return nothing
end
@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(check_derivs_array_of_trackedreals)})
    return nothing
end

check_deriv_allocation(x; label=nothing) = x
check_deriv_allocation_trackedarray() = nothing # dummy function
check_deriv_allocation_array_of_trackedreals() = x # dummy function
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
    #if abs(s2-s - length(x.value)) > ϵ; error("Perturbation check failed! Initial sum of values is $s, final sum is $s2, and the length of the array is $(length(x)). Difference: $(s2 - length(x.value))"); end
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
    @show s
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

# automatically constructed pullback for adding particles seems to not work properly. probably because it's an in-place operation.
function add_particle(pfield::ParticleField{ReverseDiff.TrackedReal{R, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}, X, Gamma, sigma; vol=0, circulation=1, C=0, static=false) where {R, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}

    # we still need the error checking
    if get_np(pfield)==pfield.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(pfield.maxparticles)"*
                            " has been reached")    
    end
    # Fetch the index of the next empty particle in the field
    i_next = get_np(pfield)+1

    # Add particle to the field
    pfield.np += 1
    
    tp = ReverseDiff.tape(Gamma, X, sigma, pfield)

    for i=1:3
        pfield.particles[X_INDEX[i], i_next] = ReverseDiff.track(X[i].value, tp)
        pfield.particles[GAMMA_INDEX[i], i_next] = ReverseDiff.track(ReverseDiff.value(Gamma[i]), tp)
        if typeof(C) <: AbstractArray
            pfield.particles[C_INDEX[i], i_next] = ReverseDiff.track(C[i], tp)
        else
            pfield.particles[C_INDEX[i], i_next] = ReverseDiff.track(C, tp)
        end
    end
    pfield.particles[SIGMA_INDEX, i_next] = ReverseDiff.track(sigma, tp)
    pfield.particles[VOL_INDEX, i_next] = ReverseDiff.track(vol, tp)
    pfield.particles[CIRCULATION_INDEX, i_next] = ReverseDiff.track(circulation, tp)
    pfield.particles[STATIC_INDEX, i_next] = ReverseDiff.track(static, tp)

    T = eltype(pfield.particles[1].value)
    for i in [U_INDEX..., VORTICITY_INDEX..., J_INDEX..., M_INDEX..., PSE_INDEX..., SFS_INDEX...]
        pfield.particles[i, i_next] = ReverseDiff.track(zero(T), tp)
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        add_particle,
                        (pfield, X, Gamma, sigma, vol, circulation, C, static),
                        pfield)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(add_particle)})
    
    input = instruction.input
    pfield, X, Gamma, sigma, vol, circulation, C, static = input
    i_next = get_np(pfield)
    ReverseDiff.istracked(X) && for i=1:3
        ReverseDiff._add_to_deriv!(X[i], pfield.particles[X_INDEX[i], i_next].deriv)
    end
    ReverseDiff.istracked(Gamma) && for i=1:3
        ReverseDiff._add_to_deriv!(Gamma[i], pfield.particles[GAMMA_INDEX[i], i_next].deriv)
    end
    ReverseDiff.istracked(sigma) && begin 
        ReverseDiff._add_to_deriv!(sigma, ReverseDiff.deriv(pfield.particles[SIGMA_INDEX, i_next]))
    end

    ReverseDiff.istracked(vol) && begin
        ReverseDiff._add_to_deriv!(vol, ReverseDiff.deriv(pfield.particles[VOL_INDEX, i_next]))
    end 

    ReverseDiff.istracked(circulation) && begin
        ReverseDiff._add_to_deriv!(circulation, ReverseDiff.deriv(pfield.particles[CIRCULATION_INDEX, i_next]))
    end

    ReverseDiff.istracked(C) && begin
        ReverseDiff._add_to_deriv!(C, ReverseDiff.deriv(pfield.particles[C_INDEX, i_next]))
    end

    ReverseDiff.istracked(static) && begin
        ReverseDiff._add_to_deriv!(static, ReverseDiff.deriv(pfield.particles[STATIC_INDEX, i_next]))
    end

    ReverseDiff.unseed!(pfield.particles[:, i_next])
    pfield.np -= 1

    return nothing
end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(add_particle)})

    input = instruction.input
    pfield, X, Gamma, sigma, vol, circulation, C, static = input

    if get_np(pfield)==pfield.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(pfield.maxparticles)"*
                            " has been reached")    
    end
    # Fetch the index of the next empty particle in the field
    i_next = get_np(pfield)+1

    # Add particle to the field
    pfield.np += 1

    #set_static(pfield, i_next, Float64(static))
    ReverseDiff.value!.(pfield.particles[i_next, X_INDEX], X)
    ReverseDiff.value!.(pfield.particles[i_next, GAMMA_INDEX], Gamma)
    ReverseDiff.value!(pfield.particles[i_next, SIGMA_INDEX], sigma)
    ReverseDiff.value!(pfield.particles[i_next, VOL_INDEX], vol)
    ReverseDiff.value!(pfield.particles[i_next, CIRCULATION_INDEX], circulation)
    ReverseDiff.value!(pfield.particles[i_next, C_INDEX], C)
    ReverseDiff.value!(pfield.particles[i_next, STATIC_INDEX], Float64(static))

    return nothing

end

# Remove this section as long as it isn't used anywhere.
#=
function copy_to_with_derivs!(dest::ReverseDiff.TrackedArray, di, src, si; warn=true)
    error()
end

function copy_to_with_derivs!(dest::AbstractArray{<:ReverseDiff.TrackedReal}, di::Integer, src::ReverseDiff.TrackedArray, si::Integer)

    dest_val_star = dest[di].value
    tp = ReverseDiff.tape(dest, src)
    dest[di].value = src.value[si]
    dest[di].tape = tp
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        copyto!_ATR_TA,
                        (dest, di, src, si),
                        nothing,
                        dest_val_star)
    return nothing

end
copyto!_ATR_TA() = nothing

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(copyto!_ATR_TA)})

    dest, di, src, si = instruction.input
    dest_val_star = instruction.cache

    dest[di].value = dest_val_star
    src.deriv[si] += dest[di].deriv
    dest[di].deriv = zero(eltype(dest[di].deriv))

    return nothing

end

function Base.copyto!(dest::ReverseDiff.TrackedArray, src::ReverseDiff.TrackedArray)

    dest_val_star = deepcopy(dest.value)
    copyto!(dest.value, src.value)
    tp = ReverseDiff.tape(dest, src)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        Base.copyto!,
                        (dest, src),
                        nothing,
                        dest_val_star)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(Base.copyto!)})

    dest, src = instruction.input
    dest_val_star = instruction.cache

    copyto!(dest.value, dest_val_star)
    src.deriv += dest.deriv
    T = eltype(dest.deriv)
    dest.deriv .= zero(T)

    return nothing

end

copyto_one_element() = nothing

function Base.copyto!(dest::ReverseDiff.TrackedArray, di::Integer, src::ReverseDiff.TrackedArray, si::Integer)

    dest_val_star = deepcopy(dest.value[di])
    dest.value[di] = src.value[si]
    tp = ReverseDiff.tape(dest, src)
    dest.tape = tp
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        copyto_one_element,
                        (dest, di, src, si),
                        dest_val_star)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(copyto_one_element)})

    dest, di, src, si = instruction.input
    dest_val_star = instruction.output

    dest.value[di] = dest_val_star
    src.deriv[si] += dest.deriv[di]
    T = eltype(dest.deriv)
    dest.deriv[di] = zero(T)

    return nothing

end
=#

function ReverseDiff.value(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}
    return ParticleField{_V, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}(
                        pfield.maxparticles,
                        #view(ReverseDiff.value(pfield.particles)), # hopefully this view stops allocations. I might nee to apply the view to pfield.particles directly, instead.
                        ReverseDiff.value.(pfield.particles),
                        pfield.formulation,
                        pfield.viscous,
                        pfield.np,
                        pfield.nt,
                        pfield.t,
                        pfield.kernel,
                        pfield.UJ,
                        pfield.Uinf,
                        pfield.SFS,
                        pfield.integration,
                        pfield.transposed,
                        pfield.relaxation,
                        pfield.fmm,
                        pfield.useGPU
                        )
end

function ReverseDiff.deriv(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}
    return ParticleField{D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}(
                        pfield.maxparticles,
                        #view(ReverseDiff.deriv.(pfield.particles), :), # hopefully this view stops allocations. I might nee to apply the view to pfield.particles directly, instead.
                        ReverseDiff.deriv.(pfield.particles),
                        pfield.formulation,
                        pfield.viscous,
                        pfield.np,
                        pfield.nt,
                        pfield.t,
                        pfield.kernel,
                        pfield.UJ,
                        pfield.Uinf,
                        pfield.SFS,
                        pfield.integration,
                        pfield.transposed,
                        pfield.relaxation,
                        pfield.fmm,
                        pfield.useGPU
                        )
end

# In-place function that breaks without an explicit rule
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch, target_buffer::ReverseDiff.TrackedArray, i_buffer)
    
    tp = ReverseDiff.tape(target_system, target_buffer)
    ustar = ReverseDiff.value.(target_system.particles[U_INDEX, i_target])
    jstar = ReverseDiff.value.(target_system.particles[J_INDEX, i_target])
    u = fmm.get_velocity(target_buffer.value, i_buffer)
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value += u[i]
    end
    j = fmm.get_velocity_gradient(target_buffer.value, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target].value += j[i]
    end
    
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.buffer_to_target_system!,
                        (target_system, i_target, derivatives_switch, target_buffer, i_buffer),
                        nothing,
                        (ustar, jstar))
    return nothing
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer = instruction.input
    ustar, jstar = instruction.cache
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value = ustar[i]
        target_buffer.deriv[i+4, i_buffer] += target_system.particles[U_INDEX[i], i_target].deriv
    end
    for i=1:9
        target_system.particles[J_INDEX[i], i_target].value = jstar[i]
        target_buffer.deriv[i+7, i_buffer] += target_system.particles[J_INDEX[i], i_target].deriv
    end
    return nothing

end

function onestep!(states, states_prev, t, t_prev, xd, xci, p)
    
    check_derivs(states; label="states at time $t")
    check_derivs(states_prev; label="states_prev at time $t")
    check_derivs(xd; label="xd at time $t")
    check_derivs(xci; label="xci at time $t")

    static_particles_function, runtime_function, verbose_nsteps, v_lvl, save_pfield, save_path, nsteps_save, vprintln, nsteps, dt, custom_UJ, pfield = p
    
    check_derivs(pfield.particles; label="pfield after passing derivatives back")
    check_derivs(states_prev[1:end-1]; label="states after passing derivatives back")
    map_flat_states_to_pfield!(pfield, states_prev)
    check_derivs(pfield.particles; label="pfield before passing derivatives back")
    check_derivs(states_prev[1:end-1]; label="states before passing derivatives back")
    # no use for xci for now

    i = pfield.nt
    if i%verbose_nsteps==0
        vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
    end

    # Relaxation step
    relax = pfield.relaxation != relaxation_none &&
            pfield.relaxation.nsteps_relax >= 1 &&
            i>0 && (i%pfield.relaxation.nsteps_relax == 0)

    org_np = get_np(pfield)

    # Add static particles
    remove = static_particles_function(pfield, pfield.t, dt)

    # Step in time solving governing equations
    #check_derivs(pfield.particles)
    nextstep(pfield, dt; relax=relax, custom_UJ=custom_UJ)
    #@show pfield.particles[1,1]
    #check_derivs(pfield.particles)

    # Remove static particles (assumes particles remained sorted)
    if remove===nothing || remove
        for pi in get_np(pfield):-1:(org_np+1)
            remove_particle(pfield, pi)
        end
    end

    # Calls user-defined runtime function
    breakflag = runtime_function(pfield, t, dt;
                vprintln= (str)-> i%verbose_nsteps==0 ?
                    vprintln(str, v_lvl+2) : nothing, xd=xd, xci=xci)

    # Save particle field
    if save_pfield && save_path!=nothing && (i%nsteps_save==0 || i==nsteps || breakflag) && eltype(pfield) <: AbstractFloat
        overwrite_time = save_time ? nothing : pfield.nt
        save(pfield, run_name; path=save_path, add_num=true,
                overwrite_time=overwrite_time)
    end
    check_derivs(pfield.particles; label="pfield after passing derivatives back")
    check_derivs(states[1:end-1]; label="states after passing derivatives back")
    map_pfield_to_flat_states!(states, pfield)
    check_derivs(pfield.particles; label="pfield before passing derivatives back")
    check_derivs(states[1:end-1]; label="states before passing derivatives back")
    return nothing

end

function initialize(t0, xd, xc0, p)

    static_particles_function, runtime_function, verbose_nsteps, v_lvl, save_pfield, save_path, nsteps_save, vprintln, nsteps, dt, custom_UJ, pfield_cache = p
    pfield_cache.t = t0
    i = 0
    if i%verbose_nsteps==0
        vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield_cache))", v_lvl+1)
    end

    # Calls user-defined runtime function
    breakflag = runtime_function(pfield_cache, t0, dt;
                                 vprintln= (str)-> vprintln(str, v_lvl+2), xd=xd, xci=xc0)

    # Save particle field
    if save_pfield && save_path!==nothing && eltype(pfield_cache) <: AbstractFloat
        overwrite_time = save_time ? nothing : pfield_cache.nt
        save(pfield_cache, run_name; path=save_path, add_num=true,
                overwrite_time=overwrite_time)
    end
    #return cat(reshape(pfield_cache.particles, length(pfield_cache.particles)), pfield_cache.np; dims=1)
    states = zeros(eltype(pfield_cache), length(pfield_cache.particles) + 1)
    tp = ReverseDiff.tape(pfield_cache.particles)
    for idx in CartesianIndices(states)
        states[idx] = ReverseDiff.track(0.0, tp)
    end
    check_derivs(pfield_cache.particles; label="pfield after passing derivatives back")
    check_derivs(states[1:end-1]; label="states after passing derivatives back")
    map_pfield_to_flat_states!(states, pfield_cache)
    check_derivs(pfield_cache.particles; label="pfield before passing derivatives back")
    check_derivs(states[1:end-1]; label="states before passing derivatives back")
    return states

end

function map_flat_states_to_pfield!(pfield, states)

    for i=1:pfield.np
        for j=1:43
            pfield[j,i] = states[j + (i-1)*43]
        end
    end
    pfield.np = Int(states[end])
    return nothing

end

function map_flat_states_to_pfield!(pfield, states::AbstractArray{<:ReverseDiff.TrackedReal})

    #tp = ReverseDiff.tape(pfield)
    tp = ReverseDiff.tape(states)
    states_value = deepcopy(ReverseDiff.value.(states)) # not sure if I need this.
    for i=1:pfield.np
        for j=1:43
            pfield.particles[j,i] = ReverseDiff.track(states[j + (i-1)*43], tp)
        end
    end
    pfield.np = Int(ReverseDiff.value(states[end]))
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        map_flat_states_to_pfield!,
                        (pfield, states),
                        nothing,
                        states_value
                        )

    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_flat_states_to_pfield!)})

    pfield, states = instruction.input
    states_value = instruction.cache
    for idx in CartesianIndices(states)
        states[idx].value = states_value[idx] # not sure if needed.
    end
    
    for i=1:pfield.np
        for j=1:43
            states[j + (i-1)*43].deriv = pfield.particles[j,i].deriv
            pfield.particles[j,i].deriv = 0.0
        end
    end
    pfield.np = Int(ReverseDiff.value(states[end]))
    return nothing

end

function map_pfield_to_flat_states!(states, pfield)

    for i=1:pfield.np
        for j=1:43
            states[j + (i-1)*43] = pfield[j,i]
        end
    end
    states[end] = pfield.np
    return nothing

end

function map_pfield_to_flat_states!(states::AbstractArray{<:ReverseDiff.TrackedReal}, pfield)

    tp = ReverseDiff.tape(pfield)
    pfield_value = deepcopy(ReverseDiff.value.(pfield.particles))
    for i=1:pfield.np
        for j=1:43
            states[j + (i-1)*43] = ReverseDiff.track(pfield.particles[j,i].value,tp)
        end
    end
    T = eltype(pfield.particles[1].value) # get the appropriate floating point type
    states[end] = ReverseDiff.track(T(pfield.np), tp)

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        map_pfield_to_flat_states!,
                        (states, pfield),
                        nothing,
                        (pfield_value, pfield.np)
                        )

    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_pfield_to_flat_states!)})

    states, pfield = instruction.input
    pfield_value, np = instruction.cache

    for idx in CartesianIndices(pfield.particles)
        pfield.particles[idx].value = pfield_value[idx] # not sure if needed.
    end
    pfield.np = np
    
    for i=1:pfield.np
        for j=1:43
            pfield.particles[j,i].deriv = states[j + (i-1)*43].deriv
            states[j + (i-1)*43].deriv = 0.0
        end
    end
    states[end].value = np
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
                        A,
                        Astar)

    return A
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(add!_trackedreal)})

    A, B = instruction.input
    A.value = instruction.cache
    B.deriv += A.deriv
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