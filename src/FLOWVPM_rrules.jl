# Levi-Civita tensor contractions for convenience. This shows up in cross products.
# ϵ with two vectors -> vector, so need one scalar index
# ϵ with one vector -> matrix, so need two scalar indices
# ϵ with one matrix -> vector, so need one scalar index
ϵ(a,x::Vector,y::Vector) = (a == 1) ? (x[2]*y[3] - x[3]*y[2]) : ((a == 2) ? (x[3]*y[1] - x[1]*y[3]) : ((a == 3) ? (x[1]*y[2] - x[2]*y[1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")))
ϵ(a,b::Number,y::Vector) = (a == b) ? zero(eltype(y)) : ((mod(b-a,3) == 1) ? y[mod(b,3)+1] : ((mod(a-b,3) == 1) ? -y[mod(b-2,3)+1] : error("attempted to evaluate Levi-Civita symbol at out-of-bounds indices $(a) and $(b)!")))
ϵ(a,x::Vector,c::Number) = -1 .*ϵ(a,c,x)
ϵ(a,x::TM) where {TM <: AbstractArray} = (a == 1) ? (x[2,3] - x[3,2]) : (a == 2) ? (x[3,1]-x[1,3]) : (a == 3) ? (x[1,2]-x[2,1]) : error("attempted to evaluate Levi-Civita symbol at out-of-bounds index $(a)!")
ϵ(a,b::Number, c::Number) = (a == b || b == c || c == a) ? 0 : (mod(b-a,3) == 1 ? 1 : -1) # no error checks in this implementation, since that would significantly increase the cost of it

#=const ϵ = begin
    
    ϵ_out = zeros(Int, 3,3,3)
    for i=1:3
        for j=1:3
            for k=1:3
                ϵ_out[i,j,k] = ϵ(i,j,k)
            end
        end
    end
    return ϵ_out

end=#

using ChainRulesCore

#############

# Normally I would just use the @grad_from_chainrules macro for the pairwise particle interaction. However, this will not work:
#    - The current form of the function is in-place. I've written up a new macro that handles this, but it would break
#    - The real issue is that the source system itself is an input. While the function evaluation and pullback can be evaluated,
#           there are several ReverseDiff functions that are not defined for the source system:
#       - track
#       - tape
#       - value and value!
#       - deriv and _add_to_deriv
#       - pull_value!
# Hm... the more I look at this, the more it seems like the correct solution is to implement these functions for the system type (which is ParticleField).
# In theory, this just means broadcasting the ReverseDiff operations from the struct to the struct contents.
#=
function ReverseDiff.track(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}, ::Type{D}, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) where {R<:Real, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS, D}
    return ParticleField{ReverseDiff.TrackedReal(R, R, tp), F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, TGPU, TEPS}(
                        pfield.maxparticles,
                        ReverseDiff.track(pfield.particles, R, tp),
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
                        pfield.useGPU,
                        ReverseDiff.track(pfield.M, R, tp),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
                        )
end


function ReverseDiff.track(pfield::ParticleField{ReverseDiff.TrackedReal{_R, _D, Nothing}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) where {_R, _D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    return ParticleField{ReverseDiff.TrackedReal(_R, _D, tp), F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, TGPU, TEPS}(
                        pfield.maxparticles,
                        ReverseDiff.track(pfield.particles, _D, tp),
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
                        pfield.useGPU,
                        ReverseDiff.track(pfield.M, _D, tp),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
                        )
end

function ReverseDiff.track!(pfield::ParticleField, pfield_val::ParticleField)
    ReverseDiff.value!(pfield, pfield_val.particles)
    ReverseDiff.unseed!(pfield)
    return pfield
end


function ReverseDiff.track!(pfield::ParticleField, pfield_val::AbstractArray)
    ReverseDiff.value!.(pfield, pfield_val)
    ReverseDiff.unseed!(pfield)
    return pfield
end

function ReverseDiff.track!(pfield::ParticleField{AbstractArray{ReverseDiff.TrackedReal{D, D, Nothing}}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}, pfield_val::ParticleField{D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}, tp::ReverseDiff.InstructionTape) where {D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    for i in eachindex(pfield.particles)
        pfield.particles[i] = track(pfield_val[i], D, tp)
    end
    return pfield

end

ReverseDiff.tape(pfield::ParticleField) = ReverseDiff.tape(pfield.particles)

#ReverseDiff.value(pfield::ParticleField) = ReverseDiff.value(pfield.particles) # needs to return a pfield to work as expected with reversediff, so this simple implementation is wrong.
function ReverseDiff.value(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    return ParticleField{_V, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}(
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
                        pfield.useGPU,
                        #view(ReverseDiff.value(pfield.M)),
                        ReverseDiff.value.(pfield.M),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
                        )
end
ReverseDiff.value!(pfield::ParticleField, val) = ReverseDiff.value!.(pfield.particles, val)
#ReverseDiff.deriv(pfield::ParticleField) = ReverseDiff.deriv(pfield.particles) # needs to return a pfield to work as expected with reversediff, so this simple implementation is wrong.
function ReverseDiff.deriv(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    return ParticleField{D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}(
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
                        pfield.useGPU,
                        #view(ReverseDiff.deriv.(pfield.M), :),
                        ReverseDiff.deriv.(pfield.M),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
                        )
end
ReverseDiff._add_to_deriv!(pfield::ParticleField, deriv) = ReverseDiff._add_to_deriv!(pfield.particles, deriv)
ReverseDiff.pull_value!(pfield::ParticleField) = ReverseDiff.pull_value!(pfield.particles)
ReverseDiff.unseed!(pfield::ParticleField) = ReverseDiff.unseed!(pfield.particles)

# next I need to handle pullbacks for in-place function calls. here's a modified macro for that.

macro inplace_grad_from_chainrules(fcall)
    Meta.isexpr(fcall, :call) && length(fcall.args) >= 2 || # meta stuff I do not want to touch
        error("`inplace_grad_from_chainrules` has to be applied to a function signature")
    f = esc(fcall.args[1])
    xs = map(fcall.args[2:end]) do x
        if Meta.isexpr(x, :(::))
            if length(x.args) == 1 # ::T without var name
                return :($(gensym())::$(esc(x.args[1])))
            else # x::T
                @assert length(x.args) == 2
                return :($(x.args[1])::$(esc(x.args[2])))
            end
        else
            return x
        end
    end
    args_l, args_r, args_track, args_fixed, arg_types, kwargs = ReverseDiff._make_fwd_args(f, xs) # should be fine as is
    return quote
        $f($(args_l...)) = ReverseDiff.track($(args_r...))
        function ReverseDiff.track($(args_track...))
            args = ($(args_fixed...),)
            tp = ReverseDiff.tape(args...)
            
            _output, back = ChainRulesCore.rrule($f, map(ReverseDiff.value, args)...; $kwargs...)
            output = ReverseDiff.track(_output, tp)
            ReverseDiff.value!.(args[1], _output)
            closure(cls_args...; cls_kwargs...) = ChainRulesCore.rrule($f, map(ReverseDiff.value, cls_args)...; cls_kwargs...)
            ReverseDiff.record!(
                tp,
                ReverseDiff.SpecialInstruction,
                $f,
                args,
                output,
                (back, closure, $kwargs),
            )
            return nothing
        end

        @noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}})
      
            output = instruction.output
            input = instruction.input
            for i=1:length(output)
                ReverseDiff.deriv!(output[i], i)
                @show ReverseDiff.deriv(output[i])
            end
            @show typeof(output)
            @show ReverseDiff.deriv(output)
            @show ReverseDiff.deriv.(input[1])
            @show ReverseDiff.deriv.(input[4].particles)
            @show ReverseDiff.deriv.(input[5])
            ReverseDiff.deriv!(output, ReverseDiff.deriv.(input[1])) # it seems inefficient to shuffle around a derivative vector like this, but all the other approaches I've tried don't work.
            ReverseDiff.unseed!(input[1])
            back = instruction.cache[1]
            back_output = back(ReverseDiff.deriv(output))
            input_derivs = back_output[2:end]
            @assert input_derivs isa Tuple
            ReverseDiff._add_to_deriv!.(input, input_derivs)
            ReverseDiff.unseed!(output)
            return nothing

        end

        # need to check to see if this still works correctly.
        @noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}})
            error("forward exec for in-place functions not implemented yet!")
            output, input = instruction.output, instruction.input
            ReverseDiff.pull_value!.(input)
            pullback = instruction.cache[2]
            kwargs = instruction.cache[3]
            pullback(input...; kwargs...)[1]
            out_value = pullback(input...; kwargs...)[1]
            ReverseDiff.value!(output, input[1])
            return nothing
        end
    end
end
=#
# alright, now I can write the actual pullback. I can just use the same math as before, which is quite convenient.
ReverseDiff.tape(pfield::ParticleField) = ReverseDiff.tape(pfield.particles)
#=

function fmm.direct!(target_buffer, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS}

    for i_source_particle in source_index

        # gamma_x, gamma_y, gamma_z = get_Gamma(source_particle)
        gamma_x, gamma_y, gamma_z = fmm.get_strength(source_buffer, source_system, i_source_particle)
        # source_x, source_y, source_z = get_X(source_particle)
        source_x, source_y, source_z = fmm.get_position(source_buffer, i_source_particle)
        # sigma = get_sigma(source_particle)[]
        sigma = source_buffer[8, i_source_particle]

        for j_target in target_index

            target_x, target_y, target_z = fmm.get_position(target_buffer, j_target)
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz

            if !iszero(r2)
                r = sqrt(r2)

                # Regularizing function and deriv
                g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)

                # K × Γp
                r3inv = one(r) / (r2 * r)
                crss1 = -const4 * r3inv * ( dy*gamma_z - dz*gamma_y )
                crss2 = -const4 * r3inv * ( dz*gamma_x - dx*gamma_z )
                crss3 = -const4 * r3inv * ( dx*gamma_y - dy*gamma_x )

                if VS
                    # U = ∑g_σ(x-xp) * K(x-xp) × Γp
                    Ux = g_sgm * crss1
                    Uy = g_sgm * crss2
                    Uz = g_sgm * crss3
                    # get_U(target_particle) .+= Ux, Uy, Uz
                    Ux0, Uy0, Uz0 = fmm.get_velocity(target_buffer, j_target)

                    val = SVector{3}(Ux, Uy, Uz)
                    fmm.set_velocity!(target_buffer, j_target, val)
                end

                if GS
                    # ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
                    # ∂u∂xj(x) = ∑p[(Δxj∂gσ∂r/(σr) − 3Δxjgσ/r^2) K(Δx)×Γp
                    aux = dg_sgmdr/(sigma*r) - 3*g_sgm / r2
                    # ∂u∂xj(x) = −∑gσ/(4πr^3) δij×Γp
                    # Adds the Kronecker delta term
                    aux2 = -const4 * g_sgm * r3inv
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
                    # @show aux, aux2, crss1, crss2, crss3, dx, dy, dz
                    # @show du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3

                    val = SMatrix{3,3}(du1x1, du2x1, du3x1, du1x2, du2x2, du3x2, du1x3, du2x3, du3x3)
                    fmm.set_velocity_gradient!(target_buffer, j_target, val)
                end
            end
        end
    end

    return nothing
end

=#

using ForwardDiff
const c4 = 1/(4*pi)

function ChainRulesCore.rrule(::typeof(fmm.direct!), target_buffer, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS}
    
    #println("started rrule forward call")
    #error()
    function direct_pullback(target_buffer_bar)

        T = eltype(target_buffer_bar)
        # allocate cotangents of inputs
        target_buffer_bar = zeros(T, size(target_buffer))
        source_system_bar = zeros(T, size(source_system.particles))
        source_buffer_bar = zeros(T, size(source_buffer))
        # pre-allocate a bunch of buffers to avoid ~2500*i*j allocations each time this is called.
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
        #u_j_star_bar = zeros(T,3)
        #du_j_star_bar = zeros(T,3,3)
        for i in source_index

            Gamma .= fmm.get_strength(source_buffer, source_system, i)
            Gamma_i_bar .= zero(T)
            x_i .= fmm.get_position(source_buffer, i)
            x_i_bar .= zero(T)
            sigma = source_buffer[8, i]
            sigma_i_bar = zero(T)
            for j in target_index
                # calculate r, dx, and check if particles actually interact
                x_j .= fmm.get_position(target_buffer, j)
                x_j_bar .= zero(T)
                @. dx = x_j - x_i # todo: make sure this is elementwise
                r2 = dx[1]*dx[1] + dx[2]*dx[2] + dx[3]*dx[3]
                if r2 > 0
                    r = sqrt(r2)
                    g_sgm, dg_sgmdr = source_system.kernel.g_dgdr(r/sigma)
                    ddg_sgmdr = ForwardDiff.derivative(source_system.kernel.dgdr,r/sigma) # derivative of g' at r/sigma

                    r3inv = one(r) / (r2 * r)
                    r4inv = r3inv/r
                    crss[1] = -const4 * r3inv * ( dx[2]*Gamma[3] - dx[3]*Gamma[2] )
                    crss[2] = -const4 * r3inv * ( dx[3]*Gamma[1] - dx[1]*Gamma[3] )
                    crss[3] = -const4 * r3inv * ( dx[1]*Gamma[2] - dx[2]*Gamma[1] )
                    if VS
                        aux1 = dg_sgmdr*r/sigma
                        aux2 = aux1/sigma
                        aux3 = g_sgm/r2
                        aux4 = g_sgm*c4*r3inv
                        u_j_bar .= fmm.get_velocity(target_buffer_bar, j)
                        for a=1:3
                            temp = 2*u_j_bar[a]*aux1*dx[a]*crss[a]
                            x_j_bar[a] += temp
                            x_i_bar[a] -= temp
                            sigma_i_bar -= u_j_bar[a]*aux2*crss[a]
                            for b=1:3
                                temp = -u_j_bar[a]*aux3*6*crss[a]*dx[b]
                                x_j_bar[b] += temp
                                x_i_bar[b] -= temp
                                for c=1:3
                                    temp = u_j_bar[a]*aux4*ϵ(a,b,c)*Gamma[c]
                                    x_j_bar[b] += temp
                                    x_i_bar[b] -= temp
                                    Gamma_i_bar[c] -= u_j_bar[a]*aux4*ϵ(a,b,c)*dx[b]
                                end
                            end
                        end
                    end
                    if GS
                        # calculate assorted coefficients that only depend on i and j
                        A = dg_sgmdr/(sigma*r) - 3*g_sgm/r2
                        B = -c4*g_sgm*r3inv
                        C = 2*ddg_sgmdr/sigma^2 - 8*dg_sgmdr*r3inv/sigma - 3*dg_sgmdr/(r*sigma) + 22*g_sgm*r4inv
                        D = -c4*A*r3inv
                        E = -2*c4*dg_sgmdr/sigma/r2 + 6*g_sgm*r3inv/r2
                        F = -ddg_sgmdr/sigma^3 + 2*dg_sgmdr/(r*sigma^2)
                        G = c4*dg_sgmdr/(r2*sigma^2)
                        du_j_bar .= fmm.get_velocity_gradient(target_buffer_bar, j)
                        for d=1:3
                            for a=1:3
                                for b=1:3
                                    #du_j_star_bar[a,b] = du_j_bar[a,b]
                                    temp = du_j_bar[a,b]*dx[d]*crss[a]*dx[b]*C
                                    for c=1:3
                                        temp += du_j_bar[a,b]*Gamma[c]*(D*ϵ(a,d,c)*dx[b] + E*dx[a]*ϵ(a,b,c))
                                        Gamma_i_bar[c] += du_j_bar[a,b] * (D*ϵ(a,d,c)*dx[d]*dx[b])
                                    end
                                    x_j_bar[d] += temp
                                    x_i_bar[d] -= temp
                                end
                            end
                        end
                        for a=1:3
                            for b=1:3
                                temp = du_j_bar[a,b]*A*crss[a]
                                x_j_bar[b] += temp
                                x_i_bar[b] -= temp
                                sigma_i_bar += du_j_bar[a,b]*F*crss[a]*dx[b]
                                for c=1:3
                                    sigma_i_bar += du_j_bar[a,b]*G*ϵ(a,b,c)*Gamma[c]
                                    Gamma_i_bar[c] -= du_j_bar[a,b]*B*ϵ(a,b,c)
                                end
                            end
                        end
                    end
                end
                # cotangents of original values of velocity/velocity gradient are trivial
                if VS
                    fmm.set_velocity!(target_buffer_bar, j, u_j_bar)
                    #target_buffer_bar[U_INDEX, j] .+= u_j_bar
                end
                if GS
                    fmm.set_velocity_gradient!(target_buffer_bar, j, du_j_bar)
                    #target_buffer_bar[J_INDEX, j] .+= du_j_bar
                end

                target_buffer_bar[1:3, j] .+= x_j_bar
                
                source_system_bar[GAMMA_INDEX, i] .+= Gamma_i_bar # this is really inefficient. I should really write to the buffer instead and never return a copy of the system's particles (which have 16*n entries, 13*n of which are zero).

                source_buffer_bar[8, i] += sigma_i_bar
                source_buffer_bar[1:3, i] .+= x_i_bar
            end
        end
        return NoTangent(), target_buffer_bar, NoTangent(), NoTangent(), source_system_bar, source_buffer_bar, NoTangent()
        #return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()

    end
    fmm.direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, source_index)
    return target_buffer, direct_pullback

end

#@inplace_grad_from_chainrules fmm.direct!(target_buffer::AbstractArray{<:ReverseDiff.TrackedReal}, target_index, derivatives_switch::fmm.DerivativesSwitch, source_system::ParticleField, source_buffer::AbstractArray{<:ReverseDiff.TrackedReal}, source_index)

#=
@inplace_grad_from_chainrules fmm.direct!(target_buffer::ReverseDiff.TrackedArray,
                                          target_index, derivatives_switch::fmm.DerivativesSwitch,
                                          source_system::ParticleField{<:ReverseDiff.TrackedReal, Formulation, ViscousScheme, Function, SubFilterScale, Kernel, Function, Function, Function, Bool, Number},
                                          source_buffer::ReverseDiff.TrackedArray,
                                          source_index)
=#

function fmm.direct!(target_buffer::AbstractArray{<:ReverseDiff.TrackedReal{V, D, O}}, target_index, derivatives_switch::fmm.DerivativesSwitch{PS,VS,GS}, source_system::ParticleField, source_buffer, source_index) where {PS,VS,GS,V,D,O}
    
    target_buffer_val = ReverseDiff.value.(target_buffer)
    target_buffer_val_star = deepcopy(target_buffer_val) # since this is an in-place function, we need to save the overwritten input.
    source_system_val = ReverseDiff.value(source_system)
    source_buffer_val = ReverseDiff.value.(source_buffer)
    tp = ReverseDiff.tape(source_system)
    fmm.direct!(target_buffer_val, target_index, derivatives_switch, source_system_val, source_buffer_val, source_index)
    for idx in CartesianIndices(target_buffer[:, target_index])
        target_buffer[idx].value = target_buffer_val[idx]
        #target_buffer[idx] = ReverseDiff.track(target_buffer_val[idx],tp)
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
    
    #=target_buffer_deriv = ReverseDiff.deriv.(target_buffer)
    source_system_deriv = ReverseDiff.deriv(source_system)
    source_buffer_deriv = ReverseDiff.deriv.(source_buffer)
    target_buffer_val = ReverseDiff.value.(target_buffer)
    source_system_val = ReverseDiff.value(source_system)
    source_buffer_val = ReverseDiff.value.(source_buffer)=#

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
                    #=for a=1:3
                        for b=1:3
                            for c=1:3
                                x_j_bar[b] -= ϵ(b,a,c)*u_j_bar[a]*Gamma[c] #ϵ(b,u_j_bar,Gamma)
                                x_i_bar[b] += ϵ(b,a,c)*u_j_bar[a]*Gamma[c] #ϵ(b,u_j_bar,Gamma)
                                Gamma_i_bar[c] -= ϵ(c,b,a)*dx[b]*u_j_bar[a] #ϵ(c, dx, u_j_bar)
                            end
                        end
                    end=#
                    
                    
                    A = -const4*g_sgm/r^3
                    if i == 1 && j == 2
                        #@show A
                    end
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
                    #=
                    for a=1:3
                        for b=1:3
                            for c=1:3
                                for d=1:3
                                    x_j_bar[a] += du_j_bar[a,b]*ϵ(b,c,d)*dx[c]*Gamma[d]
                                    x_i_bar[a] -= du_j_bar[a,b]*ϵ(b,c,d)*dx[c]*Gamma[d]
                                    x_j_bar[c] += du_j_bar[a,b]*dx[a]*ϵ(b,c,d)*Gamma[d]
                                    x_i_bar[c] -= du_j_bar[a,b]*dx[a]*ϵ(b,c,d)*Gamma[d]
                                    Gamma_i_bar[d] += du_j_bar[a,b]*dx[a]*ϵ(b,c,d)*dx[c]
                                end
                                Gamma_i_bar[c] += ϵ(a,b,c)*du_j_bar[a,b]
                            end
                        end
                    end
                    =#
                    
                    α = dg_sgmdr/(sigma*r) - 3*g_sgm/r^2
                    β = -const4*g_sgm/r^3
                    γ = (ddg_sgmdr/sigma^2 - 7*dg_sgmdr/(r*sigma) + 15*g_sgm/r^2)
                    for a = 1:3
                        for b=1:3
                            sigma_temp = 0.0
                            for c=1:3
                                sigma_temp += const4*dg_sgmdr*ϵ(a,b,c)*Gamma[c]/r^2
                                #gamma_temp = -β*ϵ(a,b,c)*du_j_bar[c, b]
                                gamma_temp = -β*ϵ(a,b,c)*du_j_bar[b, c] # swap b/c in Jbar
                                xyz_temp = 0.0
                                for d=1:3
                                    xyz_temp += (dx[b]*ϵ(c,a,d) + dx[a]*ϵ(c,b,d))*Gamma[d]
                                    #gamma_temp += α*const4/r^3*ϵ(a,c,d)*dx[c]*dx[b]*du_j_bar[d, b]
                                    gamma_temp += α*const4/r^3*ϵ(a,c,d)*dx[c]*dx[b]*du_j_bar[b, d] # swap b/d in Jbar
                                    #gamma_temp += α*const4/r^3*ϵ(a,d,c)*dx[c]*dx[b]*du_j_bar[b, d] # is this index permutation correct? nope.
                                end
                                xyz_temp *= -α*const4/r
                                xyz_temp += γ*crss[c]*dx[b]*dx[a]
                                #xyz_temp *= du_j_bar[c, b]/r^2
                                xyz_temp *= du_j_bar[b, c]/r^2 # swap b/c in Jbar
                                Gamma_i_bar[a] += gamma_temp

                                x_j_bar[a] += xyz_temp
                                x_i_bar[a] -= xyz_temp
                            end
                            sigma_temp += (-ddg_sgmdr/sigma + 2*dg_sgmdr/r)*crss[a]*dx[b]
                            sigma_i_bar += du_j_bar[b, a]/sigma^2*sigma_temp
                            #x_j_bar[a] += du_j_bar[b, a]*α*crss[b]
                            #x_i_bar[a] -= du_j_bar[b, a]*α*crss[b]
                            x_j_bar[a] += du_j_bar[a, b]*α*crss[b] # swap a/b in Jbar
                            x_i_bar[a] -= du_j_bar[a, b]*α*crss[b] # swap a/b in Jbar
                        end
                    end
                    
                end
            end
            for a=1:3
                ReverseDiff._add_to_deriv!(target_buffer[a, j], x_j_bar[a])
            end

        end
        
        for a=1:3
            #ReverseDiff._add_to_deriv!(source_system.particles[GAMMA_INDEX[a], i], Gamma_i_bar[a])
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
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.ε_tol)
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
    T = eltype(buffer[1].deriv)
    
    # manual unseed - we do not want to keep derivatives in the buffer after we map derivatives back to the particle field.
    for idx in 1:8
        buffer[idx, i_buffer].deriv = zero(T)
    end
    
    return nothing

end

function fmm.source_system_to_buffer_pullback!(buffer, i_buffer, system::ParticleField, i_body)

    #=σ = system.particles[SIGMA_INDEX, i_body]
    Γx, Γy, Γz = view(system.particles, GAMMA_INDEX, i_body)
    Γ = sqrt(Γx*Γx + Γy*Γy + Γz*Γz)
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.ε_tol)
    buffer[1:3, i_buffer] .= view(system.particles, X_INDEX, i_body)
    buffer[4, i_buffer] = ρ_σ * σ
    buffer[5:7, i_buffer] .= view(system.particles, GAMMA_INDEX, i_body)
    buffer[8, i_buffer] = σ=#

    # def y = buffer[4, i_buffer]
    # y = ρ_σ * σ
    # r(ρ_σ, σ, ω) = 0
    # ω = |Γ|
    # Apply implicit differentiation:
    # Γ_i_bar += -ybar/(dr/dρ_σ) * dr/dω * Γ_i/ω
    # σ_bar += ybar(-1/dr/dρ_σ) * dr/dσ + ρ_σ)

    # finally, we need various derivatives of r... so just apply ForwardDiff
    σ = system.particles[SIGMA_INDEX, i_body].value
    #@show σ
    ε = system.fmm.ε_tol
    #ρ_σ = buffer[4, i_buffer].value/σ
    #Γx, Γy, Γz = view(system.particles, GAMMA_INDEX, i_body)
    Γx = system.particles[GAMMA_INDEX[1], i_body].value
    Γy = system.particles[GAMMA_INDEX[2], i_body].value
    Γz = system.particles[GAMMA_INDEX[3], i_body].value
    Γ = sqrt(Γx^2 + Γy^2 + Γz^2)
    ρ_σ = solve_ρ_over_σ(σ, Γ, system.fmm.ε_tol)
    
    #@show ReverseDiff.deriv.(system.particles[1:8, i_body])

    #view(system_deriv.particles, X_INDEX, i_body) .+= buffer_deriv[1:3, i_buffer]
    for i=1:3
        ReverseDiff._add_to_deriv!(system.particles[X_INDEX[i], i_body], buffer[i, i_buffer].deriv)
        ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[i], i_body], buffer[i+4, i_buffer].deriv)
    end
    #view(system_deriv.particles, GAMMA_INDEX, i_body) .+= buffer_deriv[5:7, i_buffer]
    #@show buffer_deriv[1:3, i_buffer] view(system_deriv.particles, X_INDEX, i_body)
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[8, i_buffer].deriv)
    #system_deriv.particles[SIGMA_INDEX, i_body] += buffer_deriv[8, i_buffer]

    #@show σ Γ ε ρ_σ
    dr_dρ_σ = ForwardDiff.derivative((_ρ_σ)->residual(_ρ_σ, σ, Γ, ε), ρ_σ)
    dr_dσ = ForwardDiff.derivative((_σ)->residual(ρ_σ, _σ, Γ, ε), σ)
    dr_dω = ForwardDiff.derivative((_Γ)->residual(ρ_σ, σ, _Γ, ε), Γ)
    #=
    Γbar = view(system_deriv.particles, GAMMA_INDEX, i_body)
    Γbar[1] += -buffer_deriv[4, i_buffer]/dr_dρ_σ * dr_dω * Γx/Γ
    Γbar[2] += -buffer_deriv[4, i_buffer]/dr_dρ_σ * dr_dω * Γy/Γ
    Γbar[3] += -buffer_deriv[4, i_buffer]/dr_dρ_σ * dr_dω * Γz/Γ
    =#
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[1], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γx/Γ)
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[2], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γy/Γ)
    ReverseDiff._add_to_deriv!(system.particles[GAMMA_INDEX[3], i_body], -buffer[4, i_buffer].deriv/dr_dρ_σ*dr_dω*Γz/Γ)
    #@show sum(ReverseDiff.deriv.(system.particles[GAMMA_INDEX, i_body]))
    #system_deriv.particles[SIGMA_INDEX, i_body] += buffer_deriv[4, i_buffer]*(-1/dr_dρ_σ * dr_dσ + ρ_σ)
    ReverseDiff._add_to_deriv!(system.particles[SIGMA_INDEX, i_body], buffer[4, i_buffer].deriv*(-1/dr_dρ_σ * dr_dσ + ρ_σ))
    #@show system_deriv.particles[GAMMA_INDEX, i_body]
    #@show ReverseDiff.value.(system.particles[GAMMA_INDEX, i_body])
    #@show dr_dρ_σ dr_dσ dr_dω ρ_σ
    #@show sum(ReverseDiff.deriv.(system.particles[GAMMA_INDEX, i_body]))
    T = eltype(buffer.deriv)
    #@show ReverseDiff.deriv.(system.particles[1:8, i_body]) buffer.deriv[:, i_buffer]
    buffer.deriv[:, i_buffer] .= zero(T) # manual unseed - we do not want to keep derivatives in the buffer after we map derivatives back to the particle field.
    
    return nothing

end

function fmm.get_position_pullback!(system::ParticleField, i, buffer)
    # ReverseDiff._add_to_deriv!.(get_position(system, sort_index[i_body]), buffer.value[1:3, i_body])
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
function add_particle(pfield::ParticleField{ReverseDiff.TrackedReal{R, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU, TEPS}, X, Gamma, sigma; vol=0, circulation=1, C=0, static=false) where {R, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU, TEPS}

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

    #@show pfield.np size(pfield.particles)
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
    #@show pfield.np size(pfield.particles)
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

# I need a nice implementation of copyto! for tracked arrays.

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

#=
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch, target_buffer, i_buffer)
    target_system.particles[U_INDEX, i_target] .+= fmm.get_velocity(target_buffer, i_buffer)
    j = fmm.get_velocity_gradient(target_buffer, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target] += j[i]
    end
end
=#

#=
function fmm.buffer_to_target_system!(target_system::ParticleField, i_target, derivatives_switch, target_buffer::AbstractArray{<:ReverseDiff.TrackedReal}, i_buffer)

    u_star = deepcopy(ReverseDiff.value.(target_system.particles[U_INDEX, i_target]))
    j_star = deepcopy(ReverseDiff.value.(target_system.particles[J_INDEX, i_target]))
    #target_system.particles[U_INDEX, i_target] .= fmm.get_velocity(ReverseDiff.value.(target_buffer), i_buffer)
    u = fmm.get_velocity(target_buffer, i_buffer)
    for i = 1:3
        target_system.particles[U_INDEX[i], i_target].value += u[i].value
    end
    j = fmm.get_velocity_gradient(target_buffer, i_buffer)
    for i = 1:9
        target_system.particles[J_INDEX[i], i_target].value = j[i].value
    end
    tp = ReverseDiff.tape(target_system, target_buffer)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        fmm.buffer_to_target_system!,
                        (target_system, i_target, derivatives_switch, target_buffer, i_buffer),
                        nothing,
                        (u_star, j_star))

    return nothing

end


function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fmm.buffer_to_target_system!)})

    target_system, i_target, derivatives_switch, target_buffer, i_buffer = instruction.input
    u_star, j_star = instruction.cache
    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value = u_star[i]
        target_buffer[4+i, i_buffer].deriv += target_system.particles[U_INDEX[i], i_target].deriv
    end
    for i=1:9
        target_system.particles[J_INDEX[i], i_target].value = j_star[i]
        target_buffer[7+i, i_buffer].deriv += target_system.particles[J_INDEX[i], i_target].deriv
    end

    return nothing
    
end
=#
#5.758970865591373
#5.75664009671074
#5.7589708655913725


function _update_particle_states(pfield::ParticleField{R, <:ReformulatedVPM{R2}, V, <:Any, <:SubFilterScale, <:Any, <:Any, <:Any, <:Any, <:Any,<:Any},MM,a,b,dt::R3,Uinf,f,g,zeta0) where {R<:ReverseDiff.TrackedReal, R2, V, R3}

    # grab the floating-point type and tape from the particle field
    T = eltype(pfield.particles[1].value)
    tp = ReverseDiff.tape(pfield)
    np = pfield.np
    # we need to store the original particle field state (or at least the values for M, X, sigma, G, and MM)
    Mstar = zeros(T, np, 8)
    Xstar = zeros(T, np, 3)
    sigmastar = zeros(T, np)
    Gstar = zeros(T, np, 3)
    _MM = ReverseDiff.value.(MM) # MM is just a buffer for calculations, so we don't need to store the original value.
    for i=1:np
        for j=1:3
            Xstar[i, j] = pfield.particles[X_INDEX[j], i].value
            Gstar[i, j] = pfield.particles[GAMMA_INDEX[j], i].value
        end
        for j=1:8
            Mstar[i, j] = pfield.particles[M_INDEX[j], i].value
        end
        sigmastar[i] = pfield.particles[SIGMA_INDEX, i].value
    end

    iii = 0
    for p in iterator(pfield)
        iii += 1
        C::R = get_C(p)[1]

        # Low-storage RK step
        ## Velocity
        M = get_M(p); G = get_Gamma(p); J = get_J(p); S = get_SFS(p)
        for i=1:3
            M[i] = ReverseDiff.track(a*M[i].value + dt*(get_U(p)[i].value + ReverseDiff.value(Uinf[i])), tp)
        end

        # Update position
        X = get_X(p)
        for i=1:3
            X[i] = ReverseDiff.track(X[i].value + b*M[i].value, tp)
            #ReverseDiff.value!(X[i], X[i].value + b*M[i].value)
        end
        # Store stretching S under M[1:3]
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            _MM[1] = J[1].value*G[1].value+J[2].value*G[2].value+J[3].value*G[3].value
            _MM[2] = J[4].value*G[1].value+J[5].value*G[2].value+J[6].value*G[3].value
            _MM[3] = J[7].value*G[1].value+J[8].value*G[2].value+J[9].value*G[3].value
        else
            # Classic scheme (Γ⋅∇)U
            _MM[1] = J[1].value*G[1].value+J[4].value*G[2].value+J[7].value*G[3].value
            _MM[2] = J[2].value*G[1].value+J[5].value*G[2].value+J[8].value*G[3].value
            _MM[3] = J[3].value*G[1].value+J[6].value*G[2].value+J[9].value*G[3].value
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        _MM[4] = (f+g)/(1+3*f) * (_MM[1]*G[1].value + _MM[2]*G[2].value + _MM[3]*G[3].value)
        _MM[4] -= f/(1+3*f) * (C.value*S[1].value*G[1].value + C.value*S[2].value*G[2].value + C.value*S[3].value*G[3].value) * get_sigma(p)[].value^3/zeta0
        _MM[4] /= G[1].value^2 + G[2].value^2 + G[3].value^2

        # Store qstr_i = a_i*qstr_{i-1} + ΔΓ,
        # with ΔΓ = Δt*( S - 3ZΓ - Cϵ )

        for i=1:3
            M[i+3] = ReverseDiff.track(a*M[i+3].value + dt*(_MM[i] - 3*_MM[4]*G[i].value - C.value*S[i].value*get_sigma(p)[].value^3/zeta0), tp)
            #ReverseDiff.value!(M[i+3], a*M[i+3].value + dt*(_MM[i] - 3*_MM[4]*G[i].value - C.value*S[i].value*get_sigma(p)[].value^3/zeta0))
        end

        # Store qsgm_i = a_i*qsgm_{i-1} + Δσ, with Δσ = -Δt*σ*Z
        #ReverseDiff.value!(M[8], a*M[8].value - dt*( get_sigma(p)[].value * _MM[4] ))
        M[8] = ReverseDiff.track(a*M[8].value - dt*(get_sigma(p)[].value * _MM[4]), tp)

        # Update vectorial circulation
        for i=1:3
            #ReverseDiff.value!(G[i], G[i].value + b*M[i+3].value)
            G[i] = ReverseDiff.track(G[i].value + b*M[i+3].value, tp)
        end

        # Update cross-sectional area
        #get_sigma(p)[].value += b*M[8].value
        #ReverseDiff.value!(get_sigma(p)[], get_sigma(p)[].value + b*M[8].value)
        get_sigma(p)[] = ReverseDiff.track(get_sigma(p)[].value + b*M[8].value, tp)

    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        update_particle_states,
                        (pfield,MM,a,b,dt,Uinf,f,g,zeta0),
                        nothing,
                        (Xstar,sigmastar,Gstar))

    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(_update_particle_states)})

    pfield,MM,a,b,dt,Uinf,f,g,zeta0 = instruction.input
    Xstar, sigmastar, Gstar = instruction.cache
    np = pfield.np
    T = eltype(pfield.particles[1].value)
    # temporary containers for derivatives of variables that are updated in-place
    sigma_bar = zeros(T, 1)
    M_bar = zeros(T, 8)
    X_bar = zeros(T, 3)
    G_bar = zeros(T, 3)

    # map original values back
    for i=1:np
        for j=1:3
            pfield.particles[X_INDEX[j], i].value = Xstar[i,j]
            pfield.particles[GAMMA_INDEX[j], i].value = Gstar[i,j]
        end
        pfield.particles[SIGMA_INDEX, i].value = sigmastar[i]
    end
    
    # compute some constants
    α = -f/(1+3*f)
    β = (f+g)/(1+3*f)
    @show sum(ReverseDiff.deriv.(pfield.particles))

    # calculate cotangents for X, Gamma, M, sigma, S
    for p in iterator(pfield)
        X = get_X(p); M = get_M(p); G = get_Gamma(p); J = get_J(p); sigma = get_sigma(p)[]; C = get_C(p)[1]; S = get_SFS(p); U = get_U(p)
        
        # compute scalars that depend on the particle
        Γ2 = G[1].value^2 + G[2].value^2 + G[3].value^2
        Γ3 = G[1].value^3 + G[2].value^3 + G[3].value^3
        if pfield.transposed
            # Transposed scheme S = (Γ⋅∇')U
            MM[1].value = J[1].value*G[1].value+J[2].value*G[2].value+J[3].value*G[3].value
            MM[2].value = J[4].value*G[1].value+J[5].value*G[2].value+J[6].value*G[3].value
            MM[3].value = J[7].value*G[1].value+J[8].value*G[2].value+J[9].value*G[3].value
        else
            # Classic scheme (Γ⋅∇)U
            MM[1].value = J[1].value*G[1].value+J[4].value*G[2].value+J[7].value*G[3].value
            MM[2].value = J[2].value*G[1].value+J[5].value*G[2].value+J[8].value*G[3].value
            MM[3].value = J[3].value*G[1].value+J[6].value*G[2].value+J[9].value*G[3].value
        end

        # Store Z under MM[4] with Z = [ (f+g)/(1+3f) * S⋅Γ - f/(1+3f) * Cϵ⋅Γ ] / mag(Γ)^2, and ϵ=(Eadv + Estr)/zeta_sgmp(0)
        MM[4].value = β * (MM[1].value*G[1].value + MM[2].value*G[2].value + MM[3].value*G[3].value)
        MM[4].value += α * (C.value*S[1].value*G[1].value + C.value*S[2].value*G[2].value + C.value*S[3].value*G[3].value) * get_sigma(p)[].value^3/zeta0
        MM[4].value /= Γ2

        # 
        sigma_bar[1] += sigma.deriv
        M[8].deriv += b*sigma.deriv
        for i=1:3
            G_bar[i] += G[i].deriv
            M[i+3].deriv += b*G[i].deriv
        end
        M_bar[8] += a*M[8].deriv
        sigma.deriv -= M[8].deriv*dt*MM[4].value
        dZdC = zero(T)
        for i=1:3
            dZdC += S[i].value*G[i].value
        end
        dZdC *= α/Γ2
        for i=1:3
            dMM4dGi = zero(T)
            for j=1:3
                dMM4dGi += (β*2*G[i].value*J[j + 3*(i-1)].value - α*C.value*S[i].value)/Γ2
            end
            dMM4dGi += -2*(β*MM[i].value*G[i].value - α*C.value*S[i].value*G[i].value)/Γ3
            dZdSi = α*C.value*G[i].value/Γ2
            M_bar[i+3] += a*M[i+3].deriv
            sigma_bar[1] -= 3*M[i+3].deriv*dt*C.value*S[i].value*sigma.value^2/zeta0
            S[i].deriv += -dt*C.value*sigma.value^3*S[i].value*M[i+3].deriv/zeta0 - 3*dt*G[i].value*dZdSi*M[i+3].deriv
            C.deriv += -dt*sigma.value^3*S[i].value*M[i+3].deriv/zeta0 - 3*dt*G[i].value*dZdC*M[i+3].deriv
            G[i].deriv += -dt*3*MM[4].value*M[i+3].deriv - 3*dt*G[i].value*dMM4dGi*M[i+3].deriv
            for j=1:3
                # transposed scheme
                J[j + 3*(i-1)].deriv += dt*M[i+3].deriv*G[i].value*(1 - 3*β)
                G[i].deriv += dt*J[j + 3*(i-1)].value*M[i+3].deriv
            end
            X_bar[i] += X[i].deriv
            M[i].deriv += b*X[i].deriv
            M_bar[i] += a*M[i].deriv
            U[i].deriv += dt*M[i].deriv
        end

        # map temporary derivative containers back to the original containers
        for i=1:3
            X[i].deriv = X_bar[i]
            G[i].deriv = G_bar[i]
        end
        for i=1:8
            M[i].deriv = M_bar[i]
        end
        sigma.deriv = sigma_bar[1]
        @show ReverseDiff.deriv.(sigma)
    end

    #@show sum(ReverseDiff.deriv.(pfield.particles))

    return nothing

end

function ReverseDiff.value(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    return ParticleField{_V, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}(
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
                        pfield.useGPU,
                        #view(ReverseDiff.value(pfield.M)),
                        ReverseDiff.value.(pfield.M),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
                        )
end

function ReverseDiff.deriv(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}
    return ParticleField{D, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU, TEPS}(
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
                        pfield.useGPU,
                        #view(ReverseDiff.deriv.(pfield.M), :),
                        ReverseDiff.deriv.(pfield.M),
                        pfield.toggle_rbf,
                        pfield.toggle_sfs
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
    #@show sum(target_buffer.deriv)
    #@show sum(ReverseDiff.deriv.(target_system.particles))

    for i=1:3
        target_system.particles[U_INDEX[i], i_target].value = ustar[i]
        target_buffer.deriv[i+4, i_buffer] += target_system.particles[U_INDEX[i], i_target].deriv
    end
    for i=1:9
        target_system.particles[J_INDEX[i], i_target].value = jstar[i]
        target_buffer.deriv[i+7, i_buffer] += target_system.particles[J_INDEX[i], i_target].deriv
    end
    #@show sum(target_buffer.deriv)
    #@show sum(ReverseDiff.deriv.(target_system.particles))
    #println(" ")

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

add!(A::Number,B::Number) = A += B
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