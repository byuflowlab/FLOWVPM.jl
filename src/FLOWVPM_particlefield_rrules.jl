# Extends the ParticleField interface to work with ReverseDiff.

# Adds a ReverseDiff interface for ParticleField, which allows it to be treated similarly to an array.

# Tape access is used to ensure all saved values on the forwards pass are in the same tape. This is also used internally by ReverseDiff to check if something is tracked.
ReverseDiff.tape(pfield::ParticleField) = ReverseDiff.tape(pfield.particles)

# Catch-all implementions of methods for getting the value/derivative of a ParticleField. This supports most operations involving ParticleFields, but there are some special cases handled later.
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

# This function handles particle creation when using ReverseDiff. This implementation is a bit messy.
# The main focus is making sure that the memory for new particles is properly allocated (i.e., memory is not shared between particles) and that the instantiated particle count is properly decremented on the reverse pass.
# We also have to account for particle creation where some but not all input values are tracked. While we can't predict which incoming types will be tracked, we can promote local versions of each input on the forward pass as necessary and only propagate cotangents back to tracked inputs on the reverse pass.

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
        pfield.particles[X_INDEX[i], i_next] = ReverseDiff.track(ReverseDiff.value(X[i]), tp)
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

    # This set of indices are zero by default. If we don't explicitly initialize them individually, then their memory can end up fused together.
    T = eltype(pfield.particles[1].value)
    for i in [U_INDEX..., VORTICITY_INDEX..., J_INDEX..., M_INDEX..., PSE_INDEX..., SFS_INDEX..., U_PREV_INDEX...]
        pfield.particles[i, i_next] = ReverseDiff.track(zero(T), tp)
    end

    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        add_particle,
                        (pfield, X, Gamma, sigma, vol, circulation, C, static),
                        pfield,
                        pfield.np-1)
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
    pfield.np = instruction.cache

    if get_np(pfield)==pfield.maxparticles
        error("PARTICLE OVERFLOW. Max number of particles $(pfield.maxparticles)"*
                            " has been reached")    
    end
    # Fetch the index of the next empty particle in the field
    i_next = get_np(pfield)+1

    # Add particle to the field
    pfield.np += 1

    for i=1:3
        pfield.particles[X_INDEX[i], i_next].value = ReverseDiff.value(X[i])
        pfield.particles[GAMMA_INDEX[i], i_next].value = ReverseDiff.value(Gamma[i])
        pfield.particles[X_INDEX[i], i_next].value = X[i].value
    end
    for i=1:3
        if typeof(C) <: AbstractArray
            pfield.particles[C_INDEX[i], i_next].value = ReverseDiff.value(C[i])
        else
            pfield.particles[C_INDEX[i], i_next].value = ReverseDiff.value(C)
        end
    end
    pfield.particles[SIGMA_INDEX, i_next].value = ReverseDiff.value(sigma)
    pfield.particles[CIRCULATION_INDEX, i_next].value = ReverseDiff.value(circulation)
    pfield.particles[STATIC_INDEX, i_next].value = ReverseDiff.value(static)

    return nothing

end

# There's a long list of setters for ParticleField entries. Using metaprogramming for creating the reverse-mode versions saves several hundred lines of repetitive code.
# The original functions have the following form: pfield.particles[_idxs, i] .= val
# The augmented forward pass needs to retain the original value(s) before the destructive assignment is made; this allows the
#    original values to be put back during the reverse pass.
# The reverse pass accumulates the cotangent of the ParticleField entries into the array that originally wrote new values into the ParticleField on the forward pass.
#    The cotangent of the ParticleField entry is cleared, since the old value never propagates through to the end of the overall computation (i.e., it was destructively overwritten).
# Some extra care is needed to handle assignments where the ParticleField is involved on both sides (particularly if the same array entries are used on both sides):
#    We need to add the derivative value of the ParticleField entry into the overwriting array; if this overwriting array involves the ParticleField then we double-count the ParticleField derivatives.
#    When the ParticleField cotangent is cleared, we may also clear cotangent information we just accumulated into the overwriting array (which is just the original ParticleField).
#    The solution here is to store the value we accumulate into a temporary scaler. Then we can safely clear the ParticleField before accumulating into the overwriting array; if the overwriting array involves the ParticleField we still have the correct cotangent saved.
#    Scalars are cheap to store so this solution should have minimal overhead.
# The single-element and multi-element index lists are split to avoid errors related to looping over and indexing integer variables.
# There is also some case checking to handle both behaviors of the .= operator; this implementation should work whether .= is used to assign all entries of some particle's fields to the same value or to assign the entries elementwise to match an input array.

for g in ("X", "Gamma", "U", "vorticity", "J", "M", "C", "PSE", "SFS", "MU", "MQSTR", "MS")
    _f = Symbol(:set_, g)
    _idxs = Symbol(uppercase(g), :_INDEX)
    eval(quote
        #pfield.particles[_idxs, i] .= val
        @show $_f
        function $_f(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}, i::Int, val) where {R<:ReverseDiff.TrackedReal, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TR, useGPU}
            tp = ReverseDiff.tape(pfield, val)
            #println("$($_f) ran!")
            old_val = zeros(ReverseDiff.valtype(R), size($_idxs))
            for j = 1:length($_idxs)
                old_val[j] = ReverseDiff.value(pfield.particles[$_idxs[j], i])
                if typeof(val) <: AbstractArray
                    pfield.particles[$_idxs[j], i].value = ReverseDiff.value(val[j])
                else
                    pfield.particles[$_idxs[j], i].value = ReverseDiff.value(val)
                end
            end
            ReverseDiff.record!(tp,
                                ReverseDiff.SpecialInstruction,
                                $_f,
                                (pfield, i, val),
                                nothing,
                                (old_val, $_idxs))
            return nothing
        end
        function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($_f)})
            pfield, i, val = instruction.input
            old_val, _idxs = instruction.cache
            for j=1:length(_idxs)
                pfield.particles[_idxs[j], i].value = old_val[j]
                temp = pfield.particles[_idxs[j], i].deriv
                ReverseDiff.unseed!(pfield.particles[_idxs[j], i])
                if typeof(val) <: AbstractArray
                    ReverseDiff._add_to_deriv!(val[j], temp)
                else
                    ReverseDiff._add_to_deriv!(val, temp)
                end
            end
            #@show val pfield.particles[_idxs, i]
            return nothing
        end
        function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($_f)})

            pfield, i, val = instruction.input
            old_val, _idxs = instruction.cache
            for j = 1:length(_idxs)
                old_val[j] = ReverseDiff.value(pfield.particles[$_idxs[j], i])
                if typeof(val) <: AbstractArray
                    pfield.particles[$_idxs[j], i].value = ReverseDiff.value(val[j])
                else
                    pfield.particles[$_idxs[j], i].value = ReverseDiff.value(val)
                end
            end
            return nothing
        
        end

    end)

end

for g in ("sigma", "vol", "circulation", "static", "U_prev", "MQSGM", "MS1", "MS2", "MS3")
    _f = Symbol(:set_, g)
    _idx = Symbol(uppercase(g), :_INDEX)
    eval(quote
        @show $_f
        function $_f(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}, i::Int, val) where {R<:ReverseDiff.TrackedReal, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TR, useGPU}
            tp = ReverseDiff.tape(pfield, val)
            #println("$($_f) ran!")
            old_val = ReverseDiff.value(pfield.particles[$_idx, i])
            pfield.particles[$_idx, i].value = ReverseDiff.value(val)
            ReverseDiff.record!(tp,
                                ReverseDiff.SpecialInstruction,
                                $_f,
                                (pfield, i, val),
                                nothing,
                                ([old_val], $_idx))
            return nothing
        end
        function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($_f)})
            pfield, i, val = instruction.input
            old_val, _idx = instruction.cache
            pfield.particles[_idx, i].value = old_val[1]
            temp = pfield.particles[_idx, i].deriv
            ReverseDiff.unseed!(pfield.particles[_idx, i])
            ReverseDiff._add_to_deriv!(val, temp)
            return nothing
        end
        function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($_f)})

            pfield, i, val = instruction.input
            old_val, _idx = instruction.cache
            old_val[1] = ReverseDiff.value(pfield.particles[$_idx, i])
            pfield.particles[$_idx, i].value = ReverseDiff.value(val)
            return nothing
        end
    end)
end

#function set_one_field(pfield::ParticleField, i::Int, FIELD_INDEX::Int, val) pfield.particles[FIELD_INDEX, i] = val end
function set_one_field(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}, i::Int, FIELD_INDEX::Int, val) where {R<:ReverseDiff.TrackedReal, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TR, useGPU}

    tp = ReverseDiff.tape(pfield, val)
    old_val = ReverseDiff.value(pfield.particles[FIELD_INDEX, i])
    pfield.particles[FIELD_INDEX, i].value = ReverseDiff.value(val)
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        set_one_field,
                        (pfield, i, FIELD_INDEX, val),
                        nothing,
                        old_val)
    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(set_one_field)})
    pfield, i, FIELD_INDEX, val = instruction.input
    old_val = instruction.cache
    pfield.particles[FIELD_INDEX, i].value = old_val
    temp = pfield.particles[FIELD_INDEX, i].deriv
    ReverseDiff.unseed!(pfield.particles[FIELD_INDEX, i])
    ReverseDiff._add_to_deriv!(val, temp)
    return nothing
end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($_f)})

    pfield, i, FIELD_INDEX, val = instruction.input
    old_val = instruction.cache
    old_val = ReverseDiff.value(pfield.particles[FIELD_INDEX, i])
    pfield.particles[FIELD_INDEX, i].value = ReverseDiff.value(val)
    return nothing
end


#function set_all_fields(pfield::ParticleField, val) pfield.particles .= val end
function set_all_fields(pfield::ParticleField{R, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TR, useGPU}, val) where {R<:ReverseDiff.TrackedReal, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale, Tkernel, TUJ, Tintegration, TR, useGPU}
    tp = ReverseDiff.tape(pfield, val)
    old_val = zeros(ReverseDiff.valtype(R), size(val))
    for i = 1:pfield.maxparticles
        for j = 1:nfields
            old_val[j, i] = ReverseDiff.value(pfield.particles[j, i])
            pfield.particles[j, i].value = ReverseDiff.value(val[j, i])
        end
    end
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        set_all_fields,
                        (pfield, val),
                        nothing,
                        old_val)
    return nothing
end
function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(set_all_fields)})
    pfield, val = instruction.input
    old_val = instruction.cache
    for i=1:pfield.maxparticles
        for j=1:nfields
            pfield.particles[j, i].value = old_val[j, i]
            temp = pfield.particles[j, i].deriv
            ReverseDiff.unseed!(pfield.particles[j, i])
            ReverseDiff._add_to_deriv!(val[j, i], temp)
        end
    end
    return nothing
end
function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(set_all_fields)})

    pfield, val = instruction.input
    old_val = instruction.cache
    for i=1:pfield.maxparticles
        for j = 1:nfields
            old_val[j, i] = ReverseDiff.value(pfield.particles[j, i])
            pfield.particles[j, i].value = ReverseDiff.value(val[j, i])
        end
    end
    return nothing

end