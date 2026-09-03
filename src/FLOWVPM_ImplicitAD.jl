
function onestep!(states, states_prev, t, t_prev, xd, xci, p)

    static_particles_function, runtime_function, verbose_nsteps, v_lvl, save_pfield, save_path, nsteps_save, vprintln, nsteps, dt, custom_UJ, pfield = p
    
    map_flat_states_to_pfield!(pfield, states_prev)
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
    nextstep(pfield, dt; relax=relax, custom_UJ=custom_UJ)

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
    tp = ReverseDiff.tape(pfield)
    if eltype(states) <: ReverseDiff.TrackedReal
        for idx in CartesianIndices(states)
            states[idx] = ReverseDiff.track(0.0, tp)
        end
        map_pfield_to_flat_states!(states, pfield)
    else
        map_pfield_to_flat_states!(states, ReverseDiff.value(pfield))
    end
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
    map_pfield_to_flat_states!(states, pfield_cache)
    if eltype(xd) <: ReverseDiff.TrackedReal || eltype(xc0) <: ReverseDiff.TrackedReal
        return states
    else
        return ReverseDiff.value.(states)
    end

end

function map_flat_states_to_pfield!(pfield, states)

    for i=1:pfield.np
        for j=1:nfields
            pfield.particles[j,i] = states[j + (i-1)*nfields]
        end
    end
    pfield.np = Int(states[end])
    return nothing

end

function map_flat_states_to_pfield!(pfield, states::AbstractArray{<:ReverseDiff.TrackedReal})

    #tp = ReverseDiff.tape(pfield)
    tp = ReverseDiff.tape(states)
    states_value = deepcopy(ReverseDiff.value.(states)) # not sure if I need this.
    pfield.np = Int(ReverseDiff.value(states[end]))
    for i=1:pfield.np
        for j=1:nfields
            #pfield.particles[j,i].value = states[j + (i-1)*nfields].value
            #pfield.particles[j,i].tape = tp
            pfield.particles[j,i] = ReverseDiff.track(states[j + (i-1)*nfields].value, tp)
        end
    end
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
    tp = ReverseDiff.tape(pfield.particles)
    #tp = ReverseDiff.tape(states)
    for idx in CartesianIndices(states)
        #states[idx].value = states_value[idx] # not sure if needed. TODO: make sure commenting this out is safe
    end
    pfield.np = Int(ReverseDiff.value(states[end]))
    for i=1:pfield.np
        for j=1:nfields
            states[j + (i-1)*nfields].deriv = pfield.particles[j,i].deriv
            pfield.particles[j,i].deriv = 0.0
            states[j + (i-1)*nfields].tape = tp
            #pfield.particles[j,i].tape = tp
        end
    end
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_flat_states_to_pfield!)})

    pfield, states = instruction.input
    pfield.np = Int(ReverseDiff.value(states[end]))
    tp = ReverseDiff.tape(states)
    for i=1:pfield.np
        for j=1:nfields
            pfield.particles[j,i] = ReverseDiff.track(states[j + (i-1)*nfields].value, tp)
            #pfield.particles[j,i].value = states[j + (i-1)*nfields].value
            #pfield.particles[j,i].tape = tp
        end
    end
    return nothing

end

map_flat_states_trackedarray_to_pfield!() = error("dummy function")

function map_flat_states_to_pfield!(pfield, states::ReverseDiff.TrackedArray)

    #tp = ReverseDiff.tape(pfield)
    tp = ReverseDiff.tape(states)
    states_value = deepcopy(ReverseDiff.value(states)) # not sure if I need this.
    pfield.np = Int(states.value[end])
    for i=1:pfield.np
        for j=1:nfields
            pfield.particles[j,i].value = states.value[j + (i-1)*nfields]
            pfield.particles[j,i].tape = tp
        end
    end
    ReverseDiff.record!(tp,
                        ReverseDiff.SpecialInstruction,
                        map_flat_states_trackedarray_to_pfield!,
                        (pfield, states),
                        nothing,
                        states_value
                        )

    return nothing

end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_flat_states_trackedarray_to_pfield!)})

    pfield, states = instruction.input
    states_value = instruction.cache
    tp = ReverseDiff.tape(pfield.particles)
    #tp = ReverseDiff.tape(states)
    for idx in CartesianIndices(states)
        #states[idx].value = states_value[idx] # not sure if needed. TODO: make sure commenting this out is safe
    end
    pfield.np = Int(ReverseDiff.value(states[end]))
    for i=1:pfield.np
        for j=1:nfields
            states.deriv[j + (i-1)*nfields] = pfield.particles[j,i].deriv
            pfield.particles[j,i].deriv = 0.0
            #states[j + (i-1)*nfields].tape = tp
            #pfield.particles[j,i].tape = tp
        end
    end
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_flat_states_trackedarray_to_pfield!)})

    pfield, states = instruction.input
    pfield.np = Int(ReverseDiff.value(states[end]))
    tp = ReverseDiff.tape(states)
    for i=1:pfield.np
        for j=1:nfields
            pfield.particles[j,i].value = states.value[j + (i-1)*nfields]
            pfield.particles[j,i].tape = tp
        end
    end
    return nothing

end


function map_pfield_to_flat_states!(states, pfield)

    for i=1:pfield.np
        for j=1:nfields
            states[j + (i-1)*nfields] = pfield.particles[j,i]
        end
    end
    states[end] = pfield.np
    return nothing

end

function map_pfield_to_flat_states!(states::AbstractArray{<:ReverseDiff.TrackedReal}, pfield)

    tp = ReverseDiff.tape(pfield)
    pfield_value = deepcopy(ReverseDiff.value.(pfield.particles[:, 1:pfield.np]))
    for i=1:pfield.np
        for j=1:nfields
            states[j + (i-1)*nfields] = ReverseDiff.track(pfield.particles[j,i].value, tp)
            #states[j + (i-1)*nfields].value = pfield.particles[j,i].value
            #states[j + (i-1)*nfields].tape = tp
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
    tp = ReverseDiff.tape(states)

    for idx in CartesianIndices(pfield_value)
        #pfield.particles[idx].value = pfield_value[idx] # not sure if needed.
        pfield.particles[idx] = ReverseDiff.track(pfield_value[idx], tp)
    end
    pfield.np = np
    for i=1:pfield.np
        for j=1:nfields
            pfield.particles[j,i].deriv = states[j + (i-1)*nfields].deriv # fails - the memory for pfield.particles is shared.
            #states[j + (i-1)*nfields].deriv = 0.0
            #pfield.particles[j,i].tape = tp
        end
    end
    states[end].value = np
    return nothing

end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(map_pfield_to_flat_states!)})

    states, pfield = instruction.input
    pfield_value, np = instruction.cache
    tp = ReverseDiff.tape(pfield)
    
    pfield_value = deepcopy(ReverseDiff.value.(pfield.particles))
    for i=1:np
        for j=1:nfields
            #states[j + (i-1)*nfields].value = pfield.particles[j,i].value
            #states[j + (i-1)*nfields].tape = tp
            states[j + (i-1)*nfields] = ReverseDiff.track(pfield.particles[j,i].value, tp)
        end
    end
    T = eltype(pfield.particles[1].value) # get the appropriate floating point type
    states[end] = ReverseDiff.track(T(pfield.np), tp)
    return nothing

end

function onestep_compiled!(states, states_prev, t, t_prev, xd, xci, p)

    static_particles_function, runtime_function, verbose_nsteps, v_lvl, save_pfield, save_path, nsteps_save, vprintln, nsteps, dt, custom_UJ, pfield = p
    
    #tp = ReverseDiff.tape(states_prev)
    tp = ReverseDiff.tape(xd)
    for idx in CartesianIndices(pfield.particles)
        pfield.particles[idx].tape = tp
    end

    for idx in CartesianIndices(pfield.particles)
        pfield.particles[idx].tape = tp
    end

    #println("tp pointer: $(pointer_from_objref(tp)))")
    #println("pfield pointer: $(pointer_from_objref(ReverseDiff.tape(pfield))))")
    #println("states_prev pointer: $(pointer_from_objref(ReverseDiff.tape(states_prev))))")

    check_derivs(states_prev; label="states_prev")

    map_flat_states_to_pfield!(pfield, states_prev)
    # no use for xci for now
    check_derivs(pfield.particles; label="pfield.particles")

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
    nextstep(pfield, dt; relax=relax, custom_UJ=custom_UJ)

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
    check_derivs(pfield.particles; label="pfield 2")
    if eltype(states) <: ReverseDiff.TrackedReal
        #println("onestep tape pointer: $(pointer_from_objref(tp))")
        #println("pfield tape: $(pointer_from_objref(ReverseDiff.tape(pfield)))")
        println("length of onestep tape: $(length(tp))")
        println("length of pfield tape: $(length(ReverseDiff.tape(pfield)))")
        for idx in CartesianIndices(states)
            states[idx] = ReverseDiff.track(0.0, tp)
        end
        for idx in CartesianIndices(pfield.particles)
            #pfield.particles[idx].tape = tp
        end
        map_pfield_to_flat_states!(states, pfield)
        #println("states pointer: $(pointer_from_objref(states[1].tape))")
        #println("pfield pointer after mapping pfield to states: $(pointer_from_objref(ReverseDiff.tape(pfield)))")
        println("length of states tape: $(length(states[1].tape))")
        println("length of pfield tape after mapping pfield to states: $(length(ReverseDiff.tape(pfield)))")
    else
        map_pfield_to_flat_states!(states, ReverseDiff.value(pfield))
    end
    check_derivs(states; label="states 1")
    #check_derivs(pfield.particles; label="pfield 1")
    check_derivs(pfield.particles; label="pfield.particles")
    return nothing

end

function initialize_compiled(t0, xd, xc0, p)

    static_particles_function, runtime_function, verbose_nsteps, v_lvl, save_pfield, save_path, nsteps_save, vprintln, nsteps, dt, custom_UJ, pfield_cache = p
    pfield_cache.t = t0
    i = 0
    if i%verbose_nsteps==0
        vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield_cache))", v_lvl+1)
    end
    tp = ReverseDiff.tape(xd)
    for idx in CartesianIndices(pfield_cache.particles)
        pfield_cache.particles[idx].tape = tp
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
    #check_derivs(states; label="states 2")
    #check_derivs(pfield_cache.particles; label="pfield 2")
    map_pfield_to_flat_states!(states, pfield_cache)
    #check_derivs(states; label="states 1")
    #check_derivs(pfield_cache.particles; label="pfield 1")
    if eltype(xd) <: ReverseDiff.TrackedReal || eltype(xc0) <: ReverseDiff.TrackedReal
        return states
    else
        return ReverseDiff.value.(states)
    end

end

# This section might get moved to ImplicitAD.jl.
##############

function compile_tape(f, input)
    input_copy = init_input_copy(input)
    #@show length(ReverseDiff.tape(input_copy))
    #@show length(ReverseDiff.tape(input))
    #@show sum(input.particles) sum(input_copy.particles)
    output_copy = f(input_copy)
    #@show length(ReverseDiff.tape(input_copy))
    #@show length(ReverseDiff.tape(input))
    return (input_copy, output_copy, ReverseDiff.tape(input_copy)) # this should handle cases without an output without any extra work.
end

function compile_tape(f, inputs::Tuple)
    input_copy = init_input_copy(inputs)
    output_copy = f(input_copy...)
    return (input_copy, output_copy, ReverseDiff.tape(input_copy)) # this should handle cases without an output without any extra work.
end

init_input_copy(x::ReverseDiff.TrackedReal, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) = ReverseDiff.track(x.value, tp)
init_input_copy(x::Number, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) = ReverseDiff.track(x, tp)
init_input_copy(x::ReverseDiff.TrackedArray, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) = ReverseDiff.track(deepcopy(x.value), tp)

function init_input_copy(x::AbstractArray, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape())
    
    if !ReverseDiff.istracked(x)
        return x
    end
    out = deepcopy(x)
    for i in eachindex(x)
        #out[i] = ReverseDiff.track(deepcopy(out[i].value), tp)
        out[i] = init_input_copy(x[i], tp)
        #out[i].tape = tp
    end
    return out

end

function init_input_copy(x::NTuple{N, Any}, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) where N

    out = ntuple((i)->init_input_copy(x[i], tp), N)

    return out

end

function run_compiled_function(input, cache)
    input_copy, output_copy, inner_tape = cache
    outer_tape = _tape(input)
    _copy_val_and_deriv!(input_copy, input)
    ReverseDiff.forward_pass!(inner_tape)
    _copy_val_and_deriv!(input, input_copy)
    output = _track(output_copy, outer_tape)
    ReverseDiff.record!(outer_tape,
                        ReverseDiff.SpecialInstruction,
                        run_compiled_function,
                        input,
                        output,
                        cache)
    return output
end

function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(run_compiled_function)})
    input = instruction.input
    output = instruction.output
    input_copy, output_copy, inner_tape = instruction.cache
    #println("beginning reverse pass")
    #@show sum(ReverseDiff.deriv.(input.particles))
    #@show sum(ReverseDiff.deriv.(input_copy.particles))
    _copy_val_and_deriv!(input_copy, input)
    #@show sum(ReverseDiff.deriv.(input.particles))
    #@show sum(ReverseDiff.deriv.(input_copy.particles))
    output !== nothing && _copy_val_and_deriv!(output_copy, output)
    ReverseDiff.reverse_pass!(inner_tape)

    #@show sum(ReverseDiff.deriv.(input.particles))
    #@show sum(ReverseDiff.deriv.(input_copy.particles))
    #@show ReverseDiff.value.(input_copy.particles[SIGMA_INDEX, :])
    #@show ReverseDiff.deriv.(input_copy.particles[SIGMA_INDEX, :])
    _copy_val_and_deriv!(input, input_copy)

    #@show sum(ReverseDiff.deriv.(input.particles))
    #@show sum(ReverseDiff.deriv.(input_copy.particles))
    output !== nothing && _copy_val_and_deriv!(output, output_copy)
    #println("ending reverse pass")
    return nothing
end

function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(run_compiled_function)})

    input = instruction.input
    output = instruction.output
    input_copy, output_copy, inner_tape = instruction.cache
    _copy_val_and_deriv!(input_copy, input)
    _copy_val_and_deriv!(output_copy, output)
    ReverseDiff.forward_pass!(inner_tape)
    _copy_val_and_deriv!(input, input_copy)
    _copy_val_and_deriv!(output, output_copy)
    return nothing

end

@inline _tape(x) = ReverseDiff.tape(x)
function ReverseDiff.tape(x::NTuple{N, Any}) where N
    for i in eachindex(x)
        if length(_tape(x[i])) > 0
            return _tape(x[i])
        end
    end
    return ReverseDiff.tape(x[1])
end

@inline _track(x, tp) = error("$(typeof(x)) is not a tracked type!") # = deepcopy(x)
@inline _track(x::Nothing, tp) = nothing
@inline _track(x::ReverseDiff.TrackedReal, tp) = return ReverseDiff.track(x.value, tp)
@inline _track(x::ReverseDiff.TrackedArray, tp) = return ReverseDiff.track(deepcopy(x.value), tp)
function _track(x::AA, tp) where {AA <: AbstractArray}
    out = AA(undef,size(x))
    for i in eachindex(x)
        out[i] = _track(x[i], tp)
    end
    return out
end

function _track(x::NTuple{N, Any}, tp) where N
    out = NTuple{N, Any}
    for i in eachindex(x)
        out[i] = _track(x[i], tp)
    end
    return out
end

@inline _copy_val_and_deriv!(dest, src) = error("$(typeof(dest))\n$(typeof(src))") # copyto!(dest, src) # 
@inline _copy_val_and_deriv!(dest::ReverseDiff.TrackedReal, src::ReverseDiff.TrackedReal) = (ReverseDiff.value!(dest, src.value); ReverseDiff.deriv!(dest, src.deriv); nothing)
@inline _copy_val_and_deriv!(dest::ReverseDiff.TrackedArray, src::ReverseDiff.TrackedArray) = (ReverseDiff.value!(dest, deepcopy(src.value)); ReverseDiff.deriv!(dest, deepcopy(src.deriv)); nothing)

function init_rd_array!(arr::AbstractArray{<:ReverseDiff.TrackedReal}, tp)

    for idx in CartesianIndices(arr)
        arr[idx] = ReverseDiff.track(arr[idx].value, tp)
    end
    return nothing

end

function _copy_val_and_deriv!(dest::ReverseDiff.TrackedArray, src::AbstractArray)
    for i in eachindex(src)
        dest.value[i] = src[i].value
        dest.deriv[i] = src[i].deriv
    end
end

function _copy_val_and_deriv!(dest::AbstractArray, src::ReverseDiff.TrackedArray)
    init_rd_array!(dest, ReverseDiff.tape(dest))
    for i in eachindex(dest)
        dest[i].value = src.value[i]
        dest[i].deriv = src.deriv[i]
    end
end

function _copy_val_and_deriv!(dest::AA, src::AA) where {AA <: AbstractArray}
    if !ReverseDiff.istracked(src)
        for i in eachindex(src)
            dest[i] = src[i]
        end
        return nothing
    end
    for i in eachindex(src)
        #dest[i] = src[i] # this corrupts the primal value, probably because src or dest can end up with elements shared memory.
        _copy_val_and_deriv!(dest[i], src[i])
    end
    return nothing
end

function _copy_val_and_deriv!(dest::NTuple{N, Any}, src::NTuple{N, Any}) where N
    for i in eachindex(src)
        _copy_val_and_deriv!(dest[i], src[i])
    end
    return nothing
end

# extra methods for particle fields

function init_input_copy(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}, tp::ReverseDiff.InstructionTape = ReverseDiff.InstructionTape()) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}
    
    out = ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}(
                        pfield.maxparticles,
                        init_input_copy(pfield.particles, tp),
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

    return out
end

function _copy_val_and_deriv!(dest::ParticleField, src::ParticleField)
    _copy_val_and_deriv!(dest.particles, src.particles)
    dest.np = src.np
    return nothing
end

function _track(pfield::ParticleField{ReverseDiff.TrackedReal{_V, D, O}, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}, tp) where {_V, D, O, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}
    out = ParticleField{_V, F, V, TUinf, S, Tkernel, TUJ, Tintegration, TRelaxation, TGPU}(
                        pfield.maxparticles,
                        _track(pfield.particles, tp),
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
    return out
end

@inline _tape(x::ParticleField) = ReverseDiff.tape(x.particles[1])

# sketch of a compilation-safe for loop:

#=

    function for_compilation_safe(idxs, expr, vars)
        for i in idxs
            expr(i, vars)
        end
    end

=#

# sketch of a way to handle inputs that have optional defaults in a compilation-safe way: