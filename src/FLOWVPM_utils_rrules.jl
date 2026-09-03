# utils doesn't have any actual mathematics, but it has a bunch of code sections that AD should not try to differentiate.

# this ensures that doesn't happen.

# Saving a simulation to a file is non-differentiable.
function save(
    self::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any},
    file_name::String; path::String="",
    add_num::Bool=true, num::Int64=-1, createpath::Bool=false,
    overwrite_time=nothing) where TF <: Union{ForwardDiff.Dual, ReverseDiff.TrackedReal}

    return nothing

end
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(save), Any, Any}

# Terminal/text file output is non-differentiable.
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(Dates.DateTime), Any}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(Dates.now)}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(Base.repeat), String, Int64}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(initialize_verbose), Any, Any, Any, Any, Any, Any, Any, Any, Any}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(initialize_verbose), Vararg}

#rrule!!(f::Mooncake.CoDual{typeof(initialize_verbose)}, x::Vararg) = zero_adjoint(initialize_verbose, x...);

#rrule!!(f::Mooncake.CoDual{typeof(initialize_verbose)}, verbose, save_path, run_name, pfield, dt, nsteps_save, runtime_function, static_particles_function, v_lvl) = zero_adjoint(f, verbose, save_path, run_name, pfield, dt, nsteps_save, runtime_function, static_particles_function, v_lvl);

# verbose, save_path, run_name, pfield, dt, nsteps_save, runtime_function, static_particles_function, v_lvl

# for the sake of testing, here's a version of the run function that doesn't need functors/stateful functions.
# Functors are theoretically fine, but they break derivatives with ReverseDiff (which doesn't come with functor handling) and cause errors with Mooncake. The Mooncake errors seem to be an implementation gap, and I think I could get a MWE put together pretty quick.
# ReverseDiff could handle functors with custom primal/reverse calls (to preserve access to the functor's internal states)
# For Mooncake, I may just need to be more careful with scoping. But I can revisit that later.
function run_vpm_no_functors!(pfield::ParticleField, dt::Real, nsteps::Int;
    custom_UJ=nothing,
    # OUTPUT OPTIONS
    save_path::Union{Nothing, String}=nothing,
    save_pfield::Bool=true,
    create_savepath::Bool=true,
    run_name::String="pfield",
    save_code::String="",
    nsteps_save::Int=1, prompt::Bool=true,
    save_time=true)

    if save_path !== nothing
        # Create save path
        if create_savepath; create_path(save_path, prompt); end;

        # Save code
        if save_code!=""
            cp(save_code, joinpath(save_path, splitdir(save_code)[2]); force=true)
        end

        # Save settings
        save_settings(pfield, run_name; path=save_path)
    end

    for i in 0:nsteps
        if i%verbose_nsteps==0
            #vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
            println("Time step $i out of $nsteps\tParticles: $(get_np(pfield))")
        end

        # Relaxation step
        relax = pfield.relaxation != relaxation_none &&
        pfield.relaxation.nsteps_relax >= 1 &&
        i>0 && (i%pfield.relaxation.nsteps_relax == 0)

        org_np = get_np(pfield)

        if i!=0
            # Add static particles
            remove = static_particles_function(pfield, pfield.t, dt)
            #@show ReverseDiff.value(sum(pfield.particles))
            # Step in time solving governing equations
            #check_derivs(pfield.particles)
            nextstep(pfield, dt; relax, custom_UJ)
            #check_derivs(pfield.particles)
            #@show ReverseDiff.value(sum(pfield.particles))
            # Remove static particles (assumes particles remained sorted)
            if remove===nothing || remove
                for pi in get_np(pfield):-1:(org_np+1)
                    remove_particle(pfield, pi)
                end
            end
        end

        # Calls user-defined runtime function
        breakflag = init_runtime_function(pfield, pfield.t, dt)

        # Save particle field
        # Currently only saves when using AbstractFloat numbers (i.e., not when using AD).
        # Supporting AD types requires custom save() implementations that convert the AD types to something that hdf5 can interpret.
        #@show eltype(pfield.particles) (eltype(pfield.particles) <: AbstractFloat)
        if save_pfield && save_path!==nothing && (i%nsteps_save==0 || i==nsteps || breakflag)
            overwrite_time = save_time ? nothing : pfield.nt
            save(pfield, run_name; path=save_path, add_num=true,
            overwrite_time=overwrite_time)
        end

        # User-indicated end of simulation
        if breakflag
            break
        end
    end
end

function init_runtime_function(_pfield, t, dt)

    T = eltype(_pfield)
    X = zeros(T, 3)

    #vpm.check_deriv_allocation(_pfield.particles[1:_pfield.np, :])
    for i=1:maxp
        X[1] = input_vector[i]
        X[2] = input_vector[i]
        X[3] = input_vector[i]
        vpm.add_particle(_pfield, X, ones(T, 3), 1.4*one(T))
        #vpm.check_deriv_allocation(_pfield.particles[1:_pfield.np, :])
    end

    return false

end