#=##############################################################################
# DESCRIPTION
    Runs the VPM. Code is pulled from the original utils file FLOWVPM_utils.

# AUTHORSHIP
  * Author    : Eric Green
  * Email     : eric.parley.green@gmail.com
  * Created   : Feb 2023
=###############################################################################

"""
`run_vpm!(pfield, dt, nsteps; runtime_function=nothing, save_path=nothing,
run_name="pfield", nsteps_save=1, verbose=true, prompt=true)`

Solves `nsteps` of the particle field with a time step of `dt`.

**Optional Arguments**
* `runtime_function::Function`   : Give it a function of the form
                          `myfun(pfield, t, dt)`. On each time step it
                          will call this function. Use this for adding
                          particles, deleting particles, etc.
* `static_particles_function::Function`   : Give it a function of the form
                          `myfun(pfield, t, dt)` to add static particles
                          representing solid boundaries to the solver. This
                          function is called at every time step right before
                          solving the governing equations, and any new
                          particles added by this function are immediately
                          removed.
* `save_path::String`   : Give it a string for saving VTKs of the particle
                          field. Creates the given path.
* `run_name::String`    : Name of output files.
* `nsteps_save::Int64`  : Saves vtks every this many time steps.
* `prompt::Bool`        : If `save_path` already exist, it will prompt the
                          user before overwritting the folder if true; it will
                          directly overwrite it if false.
* `verbose::Bool`       : Prints progress of the run to the terminal.
* `verbose_nsteps::Bool`: Number of time steps between verbose.
"""
function run_vpm(pfield::PF, dt::Real, nsteps::Int;
                    # RUNTIME OPTIONS
                    runtime_function::Function=runtime_default,
                    static_particles_function::Function=static_particles_default,
                    # OUTPUT OPTIONS
                    save_path::Union{Nothing, String}=nothing,
                    create_savepath::Bool=true,
                    run_name::String="pfield",
                    save_code::String="",
                    nsteps_save::Int=1, prompt::Bool=true,
                    verbose::Bool=true, verbose_nsteps::Int=10, v_lvl::Int=0,
                    save_time=true, use_implicitAD=true) where {PF <: ParticleField}

    # ERROR CASES
    ## Check that viscous scheme and kernel are compatible
    compatible_kernels = _kernel_compatibility[typeof(pfield.viscous).name]

    if !(pfield.kernel in compatible_kernels)
        error("Kernel $(pfield.kernel) is not compatible with viscous scheme"*
                " $(typeof(pfield.viscous).name); compatible kernels are"*
                " $(compatible_kernels)")
    end

    if save_path!=nothing
        # Create save path
        if create_savepath; create_path(save_path, prompt); end;

        # Save code
        if save_code!=""
            cp(save_code, joinpath(save_path, splitdir(save_code)[2]); force=true)
        end

        # Save settings
        save_settings(pfield, run_name; path=save_path)
    end

    # Initialize verbose
    (line1, line2, run_id, file_verbose,
        vprintln, time_beg) = initialize_verbose(   verbose, save_path, run_name, pfield,
                                                    dt, nsteps_save,
                                                    runtime_function,
                                                    static_particles_function, v_lvl)

    # RUN
    # Feb 15: Split this into its own function because I'll need to pass it into ImplicitAD.


    if use_implicitAD
        pfield_out = solve_vpm_implicitAD(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)
        finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
        return pfield_out
    else
        solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)
        finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
        return pfield
    end
    # Finalize verbose

    #return nothing
end

function solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)

    for i in 0:nsteps

        if i%verbose_nsteps==0
            vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
        end

        # Relaxation step
        #=relax = pfield.relaxation != relaxation_none &&
                pfield.relaxation.nsteps_relax >= 1 &&
                i>0 && (i%pfield.relaxation.nsteps_relax == 0)=#

        org_np = get_np(pfield)

        # Time step
        if i!=0
            # Add static particles
            remove = static_particles_function(pfield, pfield.t, dt)

            # Step in time solving governing equations
            nextstep(pfield, dt)#; relax=relax)
            # relaxation moved up a level, although this is really just a placeholder. It might end up disabled anyway.
            #if relax
            #    pfield.relaxation(p)
            #end

            # Remove static particles (assumes particles remained sorted)
            if remove==nothing || remove
                for pi in get_np(pfield):-1:(org_np+1)
                    remove_particle(pfield, pi)
                end
            end
        end

      # Calls user-defined runtime function
        breakflag = runtime_function(pfield, pfield.t, dt;
                                    vprintln= (str)-> i%verbose_nsteps==0 ?
                                    vprintln(str, v_lvl+2) : nothing)

      # Save particle field
        if save_path!=nothing && (i%nsteps_save==0 || i==nsteps || breakflag)
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

using ImplicitAD

function solve_vpm_implicitAD(pfield, nsteps, dt, verbose_nsteps,
                              static_particles_function, runtime_function,
                              v_lvl, vprintln, save_path, nsteps_save,
                              save_time, run_name)

    pfield_vec, settings = pfield2vec(pfield)
    println(typeof(pfield_vec))
    println(typeof(pfield))
    # sim_settings is a an array of type any that carries a bunch of information that the VPM needs that doesn't change through time.
    sim_settings = [nsteps, dt, verbose_nsteps,
                    static_particles_function,
                    runtime_function, v_lvl, vprintln,
                    save_path, nsteps_save, save_time,
                    run_name]
    function VPM_step_2!(y,yprev,t,tprev,x,p)
        if t == tprev
            _pfield_vec,_settings = VPM_AD_solve(x,settings,t,sim_settings) # This is necessary because the first yprev is x with all the AD information removed.
            y .= _pfield_vec
        else
            _pfield_vec,_settings = VPM_AD_solve(yprev,settings,t,sim_settings)
            y .= _pfield_vec
        end
    end
    VPM_solver(_state,p) = VPM_solve(_state,settings,sim_settings,nsteps)
    out_u,out_t = ImplicitAD.explicit_unsteady(VPM_solver,VPM_step_2!,pfield_vec,())
    final_timestep_out = out_u[:,end]
    #println(final_timestep_out[1:3])
    pfield_out = vec2pfield(final_timestep_out,settings)
    println(pfield.np_hist)
    println(pfield_out.np_hist)
    #println(pfield_out.particles[1].X)
    return pfield_out

end

# Runs the VPM by stepping forward through time. This function is essentially the solver passed to ImplicitAD (although there is actually an extra function wrapper to get the correct function signature)
function VPM_solve(pfield_vec_in,settings,sim_settings,nsteps)

    #println("running forward solve")
    #R = eltype(pfield_vec_in)
    R = typeof(pfield_vec_in[1])
    println(R)
    len = nsteps+1
    dt = sim_settings[2]
    t = Vector{R}(undef,len)
    u = zeros(R,len,length(pfield_vec_in))
    settings[12] .= -1.0
    cache_pfield = vec2pfield(pfield_vec_in,settings)
    for i=1:len
        if i == 1
            t[i] = R(0.0) # see march 16 notes; this is a weird way to get the initial time and it should probably come from sim_settings
            u[i,:],settings = VPM_forward_solve(pfield_vec_in,settings,t[i],sim_settings,cache_pfield)
        else
            t[i] = t[i-1] + dt
            u[i,:],settings = VPM_forward_solve(u[i-1,:],settings,t[i],sim_settings,cache_pfield)
        end
    end
    return u',t # the transpose of u makes sure that ImplicitAD correctly interprets the contents of u
end

# Step forward in time. This function is essentially a wrapper for the actual vpm_step! function. This function still needs particle count tracking implemented and tested.
# There is also a large amount of unnecessary allocation here, since I'm allocating twice the size of the state vector on each time step, which single-handedly triples the memory usage.
function VPM_AD_solve(pfield_vec,settings,t,sim_settings;time_step_tol = 1e-6)

    #println("running AD solve")
    pfield = vec2pfield(pfield_vec,settings)
    i = -1
    len = length(settings[12])
    for j=1:len
        if abs(t - settings[12][j]) < time_step_tol
            i = j
        end
    end
    if i == -1
        error("Could not determine correct time step! Time: $t\tTime history: $(settings[12])")
    end
    pfield.np = settings[13][i]

    VPM_step!(pfield,t,sim_settings...)
    pfield_vec_out,settings = pfield2vec(pfield)
    return pfield_vec_out,settings

end

function VPM_forward_solve(pfield_vec,settings,t,sim_settings,pfield;ε=1e-6)

    pfield = vec2pfield(pfield_vec,settings)
    #=if length(pfield.t_hist) == 0 || pfield.t >= pfield.t_hist[end]
        push!(pfield.t_hist,t)
        push!(pfield.np_hist, pfield.np)
    end=#
    len = length(pfield.t_hist)
    if len >= 1
        i = -1
        for j=1:len
            if abs(pfield.t_hist[j] + 1) < ε
                i = j
                break
            end
        end
        pfield.np_hist[i] = pfield.np
        pfield.t_hist[i] = t
    end
    VPM_step!(pfield,t,sim_settings...)
    
    pfield_vec_out = pfield2vec(pfield,settings)
    return pfield_vec_out,settings

end

function VPM_step!(pfield,t,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name;time_step_tol = 1e-6)

    i = -1
    len = length(pfield.t_hist)
    for j=1:len
        if abs(t - pfield.t_hist[j]) < time_step_tol
            i = j - 1 # initial time step is zero so the -1 makes i line up with that.
            break
        end
    end
    pfield.np = pfield.np_hist[i+1]
    if i == -1
        error("Could not determine correct time step! Time: $t\tTime history: $(pfield.t_hist)")
    end
    if i%verbose_nsteps==0
        vprintln("Time step $(Int(i)) out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
    end

    # Relaxation step # disabled because relaxation was causing weird type errors.
    #=relax = pfield.relaxation != relaxation_none &&
            pfield.relaxation.nsteps_relax >= 1 &&
            i>0 && (i%pfield.relaxation.nsteps_relax == 0)=#
    org_np = get_np(pfield)
    if i !== 0
        # Time step
        # Add static particles
        remove = static_particles_function(pfield, pfield.t, dt)

        # Step in time solving governing equations
        #nextstep(pfield, dt; relax=relax)
        nextstep(pfield, dt)
        # relaxation moved up a level, although this is really just a placeholder. It might end up disabled anyway.
        #if relax
        #    pfield.relaxation(p)
        #end

        # Remove static particles (assumes particles remained sorted)
        if remove==nothing || remove
            for pi in get_np(pfield):-1:(org_np+1)
                remove_particle(pfield, pi)
            end
        end
    else
        if length(pfield.t_hist) == 0
            pfield.nt += 1
        end
    end

     # Calls user-defined runtime function
    breakflag = runtime_function(pfield, t, dt;
                                vprintln= (str)-> i%verbose_nsteps==0 ?
                                vprintln(str, v_lvl+2) : nothing)

    # Save particle field. It only runs on the initial solve and not on successive AD calls.
    if eltype(pfield) <: AbstractFloat
        if save_path!=nothing && (i%nsteps_save==0 || i==nsteps || breakflag)
            overwrite_time = save_time ? nothing : pfield.nt
            save(pfield, run_name; path=save_path, add_num=true,
                                overwrite_time=overwrite_time)

        end
    end

    # User-indicated end of simulation # Currently disabled because the time stepping runs in its own function now instead of a loop.
    if breakflag
        #break
    end
    #return pfield
    
end