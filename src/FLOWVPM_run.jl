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
                    save_time=true, use_implicitAD=true, xc=nothing, xd=nothing) where {PF <: ParticleField}

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
        #save_settings(pfield, run_name; path=save_path) # TODO: Fix this.
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
        pfield_out = solve_vpm_implicitAD(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name,xc,xd)
        finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
        return pfield_out
    else
        solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name,xc,xd)
        finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
        return pfield
    end
    # Finalize verbose

    #return nothing
end

function solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name,xc,xd)

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
        breakflag = runtime_function(pfield, pfield.t, dt, xc, xd;
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
                              save_time, run_name, xc, xd)

    t = range(0,dt*nsteps,nsteps+1)
    pfield_vec, settings = pfield2vec(pfield)
    # sim_settings is a an array of type any that carries a bunch of information that the VPM needs that doesn't change through time.
    sim_settings = [nsteps, dt, verbose_nsteps,
                    static_particles_function,
                    runtime_function, v_lvl, vprintln,
                    save_path, nsteps_save, save_time,
                    run_name]
    function init(_t0,_xd,_xc0,_p)
        #T = promote_type(eltype(_xd), eltype(_xc0))
        #T = promote_type(eltype(_xd),eltype(pfield_vec))
        T = eltype(_xd)
        _pfield = vec2pfield(pfield_vec,settings,T)
        VPM_step!(_pfield,_t0,_xc0,_xd,sim_settings...)
        _pfield_vec,settings = pfield2vec(_pfield)
        println("pfield_vec type: $(typeof(_pfield_vec))")
        return _pfield_vec
    end
    function onestep!(y,yprev,_t,tprev,_xd,xci,p)
        T = promote_type(eltype(_xd), eltype(xci))
        _pfield = vec2pfield(yprev,settings,T)
        VPM_step!(_pfield,_t,xci,_xd,sim_settings...)
        _pfield_vec,settings = pfield2vec(_pfield)
        y .= _pfield_vec
    end
    #xd = [0.0]
    #=println(typeof(xd))
    println(typeof(xc))
    T = promote_type(eltype(xd), eltype(xc))
    y = zeros(T, length(pfield_vec), length(t))
    println(y[1])=#
    p = ()
    out = ImplicitAD.explicit_unsteady(init,onestep!,t,xd,xc, p; cache=nothing)
    pfield_out = vec2pfield(out[:,end],settings)
    return pfield_out

end

function VPM_step!(pfield,t,xc,xd,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name;time_step_tol = 1e-6)

    i = Int(round(t/dt))
    pfield.np = pfield.np_f(t)
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
    end

     # Calls user-defined runtime function
    breakflag = runtime_function(pfield, t, dt, xc, xd;
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

# Rewrite to account for changes in ImplicitAD and to allow variable particle counts:
# The function call is updated to match the new format.
# The particle count at each time is now passed in as a function.
# I've accounted for the way that parameters are passed in... mostly. It's a bit of a hack.
# I think that the whole initialization process could be passed in as a function to the VPM. Then the init function call would make more sense.
#    This approach would also save me the headache of trying to figure out which types should be inherited at different points during the solution process.