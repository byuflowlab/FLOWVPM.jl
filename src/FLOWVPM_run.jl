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
function run_vpm!(pfield::PF, dt::Real, nsteps::Int;
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
                    save_time=true) where {PF <: ParticleField}

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
    # TODO: Make sure to have a settings to decide whether to use the unmodified solver process or the new one.
    #    This should make it easier to make sure that I didn't accidentally break anything.
    #solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)
    solve_vpm_implicitAD!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)

    # Finalize verbose
    finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)

    return nothing
end

function solve_vpm!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)

    for i in 0:nsteps

        if i%verbose_nsteps==0
            vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
        end

        # Relaxation step
        relax = pfield.relaxation != relaxation_none &&
                pfield.relaxation.nsteps_relax >= 1 &&
                i>0 && (i%pfield.relaxation.nsteps_relax == 0)

        org_np = get_np(pfield)

        # Time step
        if i!=0
            # Add static particles
            remove = static_particles_function(pfield, pfield.t, dt)

            # Step in time solving governing equations
            nextstep(pfield, dt; relax=relax)
            # relaxation moved up a level, although this is really just a placeholder. It might end up disabled anyway.
            #if relax
            #    pfield.relaxation(p)
            #end
            #solveW(_pfield) = nextstep(_pfield, dt; relax=relax)
            #residual = 
            #ImplicitAD.implicit_unsteady(solveW, residual, pfield, ())

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
            # If issues come up related to trying to autodifferentiate the save function, this should fix it.
            #=wsave(_pfield,_p) = save(_pfield, run_name; path=save_path, add_num=true, overwrite_time=overwrite_time)
            null_J(x,p) = 0.0
            ImplicitAD.provide_rule(wsave,pfield,();mode="jacobian",jacobian=null_J)=#

        end

    # User-indicated end of simulation
        if breakflag
            break
        end

    end
    
end

#using ImplicitAD

function solve_vpm_implicitAD!(pfield,nsteps,dt,verbose_nsteps,static_particles_function,runtime_function,v_lvl,vprintln,save_path,nsteps_save,save_time,run_name)

    t = range(0.0,dt*nsteps,length=nsteps)
    pfield_vec = get_state_vector(pfield)
    #dpfield = similar(pfield_vec)
    resid = create_time_evolution_residual(pfield)
    np = pfield.np

    for i=1:nsteps

        if i%verbose_nsteps==0
            vprintln("Time step $i out of $nsteps\tParticles: $(np)", v_lvl+1)
        end

        #org_np = get_np(pfield_vec)

        # Time step
        if i!=1
            # Add static particles
            # Currently disabled but might need to be re-enabled.
            #remove = static_particles_function(pfield, pfield.t, dt)

            # Step in time solving governing equations
            #pfield_vec = ImplicitAD.implicit(nextstep,resid(np),pfield_vec,(dt,resid(np)))
            tspan = (t[i-1],t[i])
            wEulerStep(_pfield,_p) = EulerStep(resid(np),_pfield,_p,tspan,dt)
            _t,pfield_vec = ImplicitAD.implicit(wEulerStep,resid(np),pfield_vec,())
            #nextstep(pfield, dt)
            #EulerStep(f,u0,p,tspan;dt=1.0)
            
            # relaxation moved up a level, although this is really just a placeholder. It might end up disabled anyway.
            #relax = pfield.relaxation != relaxation_none && pfield.relaxation.nsteps_relax >= 1 && i>1 && (i%pfield.relaxation.nsteps_relax == 0)
            #if relax
            #    pfield.relaxation(p)
            #end
            #solveW(_pfield) = nextstep(_pfield, dt; relax=relax)
            #residual = 
            #ImplicitAD.implicit_unsteady(solveW, residual, pfield, ())

            # Remove static particles (assumes particles remained sorted)
            #=if remove==nothing || remove
                for pi in get_np(pfield):-1:(org_np+1)
                    remove_particle(pfield, pi)
                end
            end=#
        end

        # Calls user-defined runtime function
         # Will need a wrapper function that converts the pfield call to a vector call
        #=breakflag = runtime_function(pfield, pfield.t, dt;
                                    vprintln= (str)-> i%verbose_nsteps==0 ?
                                    vprintln(str, v_lvl+2) : nothing)=#

        # Save particle field
        if save_path!=nothing && (i%nsteps_save==0 || i==nsteps || breakflag)
            overwrite_time = save_time ? nothing : pfield.nt
            save(pfield, run_name; path=save_path, add_num=true,
                                   overwrite_time=overwrite_time)
        end

    # User-indicated end of simulation
        #if breakflag
        #    break
        #end

    end

end