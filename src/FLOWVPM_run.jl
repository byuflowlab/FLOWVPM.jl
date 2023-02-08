# This file contains the code that runs the VPM. It contains code that used to reside in FLOWVPM_utils.jl. That file was split in two to keep the code clean.
# Added by Eric Green on 2 June 2022

# First, the original run_vpm!() function that Eduardo wrote:

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
* `nsteps_relax::Int`   : Relaxes the particle field every this many time steps.
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
function run_vpm!(pfield::ParticleField, dt::Real, nsteps::Int;
                      # RUNTIME OPTIONS
                      runtime_function::Function=runtime_default,
                      static_particles_function::Function=static_particles_default,
                      nsteps_relax::Int64=-1,
                      # OUTPUT OPTIONS
                      save_path::Union{Nothing, String}=nothing,
                      create_savepath::Bool=true,
                      run_name::String="pfield",
                      save_code::String="",
                      nsteps_save::Int=1, prompt::Bool=true,
                      verbose::Bool=true, verbose_nsteps::Int=10, v_lvl::Int=0,
                      save_time=true)

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
                                                    dt, nsteps_relax, nsteps_save,
                                                    runtime_function,
                                                    static_particles_function, v_lvl)

    # RUN
    for i in 0:nsteps

        if i%verbose_nsteps==0
            vprintln("Time step $i out of $nsteps\tParticles: $(get_np(pfield))", v_lvl+1)
        end

        # Relaxation step
        relax = pfield.relax && (nsteps_relax>=1 && i>0 && i%nsteps_relax==0)

        org_np = get_np(pfield)

        # Time step
        if i!=0
            # Add static particles
            static_particles_function(pfield, pfield.t, dt)

            # Step in time solving governing equations
            nextstep(pfield, dt; relax=relax)

            # Remove static particles (assumes particles remained sorted)
            for pi in get_np(pfield):-1:(org_np+1)
                remove_particle(pfield, pi)
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

    # Finalize verbose
    if verbose
        finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
    end

    return nothing
end

# Next my revised version:

"""
    This provides a new interface for using time integration schemes from DifferentialEquations.jl. Additional methods were defined for operations
    on particles and particles fields. Addition and scalar multiplication were both needed; the implementation should be fully broadcastable.
    Arbitrary operations can be run on particle fields because they just broadcast those operations to their interior particles, but only addition
    and scalar multiplication/division are defined for particles. If other operations are needed, add more methods for particles.

    The function structure is kept close to the original run_vpm!; however, the actual solving is offloaded to DifferentialEquations.jl
"""

using DifferentialEquations
diffeq = DifferentialEquations
using DiffEqSensitivity
des = DiffEqSensitivity
using DiscreteAdjoint
disadj = DiscreteAdjoint

### TODO: Re-implement relaxation

function run_vpm_alternate_time_marching!(pfield::ParticleField, dt::Real, nsteps::Int;
    # RUNTIME OPTIONS
    mode="forwards",
    parameters=nothing,
    static_particles_function::Function=static_particles_default,
    nsteps_relax::Int64=-1,
    AD=true,
    new_particle_times=nothing,
    # OUTPUT OPTIONS
    #save_path::Union{Nothing, String}=nothing,
    create_savepath::Bool=true,
    save_parameters=nothing,
    #run_name::String="pfield",
    save_code::String="",
    nsteps_save::Int=1, prompt::Bool=true,
    verbose::Bool=true, verbose_nsteps::Int=10, v_lvl::Int=0,
    save_time=true,
    return_sol=false,
    nps=nps_default(pfield.np),
    init_t = init_t_default,
    init_f = init_f_default)

    save_path = save_parameters[1]
    run_name = save_parameters[2]

    ### I'm leaving these checks in place for now, even if their relevant functionality is disabled. This may change later.
    ### Update: This check currently triggers the error output, so it is disabled until I get the viscous scheme/kernel compatibility list updated.
    # ERROR CASES
    ## Check that viscous scheme and kernel are compatible
    #=compatible_kernels = _kernel_compatibility[typeof(pfield.viscous).name]

    if !(pfield.kernel in compatible_kernels)
        error("Kernel $(pfield.kernel) is not compatible with viscous scheme"*
        " $(typeof(pfield.viscous).name); compatible kernels are"*
        " $(compatible_kernels)")
    end=#

    if save_path!==nothing
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
   #= (line1, line2, run_id, file_verbose,
    vprintln, time_beg) = initialize_verbose(   verbose, save_path, run_name, pfield,
                                    dt, nsteps_relax, nsteps_save,
                                    runtime_function,
                                    static_particles_function, v_lvl)=#

    ## There are two options for the mode:
    # mode="forwards": runs the VPM through DE.jl solvers
    # mode="adjoint": runs adjoint computations and has some additional outputs
    
    # RUN
    if mode == "forwards" || mode == "adjoint"
        tspan = (0.0,dt*nsteps)
        #p = [1.0]
        #=if extra_numerical_parameters !== nothing
            num_param = [pfield.viscous.nu, extra_numerical_parameters...]
        else
            num_param = [pfield.viscous.nu]
        end
        if runtime_function !== nothing
            if discrete_CB_affect_functions !== nothing
                discrete_CB_condition_functions = [discrete_CB_condition_functions..., TrueCondition]
                discrete_CB_affect_functions = [discrete_CB_affect_functions..., runtime_function]
            else
                discrete_CB_condition_functions = [TrueCondition]
                discrete_CB_affect_functions = [runtime_function]
            end
        end=#
        
        #=p = SolverSettings{Float64}(;numerical_parameters = num_param,
                            string_parameters = [save_path, run_name])=#
                            #discrete_CB_condition_functions = discrete_CB_condition_functions,
                            #discrete_CB_affect_functions = discrete_CB_affect_functions,
                            #continuous_CB_condition_functions = continuous_CB_condition_functions,
                            #continuous_CB_affect_functions = continuous_CB_affect_functions)
        #=p = SolverSettings{Float64}(;numerical_parameters = num_param,
                            string_parameters = [save_path, run_name],
                            kernel=pfield.kernel,UJ=pfield.UJ,
                            formulation=pfield.formulation,
                            viscous=pfield.viscous,
                            Uinf=pfield.Uinf,
                            sgsmodel=pfield.sgsmodel,
                            sgsscaling=pfield.sgsscaling,
                            integration=pfield.integration,
                            transposed=pfield.transposed,
                            nt=pfield.nt,
                            np=pfield.np,
                            maxparticles=pfield.maxparticles
                            )=#

        ### TODO: Add the new particle function and other function here. They will both need to be optional inputs to the run_vpm_alternate_time_marching function.
        p = SolverSettings{Float64}(;save_parameters = save_parameters,
                            kernel=pfield.kernel,UJ=pfield.UJ,
                            formulation=pfield.formulation,
                            viscous=pfield.viscous,
                            Uinf=pfield.Uinf,
                            sgsmodel=pfield.sgsmodel,
                            sgsscaling=pfield.sgsscaling,
                            integration=pfield.integration,
                            transposed=pfield.transposed,
                            nt=pfield.nt,
                            np=pfield.np,
                            maxparticles=pfield.maxparticles,
                            verbose=v_lvl
                            )
        #p.numerical_parameters = [1.0]

        ### TODO: Make callbacks work again with the new interface. Might need to base verbosity output off of verbosity setting.

        # all callbacks will be discrete for now
        # First: Save callback
        cb1 = DiscreteCallback(TrueCondition,SaveAffect!)
        # Second: Verbosity callback
        if verbose > 0
            cb2 = DiscreteCallback(TrueCondition,VerboseAffect!)
        else
            cb2 = DiscreteCallback(FalseCondition,NullAffect!)
        end

        #cbs = CallbackSet(cb1,cb2,cb3,cb4)
        #cbs = CallbackSet(cb2,cb3,cb4)
        cbs = nothing

        #cb1 = DiscreteCallback(DiscreteCBConditions,DiscreteCBAffects!)
        #cb2 = ContinuousCallback(ContinuousCBConditions,ContinuousCBAffects!)
        #=if discrete_CB_condition_functions !== nothing
            cb1s = Vector{DiscreteCallback}(undef,length(discrete_CB_affect_functions))
            for i=1:length(discrete_CB_affect_functions)
                cb1s[i] = DiscreteCallback(discrete_CB_condition_functions[i],discrete_CB_affect_functions[i],save_positions = (true,true))
            end
        else
            cb1s = nothing
        end
        if continuous_CB_condition_functions !== nothing
            
            cb2 = ContinuousCallback(continuous_CB_condition_functions,continuous_CB_affect_functions)
            #cb2 = VectorContinuousCallback(continuous_CB_condition_functions,continuous_CB_affect_functions,length(continuous_CB_affect_functions))
            if cb1s !== nothing
                cbs = CallbackSet(cb1s...,cb2)
            else
                cbs = CallbackSet(cb2)
            end
        else
            cbs = CallbackSet(cb1s...)
        end=#

        org_np = pfield.np
        #org_np = get_np(pfield)

        ### TODO: See if this is actually needed. Maybe they are used for regular evaluation points for simulation visualization? I should check the static_particles_function code.
        # Add static particles
        if parameters === nothing
            parameters = [1.0]
        end
        static_particles_function(pfield, pfield.t, dt)

        # Convert particle data into a vector. This avoids tricky vector interface issues and speeds up the simulation.
        u0 = zeros(length(pfield)+1)
        u0[1:end-1] .= pfield
        u0[end] = pfield.np
        println("particles: $(pfield.np)")
        #ODE_f = setup_diffeq(p;nps=nps,init_t=init_t,init_f=init_f,∆t=dt)
        ODE_f = setup_diffeq(p;∆t=dt,nps=nps,init_f=init_f)
        
        if mode == "forwards"

            prob = diffeq.ODEProblem(ODE_f,u0,tspan,parameters)
            sol = diffeq.solve(prob,p.integration,callback=cbs;dt=dt,save_on=false,alias_u0=true)
            dp = 0.0
            du0 = 0.0

        elseif mode == "adjoint"
            if pfield.UJ == UJ_fmm && AD == true
                @warn("Warning: C++-based FMM does not support AD! Setting autodiff to false.")
                AD = false
            end
            if return_sol == false
                @warn("Warning: sensitivity may fail if full solution is not returned by solver!")
            end

            prob = diffeq.ODEProblem(ODE_f,u0,tspan,parameters)
            tracked_cbs = disadj.setup_tracked_callbacks(CallbackSet(cbs),u0,p,tspan[1])
            @time sol = diffeq.solve(prob,p.integration,callback=tracked_cbs;adaptive=false,dt=dt,alias_u0=true,save_on=true)
            ts = sol.t
            function _dZ(_out,_u,p,t,i)

                if i == length(sol.t)
                    _np = get_int_np(_u[end])
                    for j=1:Int(_np)
                        _out[(j-1)*size(Particle) + 3] = one(eltype(_u))
                    end
                    _out ./= _np
                else
                    _np = get_int_np(_u[end])
                    for j=1:Int(_np)
                        _out[(j-1)*size(Particle) + 3] = zero(eltype(_u))
                    end
                    _out ./= _np
                end
            
            end

            dp,du0 = DiscreteAdjoint.discrete_adjoint(sol,_dZ,ts;cb=tracked_cbs,autojacvec = disadj.ReverseDiffVJP(false))
            #dp,du0 = DiscreteAdjoint.discrete_adjoint(sol,_dZ,ts;cb=tracked_cbs,autojacvec = disadj.ForwardDiffVJP())
        end
        # Remove static particles (assumes particles remained sorted)
        ### TEMP: Disabled
        #=for pi in get_np(p):-1:(org_np+1)
            remove_particle(pfield, pi)
        end=#

        # Calls user-defined runtime functions # this should be reformulated to run through callbacks (probably VectorContinuousCallback objects)
        #=breakflag = runtime_function(pfield, pfield.t, dt;
                        vprintln= (str)-> i%verbose_nsteps==0 ?
                                vprintln(str, v_lvl+2) : nothing)=#

        #save(pfield, run_name; path=save_path)

        # Finalize verbose ## no longer runs when verbose == false, since it outputs nothing in that case anyway
        #=if verbose
            finalize_verbose(time_beg, line1, vprintln, run_id, v_lvl)
        end=#
        return sol,du0,dp,sol[end]
        #=if mode=="forwards"
            if return_sol
                return sol
            else
                return nothing
            end
        end
        if mode=="adjoint"
            return sol,du0,dp,sol[end]
        end=#

    else

        error("VPM solver mode $mode not recognized!")

    end

    return nothing
end

function nps_default(np)

    nps_f(t) = np
    return nps_f

end

init_t_default(t) = false
init_f_default(pfield,t) = zeros(eltype(pfield),size(pfield))



# TODO:
# Clean up comments and dev code
# Make sure verbosity, UJ function, and Uinf function are properly passed in to the run_vpm!() function
#    UJ and Uinf will probably be inherited from the pfield that is passed in
#    The UJ function stuff might need some FMM data; this should also be passed through
# Make sure runtime functions have a reasonable interface
#    They should be inputs to the run_vpm!() function and be passed down to the actual time integration function
# Add more settings (and probably have them be optional inputs):
#    ODE solver algorithm
#    Adjoint algorithm
#    Use an initial dt? true/false
#    other settings passed to the solver i.e autodiff, what to save, etc
# Make sure the objective function is passed in
# Data about creating new particles will need to be passed in, especially if it should occur at regular intervals
#    If at regular intervals, there is a prebuilt callback to do it