using ForwardDiff
using ReverseDiff
using ImplicitAD
using FLOWVPM
using Plots
vpm = FLOWVPM

function duct_sim_2(inputs;save_path=nothing,nsteps=3,ctot=6,new_particle_interval=1,
    Nphi=40, Re=100000, ε=1e-6,input_count = 4)

    #xd = [circulation,U0,chord,span]
    xd = inputs[1:4]
    #xc = zeros((1,nsteps+1))
    #xc = zeros(eltype(xd),1,nsteps+1)
    #T = eltype(xd)

    num_particles_array = zeros(Int,nsteps+2)
    maxp_sim = 200 # max particle count for simulation; larger amounts are liable to cause memory overflows.
    for i=1:min(Int(ceil(maxp_sim/Nphi)),nsteps+2)
        num_particles_array[i] = Nphi*(i-1)
    end
    for i=Int(ceil(maxp_sim/Nphi))+1:nsteps+2
        num_particles_array[i] = Nphi*Int(ceil(maxp_sim/Nphi))
    end
    xc = reshape([inputs[5:end]...],input_count,nsteps+1)
    T = promote_type(eltype(xd),eltype(xc))

    sigma = 1.5*inputs[4]/Nphi ### kind of arbitrary... but there should be a way to calculate the correct value.
    O = zeros(T,3)
    Oaxis = T[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    beta = T(999)
    faux = T(1.0)
    nu = inputs[2]*2*inputs[3]/Re

    #Uinf(t::T) where {T} = [0.0,0.0,2.5]
    Uinf(t) = [0.0;0.0;20.0]

    Uref = inputs[2]
    dt = 0.0075/2 # ctot/Uref/nsteps

    #num_particles(Nphi,new_ring_interval,_nsteps) = Int(Nphi*(ceil(_nsteps/new_particle_interval)))
    #np_f(t) = num_particles(Nphi,new_particle_interval,Int(round(t/dt)))
    get_step(t) = Int(round(t/dt))
    np_f(t) = num_particles_array[get_step(t)+1]

    solver = (
             UJ=vpm.UJ_direct,
             R = T,
             nsteps = nsteps,
             Uinf=Uinf,
             np_f = np_f
             #integration=vpm.euler
             )

    function init_new_particle_runtime_function()

        runtime_function = let Nphi=Nphi, sigma=T(sigma), T=T, new_particle_interval=new_particle_interval, ε=ε, num_particles_array=num_particles_array, input_count=input_count
            (pfield,t,dt,xc,xd;vprintln=nothing) -> begin

                circulation = xd[1]
                U0 = xd[2]
                chord = xd[3]
                span = xd[4]
                #=tmp = t/dt
                #println("tmp is $tmp")
                while abs(tmp) > ε && tmp > -ε
                    tmp -= new_particle_interval
                end
                if tmp < -ε
                    return false
                end=#
                #println("adding particles")
                X = zeros(T,3)
                Gamma = zeros(T,3)
                _sigma = zeros(T,1)
                for i=1:num_particles_array[get_step(t)+2] - num_particles_array[get_step(t)+1]
                    #i0 = pfield.np + 1
                    X[1] = (i - 1)*span/(Nphi-1) - span/2
                    X[2] = T(0.0)#xc[1]*X[1]^2#T(0.0)
                    X[3] = T(0.0)
                    #Gamma[1] = -xc[1]*xd[1]*sqrt(span/2 - X[1]^2) - 0.000001 # zero-magnitude gamma causes division by zero
                    Gamma[1] = circulation*cos(pi*X[1]/span - pi)#atan(circulation/((Nphi-1)/2 - i))#circulation/((Nphi-1)/2 - i)
                    Gamma[2] = T(0.0)#atan(circulation/((Nphi-1)/2 - i))
                    Gamma[3] = T(0.0)#circulation/((Nphi-1)/2 - i)
                    _sigma[1] = sigma
                    vpm.add_particle(pfield, X, Gamma, _sigma;circulation=circulation)
                end
                for i=1:input_count

                    i0 = (i-1)*1
                    #println(i)
                    pfield.particles[i].X[1] += xc[i0+1]
                    #pfield.particles[i].X[2] += xc[i0+2]
                    #pfield.particles[i].X[3] += xc[i0+3]
                    #pfield.particles[i].Gamma[1] += xc[i0+4]
                    #pfield.particles[i].Gamma[2] += xc[i0+5]
                    #pfield.particles[i].Gamma[3] += xc[i0+6]

                end
                return false
            end
        end
        return runtime_function

    end

    #maxp = num_particles(Nphi,new_particle_interval,nsteps)
    #maxp = nps(dt*nsteps)
    maxp = Int(Nphi*ceil(nsteps/new_particle_interval+1)) # hardcoded for now

    pfieldargs = solver

    pfield = vpm.ParticleField(maxp;pfieldargs...)

    prompt = false
    save_path = "/home/eric/VPM_derivatives/src/output"
    if save_path != nothing # save output for visualization
        #display(save_path)
        vpm.create_path(save_path,prompt)
    end

    calc_monitors = true
    if calc_monitors # runs monitor functions

    end

    f = init_new_particle_runtime_function()
    runtime_function = f

    # more monitor stuff goes here
    function monitors(args...;optargs...)
        return false
    end
    function runtime_function_default(args...; optargs...)
        return false
    end

    ### I need to make sure the runtime function is set correctly.
    # runtime function stuff goes here
    this_runtime_function(args...; optargs...) = monitors(args...; optargs...) || runtime_function(args...; optargs...)
    #this_runtime_function(args...; optargs...) = monitors(args...; optargs...)
    run_name = "wing"
    verbose = false
    v_lvl = 1
    verbose_nsteps = 1
    optargs = ()

    # I shouldn't need this initialization anymore.
    #new_particle_function(integrator) = init_duct_outflow(integrator.u,integrator.p,Nphi,nsteps,new_ring_interval)

    pfield_out = vpm.run_vpm(pfield,dt,nsteps;
                        save_path=save_path,
                        run_name = run_name,
                        runtime_function = this_runtime_function,
                        create_savepath=true,
                        prompt=prompt,
                        save_code = "",
                        nsteps_save = 1,
                        verbose=verbose, verbose_nsteps = verbose_nsteps, v_lvl=v_lvl,
                        save_time=true, use_implicitAD=true, xc=xc ,xd=xd
                        )

    return pfield_out

end

function run_duct_sim_2(inputs;nsteps,input_count)

    #circulation = inputs[1]
    #U0 = inputs[2]
    #chord = inputs[3]
    #span = inputs[4]
    save_path = "/home/eric/VPMAdjoint/output"
    pfield_out = duct_sim_2(inputs;save_path=save_path,nsteps=nsteps,input_count=input_count)
    return usquared(pfield_out)
    #return Lprime(pfield_out)
    #return pfield_out.particles[1].X

end

function run_duct_sim_2_standalone(nsteps)
    save_path = "/home/eric/VPMAdjoint/output"
    nsteps=5
    input_count=4
    inputs = [0.1,1.0,0.1,1.0,1e-4*ones(input_count*(nsteps+1))...]
    _f(_inputs) = run_duct_sim_2(_inputs;nsteps=nsteps,input_count=input_count)
    @time output_value = _f(inputs)
    return output_value
end

# This function needs to be rewritten to match current code.
function usquared(pfield)

    out = 0.0
    np = pfield.np
    pfield.UJ(pfield)
    for i=1:Int(np)
        out += sqrt(pfield.particles[i].U[1]^2 + pfield.particles[i].U[2]^2 + pfield.particles[i].U[3]^2)
    end
    out /= np
    #println(out)
    return out

end

function Lprime(pfield)

    out = 0.0
    np = pfield.np
    rho = 1.225
    Uinf = 10.0
    for i=1:Int(np)
        out += rho*Uinf*pfield.particles[i].Gamma[1]
    end
    out /= np
    return out

end

function calc_gradient(input_count,nsteps)

    inputs = [0.1,1.0,0.1,1.0,1e-4*ones(input_count*(nsteps+1))...]
    _f(_inputs) = run_duct_sim_2(_inputs;nsteps=nsteps,input_count=input_count)
    @time output_value = _f(inputs)
    @time forwarddiff_gradient = ForwardDiff.gradient(_f,inputs)    
    @time reversediff_gradient = ReverseDiff.gradient(_f,inputs)

    f_tape = ReverseDiff.GradientTape(_f,similar(inputs))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    cfg = ReverseDiff.GradientConfig(inputs)
    results = similar(inputs)
    #all_results = map(DiffResults.GradientResult, results)
    @time ReverseDiff.gradient!(results, compiled_f_tape, inputs)
    #return results
    #return output_value, forwarddiff_gradient,reversediff_gradient,results
    #return output_value, forwarddiff_gradient,reversediff_gradient
    return output_value, forwarddiff_gradient,results
    #return output_value,results

end

function get_rms_error(fd_grad,rd_grad)
    len = length(fd_grad)
    out = 0.0
    for i=1:len
        out += (fd_grad[i] - rd_grad[i])^2
    end
    out /= len
    out = sqrt(out)
    return out
end

function get_runtimes(max_inputs,nsteps)

    input_count_range = range(1,max_inputs,min(max_inputs,20))
    input_count = Int.(round.(input_count_range))
    rms_error = zeros(length(input_count))
    for i=1:length(input_count)
        println("Inputs: $(input_count[i]*(nsteps+1))")
        ov,fd_grad,rd_grad = calc_gradient(input_count[i],nsteps)
        rms_error[i] = get_rms_error(fd_grad,rd_grad)
    end
    return rms_error

end