using ReverseDiff
revdiff = ReverseDiff

function SolveODEfromParticleField!(pfield,dt,nsteps;save_path=nothing,run_name="default",adjoint=false,runtime_functions=nothing)

    # destructures a pfield into a set of vectors

    # States (10*np):
    # 1-3: X
    # 4-6: Γ
    # 7: σ
    # 8: vol
    # 9: circulation
    # 10: index (for fmm sorting?)

    # temporary storage (45*np):
    # 1-3: velocity
    # 4-12: J (column-descending order)
    # 13-15: PSE
    # 16-24: M (temporary memory for computations) (column-descending order)
    # 25-33: Jexa (column-descending order)
    # 34-42: dJdx1exa (column-descending order)
    # 43-51: dJdx2exa (column-descending order)
    # 52-60: dJdx3exa (column-descending order)

    # Why is reversediff failing?
    # There is an undefined reference somewhere on the backwards solution direction.
    # The ODE function is successfully called once or twice in the backwards direction.
    # The full solution is saved properly.
    # For now, I think I caught all the issues with variable type mismatches (i.e. TrackedArray vs. Array).
    # The correct checkpointing times are also passed
    # The code is also quite similar to my MWE
    # The objective function is not the issue either - the same error occurs when it just outputs a constant 1.0 (and it appears it doesn't even get to run)
    # Disabling checkpointing does not solve the issue
    # Setting the temporary memory to be a normal number type and then using TrackedArray.value does not solve the issue
    # Setting the temporary memory to be the same type as the input does not solve the issue
    # I tried commenting out every if statement (and commented out the UJ function entirely). It still has an undefined reference error
    # I also tried commenting out the entire function (!) and it still has the same behavior.
    # Removing the callbacks doesn't change the behavior.

    particle_state_size = 10
    temp_mem_size = 60
    total_size = particle_state_size+temp_mem_size

    tspan = [0, dt*nsteps]
    calcUinf = pfield.Uinf
    UJ = pfield.UJ
    if UJ == UJ_direct
        UJ = UJ_direct_vectorinputs!
    end
    sgsmodel = pfield.sgsmodel
    sgsscaling = pfield.sgsscaling
    kernel = pfield.kernel
    zeta0 = kernel.zeta(0)
    #numerical_parameters = [pfield.viscous.nu]
    numerical_parameters = [1.0]
    np = pfield.np
    g_dgdr = kernel.g_dgdr
    viscous = pfield.viscous
    transposed = pfield.transposed
    
    nt = 0

    #pfield_temp = zeros(pfield.np*temp_mem_size)
    pfield_state = zeros(pfield.np*particle_state_size)
    for i=1:np
        pfield_state[particle_state_size*(i-1)+1:particle_state_size*i-1] .= pfield.particles[i][1:9]
        pfield_state[particle_state_size*i] = pfield.particles[i].index[1]
    end
    #=pfield_state = zeros(pfield.np*total_size)
    for i=1:np
        pfield_state[total_size*(i-1)+1:total_size*(i-1)+9] .= pfield.particles[i][1:9]
        pfield_state[total_size*(i-1)+10] = pfield.particles[i].index[1]
    end
    temp_inds = zeros(Int,np*temp_mem_size)
    for i=1:np
        for j=1:temp_mem_size
            temp_inds[(i-1)*temp_mem_size+j] = total_size*(i-1) + particle_state_size + j
        end
    end=#

    #println(temp_inds)

    #println(np)
    #println(length(pfield.particles))
    #println(sizeof(pfield.particles))
    #println(sizeof(pfield_state))
    #error("")

    function DiffEQ_derivative_function_vec!(dpfield,pfield_vec,settings,t)

        #dpfield .= zero(eltype(dpfield))
        #dpfield .= zero(eltype(pfield_vec))

        #println("Derivative function evaluation start at time $t")

        ν = @view settings[1]

        # Reset U and J to zero
        #println("temp memory before reset at time $t")
        #pfield_temp = zeros(eltype(pfield_vec),np*temp_mem_size)
        pfield_temp = zeros(eltype(pfield_vec),np*temp_mem_size)
        #pfield_temp = zeros(np*temp_mem_size)
        #println("temp memory after reset at time $t")
        #pfield_temp .= 0
        #pfield_temp = @view pfield_vec[temp_inds]
        #for pft in pfield_temp
        #    pft = zero(pft)
        #end
    
        # Calculate interactions between particles: U and J
        #UJ(pfield_vec,pfield_vec,pfield_temp, np, particle_state_size, temp_mem_size, g_dgdr)
        #println("UJ function evaluation start at time $t")
        UJ(pfield_vec,pfield_vec,pfield_temp, np, particle_state_size, temp_mem_size, g_dgdr)
        #println("UJ function evaluation end at time $t")
    
        ### ### ### sgs currently disabled; to get it working the indiviual sgs functions will need to be rewritten to take vector inputs
        # Calculate subgrid-scale contributions
        ### Needs to be adjusted to handle vector inputs
        #_reset_particles_sgs(pfield)
        ### Needs to be adjusted to handle vector inputs
        #pfield.sgsmodel(pfield)
    
        # Calculate freestream
        #Uinf = calcUinf(t)
        Uinf = ones(eltype(pfield_vec),3)*1e-12

        for pval in pfield_vec
            pval === nothing ? error("nothing") : nothing
            isnan(pval) ? error("NaN") : nothing
            pval == undef ? error("undef") : nothing
        end
 
        # Update the particle field: convection and stretching
        for i=1:np
            p = @view pfield_vec[particle_state_size*(i-1)+1:particle_state_size*i]
            p_temp = @view pfield_temp[temp_mem_size*(i-1)+1:temp_mem_size*i]
            dp = @view dpfield[particle_state_size*(i-1)+1:particle_state_size*i]
    
            # sgs currently disabled
            #scl = sgsscaling(p, pfield)
    
            # Update position
            #dp.X .= p.U .+ Uinf
            dp[1] = p_temp[1] + Uinf[1]
            dp[2] = p_temp[2] + Uinf[2]
            dp[3] = p_temp[3] + Uinf[3]
    
            # Update vectorial circulation
            ## Vortex stretching contributions
            if transposed
                # Transposed scheme (Γ⋅∇')U
                dp[4] = p_temp[4]*p[4]+p_temp[5]*p[5]+p_temp[6]*p[6]
                dp[5] = p_temp[7]*p[4]+p_temp[8]*p[5]+p_temp[9]*p[6]
                dp[6] = p_temp[10]*p[4]+p_temp[11]*p[5]+p_temp[12]*p[6]
            else
                # Classic scheme (Γ⋅∇)U
                dp[4] = p_temp[4]*p[4]+p_temp[7]*p[5]+p_temp[10]*p[6]
                dp[5] = p_temp[5]*p[4]+p_temp[8]*p[5]+p_temp[11]*p[6]
                dp[6] = p_temp[6]*p[4]+p_temp[9]*p[5]+p_temp[12]*p[6]
            end
    
            # SGS currently disabled
            #dp[4] += scl*get_SGS1(p)*(p.sigma[1]^3/zeta0)
            #dp[5] += scl*get_SGS2(p)*(p.sigma[1]^3/zeta0)
            #dp[6] += scl*get_SGS3(p)*(p.sigma[1]^3/zeta0)

            dp[8] = 0.0
            dp[9] = 0.0
            dp[10] = 0.0

            dp[7] = ν[1]/p[7]
            if !isassigned(p,1)
                println("Particle $i is unassigned!\n")
            end
            if sum(isnan.(p)) > 0
                println("Particle $i has NaN!\n")
            end

            #if i == 1
            #    println("state 1 at time $t: $(pfield_vec[1])\t derivative: $(dpfield[1])\n")
            #end


        end

        #println("state 1 at time $t: $(pfield_vec[1])\t derivative: $(dpfield[1])\n")
        #println("pfield size is $(size(pfield_vec))")
        #println("dpfield size is $(size(dpfield))")
        #println("Derivative function evaluation end at time $t")
    end

    ###

    #test = zeros(np*particle_state_size)
    #DiffEQ_derivative_function_vec!(test,pfield_state,p,1.0)
    #println(test)

    #output = zeros(np*particle_state_size)
    #input = (pfield_state,p,[1.0])
    #input = pfield_state
    #f_wrap!(out,in) = DiffEQ_derivative_function_vec!(out,in,p,[1.0])
    #test2 = ReverseDiff.jacobian(f_wrap!,output,input,ReverseDiff.JacobianConfig(output, input))
    #println(size(test2))

    #error("!")

    ###

    function SaveCondition_vec(u,t,integrator)
        #println("Objective condition evaluated at time $t")
        (save_path !== nothing) ? true : false
    end
    
    function SaveAffect_vec!(integrator)
        nt += 1
        overwrite_time = integrator.t
        save(integrator.u, run_name; path=save_path, add_num=true,
                            overwrite_time=overwrite_time,np,nt)
    end
    VerboseCondition_vec(u,t,integrator) = true
    VerboseAffect_vec!(integrator) = println("Time: $(integrator.t)\tTimestep: $(nt)\tParticles: $(np)")

    function ring_centroid_location_adjcalc_vec(u,p,t)

        S = zeros(eltype(u),1)
        pfield_vec = u
        for i=1:np
            S += pfield_vec[particle_state_size*(i-1)+3]
        end
        S /= np
        return S
    
    end

    function dt_adjcalc_vec2(out,u,p,t,i)
        #println("Objective function evaluated at time $t")
        out = ones(length(out))
    end

    function dt_adjcalc_vec(u,p,t)
        #println("Objective function evaluated at time $t")
        return u[1] + p[1]
        #return 1.0
    end

    cb1 = DiscreteCallback(SaveCondition_vec,SaveAffect_vec!)
    cb2 = DiscreteCallback(VerboseCondition_vec,VerboseAffect_vec!)
    cbs = CallbackSet(cb1,cb2)

    if !adjoint
        prob = ODEProblem(DiffEQ_derivative_function_vec!,pfield_state,tspan,numerical_parameters)
        sol = diffeq.solve(prob,ORK256(),callback=cbs;dt=dt,save_on=true,alias_u0=true)
    else
        prob = ODEProblem(DiffEQ_derivative_function_vec!,pfield_state,tspan,numerical_parameters)
        #prob = ODEForwardSensitivityProblem(DiffEQ_derivative_function_vec!,pfield_state,tspan,numerical_parameters)
        #sol = diffeq.solve(prob,Tsit5(),callback=cbs;dt=dt,save_on=true,alias_u0=true)
        #save_t = [0:dt:dt*nsteps...]
        sol = diffeq.solve(prob,Tsit5(),callback=cbs;dt=dt,save_on=true,alias_u0=false)
        #sol = diffeq.solve(prob,Tsit5())
        #println(size(sol.u[29]))
        #println(sol[1,1])
        #println(sol[1,2])
        #ts = [0:dt:dt*nsteps...]
        #ts = save_t
        ts = sol.t
        du0,d_param = adjoint_sensitivities(sol,Tsit5(),ring_centroid_location_adjcalc_vec,nothing,nothing;dt=dt,sensealg=BacksolveAdjoint(;autodiff=false,autojacvec=false),checkpoints=ts)
        #du0,d_param = adjoint_sensitivities(sol,ORK256(),dt_adjcalc_vec,nothing,nothing;dt=dt,sensealg=BacksolveAdjoint(;autodiff=true,autojacvec=TrackerVJP()),checkpoints=ts)
        #du0,d_param = adjoint_sensitivities(sol,ORK256(),dt_adjcalc_vec,nothing,nothing;dt=dt,sensealg=ReverseDiffAdjoint(),checkpoints=ts)
        #du0,d_param = adjoint_sensitivities(sol,ORK256(),dt_adjcalc_vec2,ts;dt=dt,sensealg=ReverseDiffAdjoint())#,checkpoints=ts)
        #d_param,du0 = DiscreteAdjoint.discrete_adjoint(sol,dt_adjcalc_vec2,ts;autojacvec=DiscreteAdjoint.ReverseDiffVJP())
    end

    for i=1:np
        pfield.particles[i][1:3] .= sol[end][particle_state_size*(i-1)+1:particle_state_size*(i-1)+3]
        pfield.particles[i][4:6] .= sol[end][particle_state_size*(i-1)+4:particle_state_size*(i-1)+6]
        pfield.particles[i][7] = sol[end][particle_state_size*(i-1)+7]
        pfield.particles[i][8] = sol[end][particle_state_size*(i-1)+8]
        pfield.particles[i][9] = sol[end][particle_state_size*(i-1)+9]
    end

    if !adjoint
        return sol
    else
        return sol,du0,d_param
    end

end

function save(self::Array{T,1}, file_name::String; path::String="",
    add_num::Bool=true, num::Int64=-1, createpath::Bool=false,
    overwrite_time=nothing,np=-1,nt=0) where T

    if np <= 0
        np = Int(length(self)/70)
    end

    # Save a field with one dummy particle if field is empty
    if np==0
        dummy_pfield = ParticleField(1; nt=self.nt, t=self.t,
                                        formulation=formulation_classic)
        add_particle(dummy_pfield, (0,0,0), (0,0,0), 0)
        return save(dummy_pfield, file_name;
                path=path, add_num=add_num, num=num, createpath=createpath,
                overwrite_time=overwrite_time)
    end

    if createpath; create_path(path, true); end;

    fname = file_name*(add_num ? num==-1 ? ".$(nt)" : ".$num" : "")
    h5fname = fname*".h5"

    time = overwrite_time != nothing ? overwrite_time : error("overwrite time required for vector format")

    # Creates/overwrites HDF5 file
    h5 = HDF5.h5open(joinpath(path, h5fname), "w")

    # Writes parameters
    h5["np"] = np
    h5["nt"] = nt
    h5["t"] = time

    # Writes fields
    # NOTE: It is very inefficient to convert the data structure to a matrices
    # like this. This could help to make it more efficient: https://stackoverflow.com/questions/58983994/save-array-of-arrays-hdf5-julia
    #=h5["X"] = [P.X[i] for i in 1:3, P in iterate(self)]
    h5["Gamma"] = [P.Gamma[i] for i in 1:3, P in iterate(self)]
    h5["sigma"] = [P.sigma[1] for P in iterate(self)]
    h5["circulation"] = [P.circulation[1] for P in iterate(self)]
    h5["vol"] = [P.vol[1] for P in iterate(self)]
    h5["i"] = [P.index[1] for P in iterate(self)]=#
    h5["X"] = [get_X(self,P)[i] for i in 1:3, P in 1:Int(np)]
    h5["Gamma"] = [get_Gamma(self,P)[i] for i in 1:3, P in 1:Int(np)]
    h5["sigma"] = [get_sigma(self,P) for P in 1:Int(np)]
    h5["circulation"] = [get_circulation(self,P) for P in 1:Int(np)]
    h5["vol"] = [get_vol(self,P) for P in 1:Int(np)]
    h5["i"] = [get_index(self,P) for P in 1:Int(np)]

    # Connectivity information
    h5["connectivity"] = [i%3!=0 ? 1 : Int(i/3)-1 for i in 1:3*Int(np)]

    # # Write fields
    # dtype = HDF5.datatype(T)
    #
    # for (field, dim) in [("X", 3), ("Gamma", 3), ("sigma", 1)] # Iterate over fields
    #
    #     dims = dim==1 && false ? HDF5.dataspace(np) : HDF5.dataspace(dim, np)
    #     chunk = dim==1 && false ? (np,) : (1, np)
    #     dset = HDF5.d_create(h5, field, dtype, dims, "chunk", chunk)
    #
    #     for (pi, P) in enumerate(iterator(self))
    #         dset[:, pi] .= getproperty(P, Symbol(field))
    #     end
    #
    # end

    # # Connectivity information
    # dtype = HDF5.datatype(Int)
    # dims = HDF5.dataspace(3*np, 1)
    # chunk = (np, 1)
    # dset = HDF5.d_create(h5, "connectivity", dtype, dims, "chunk", chunk)
    # for i in 1:np
    #     dset[3*(i-1)+1, 1] = 1
    #     dset[3*(i-1)+2, 1] = 1
    #     dset[3*(i-1)+3, 1] = i-1
    # end

    close(h5)

    # Generates XDMF file specifying fields for paraview
    xmf = open(joinpath(path, fname*".xmf"), "w")

    # Open xmf block
    print(xmf, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n")
    print(xmf, "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"3.0\">\n")
    print(xmf, "\t<Domain>\n")
    print(xmf, "\t\t<Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n")
    print(xmf, "\t\t\t<Grid Name=\"particles\">\n")

            print(xmf, "\t\t\t\t<Time Value=\"", time, "\" />\n")

    # Nodes: particle positions
    print(xmf, "\t\t\t\t<Geometry Origin=\"\" Type=\"XYZ\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 3,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":X</DataItem>\n")
    print(xmf, "\t\t\t\t</Geometry>\n")

    # Topology: every particle as a point cell
    print(xmf, "\t\t\t\t<Topology Dimensions=\"", np, "\" Type=\"Mixed\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Int\"",
                    " Dimensions=\"", np*3,
                    "\" Format=\"HDF\" Precision=\"8\">",
                    h5fname, ":connectivity</DataItem>\n")
    print(xmf, "\t\t\t\t</Topology>\n")

    # Attribute: Gamma
    print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                " Name=\"Gamma\" Type=\"Vector\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 3,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":Gamma</DataItem>\n")
    print(xmf, "\t\t\t\t</Attribute>\n")

    # Attribute: sigma
    print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                " Name=\"sigma\" Type=\"Scalar\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 1,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":sigma</DataItem>\n")
    print(xmf, "\t\t\t\t</Attribute>\n")

    # Attribute: circulation
    print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                " Name=\"circulation\" Type=\"Scalar\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 1,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":circulation</DataItem>\n")
    print(xmf, "\t\t\t\t</Attribute>\n")

    # Attribute: vol
    print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                " Name=\"vol\" Type=\"Scalar\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 1,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":vol</DataItem>\n")
    print(xmf, "\t\t\t\t</Attribute>\n")


    # Attribute: index
    print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                " Name=\"i\" Type=\"Scalar\">\n")
        print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                    " Dimensions=\"", np, " ", 1,
                    "\" Format=\"HDF\" Precision=\"4\">",
                    h5fname, ":i</DataItem>\n")
    print(xmf, "\t\t\t\t</Attribute>\n")

    print(xmf, "\t\t\t</Grid>\n")
    print(xmf, "\t\t</Grid>\n")
    print(xmf, "\t</Domain>\n")
    print(xmf, "</Xdmf>\n")

    close(xmf)

    return fname*".xmf;"
end