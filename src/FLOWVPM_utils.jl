#=##############################################################################
# DESCRIPTION
    Utilities

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################



"""
  `run_vpm!(pfield, dt, nsteps; runtime_function=nothing, save_path=nothing,
run_name="pfield", nsteps_save=1, verbose=true, prompt=true)`

Solves `nsteps` of the particle field with a time step of `dt`.

**Optional Arguments**
* `runtime_function::Any`   : Give it a function of the form
                              `myfun(pfield, t, dt)`. On each time step it
                              will call this function. Use this for adding
                              particles, deleting particles, etc.
* `nsteps_relax::Int64` : Relaxes the particle field every this many time steps.
* `beta::Real`          : Maximum core size growth σ/σ_0 before reset.
* `sgm0::Real`          : Default core size upon reset.
* `save_path::Any`      : Give it a string for saving VTKs of the particle
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
                      runtime_function::Function=(pfield, t, dt)->false,
                      nsteps_relax::Int64=-1,
                      # RBF OPTIONS
                      beta::Real=1.5,
                      sgm0::Union{Nothing, Real}=nothing,
                      rbf_itmax::Int64=15, rbf_tol::Real=1e-3,
                      rbf_ign_iterror::Bool=false, rbf_verbose_warn::Bool=true,
                      # OUTPUT OPTIONS
                      save_path::Union{Nothing, String}=nothing,
                      run_name::String="pfield",
                      nsteps_save::Int64=1, prompt::Bool=true,
                      verbose::Bool=true, verbose_nsteps::Int64=10, v_lvl=0)

    run_id = save_path!=nothing ? joinpath(save_path,run_name) : ""

    # ERROR CASES
    if pfield.nu!=0 && sgm0==nothing
        error("Core Spreading activated but received no sgm0 parameter.")
    end

    # Creates save path
    if save_path!=nothing
        create_path(save_path, prompt)
    end

    # Core Spreading method
    cs_flag = pfield.nu!=0                  # CS activated flag
    t_sgm = 0.0                             # Time since last core size reset

    if verbose
        time_beg = Dates.DateTime(Dates.now())
        println("\t"^v_lvl*"*"^(73-8*v_lvl)*"\n"*"\t"^v_lvl*"START $run_id\t$time_beg\n"*
                "\t"^v_lvl*"*"^(73-8*v_lvl))
    end

    # RUN
    for i in 0:nsteps

        if verbose && i%verbose_nsteps==0
            println("\t"^(v_lvl+1)*"Time step $i out of $nsteps"*
            "\tParticles: $(get_np(pfield))")
        end

        # Relaxation step
        relax = pfield.relax && (nsteps_relax>=1 && i>0 && i%nsteps_relax==0)

        # Time step
        if i!=0
            nextstep(pfield, dt; relax=relax)
        end


        # # If Core Spreading activated, controls core sizes
        # if cs_flag
        #
        #     t_sgm += dt
        #
        #     # Case that core sizes grew bigger than beta*sgm0: Reset and run RBF
        #     if t_sgm >= sgm0^2*(beta^2-1)/(2*pfield.nu)
        #
        #         # Evaluates vorticity at each particle
        #         X_targ = [p.X for p in iterate(pfield)]
        #         # TODO: Make this more efficient
        #         omega_targ = [omega_approx(pfield, X) for X in X_targ]
        #
        #         # Resets core sizes
        #         for p in iterator(pfield)
        #             p.sigma[:] = sgm0
        #         end
        #
        #         # Calculates new strengths through RBF
        #         # TODO: Make this more efficient
        #         rbf_cg(pfield, omega_targ, X_targ; itmax=rbf_itmax, tol=rbf_tol,
        #         ign_iterror=rbf_ign_iterror, verbose_warn=rbf_verbose_warn)
        #
        #         t_sgm = 0
        #     end
        # end

        # Calls user-defined runtime function
        breakflag = runtime_function(pfield, pfield.t, dt)

        # Save
        if save_path!=nothing && (i%nsteps_save==0 || i==nsteps || breakflag)
            save(pfield, run_name; path=save_path, add_num=true)
        end

        # User-indicated end of simulation
        if breakflag
            break
        end

    end

    if verbose
        time_end = Dates.DateTime(Dates.now())
        hrs,mins,secs = timeformat(time_beg, time_end)
        println("\t"^v_lvl*"*"^(73-8*v_lvl))
        println("\t"^v_lvl*"END $run_id\t$time_end")
        println("\t"^v_lvl*"*"^(73-8*v_lvl))
        println("\t"^v_lvl*"ELAPSED TIME: $hrs hours $mins minutes $secs seconds")
    end

    return nothing
end


"""
  `save(pfield, file_name; path="")`

Saves the particle field in HDF5 format and a XDMF file especifying its the
attributes. This format can be opened in Paraview for post-processing and
visualization.
"""
function save(self::ParticleField{T}, file_name::String; path::String="",
                add_num::Bool=true, num::Int64=-1, createpath=false) where {T}

    if createpath; create_path(path, true); end;

    fname = file_name*(add_num ? num==-1 ? ".$(self.nt)" : ".$num" : "")
    h5fname = fname*".h5"
    np = get_np(self)

    # Creates/overwrites HDF5 file
    h5 = HDF5.h5open(joinpath(path, h5fname), "w")

    # Writes parameters
    h5["np"] = np
    h5["nt"] = self.nt
    h5["t"] = typeof(self.t) in [Float64, Int64] ? self.t : self.t.value

    # Writes fields
    # NOTE: It is very inefficient to convert the data structure to a matrices
    # like this. This could help to make it more efficient: https://stackoverflow.com/questions/58983994/save-array-of-arrays-hdf5-julia
    h5["X"] = [P.X[i] for i in 1:3, P in iterate(self)]
    h5["Gamma"] = [P.Gamma[i] for i in 1:3, P in iterate(self)]
    h5["sigma"] = [P.sigma[1] for P in iterate(self)]
    # h5["vol"] = [get_vol(self, pi) for pi in 1:np]
    h5["i"] = [P.index[1] for P in iterate(self)]

    # Connectivity information
    h5["connectivity"] = [i%3!=0 ? 1 : Int(i/3)-1 for i in 1:3*np]

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

        			  print(xmf, "\t\t\t\t<Time Value=\"", typeof(self.t) in [Float64, Int64] ? self.t : self.t.value, "\" />\n")

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

              # # Attribute: vol
              # print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
              #             " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
              #             " Name=\"vol\" Type=\"Scalar\">\n")
              #   print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
              #               " Dimensions=\"", np,
              #               "\" Format=\"HDF\" Precision=\"4\">",
              #               h5fname, ":vol</DataItem>\n")
              # print(xmf, "\t\t\t\t</Attribute>\n")


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
end



# """
#   `load_particlefield(pfield, h5_fname; path="")`
#
# Reads an HDF5 file containing particle data created with `save()` and adds
# all particles the the particle field `pfield`.
# """
# function load_particlefield(pfield::ParticleField, h5_fname::String;
#                             path::String="", load_time::Bool=false)
#
#   # Opens the HDF5 file
#   fname = h5_fname * (h5_fname[end-3:end]==".h5" ? "" : ".h5")
#   h5 = HDF5.h5open(joinpath(path, h5_fname), "r")
#
#   # Number of particles
#   np = load(h5["np"])
#
#   # Data
#   X = h5["X"]
#   Gamma = h5["Gamma"]
#   sigma = h5["sigma"]
#   vol = h5["vol"]
#
#   # Loads particles
#   for i in 1:np
#     p = Particle(X[1:3, i], Gamma[1:3, i], sigma[i], vol[i])
#     addparticle(pfield, p)
#   end
#
#   # Loads time stamp
#   if load_time
#     pfield.t = load(h5["t"])
#     pfield.nt = load(h5["nt"])
#   end
# end


"""
  `create_path(save_path::String, prompt::Bool)`

Create folder `save_path`. `prompt` prompts the user if true.
"""
function create_path(save_path::String, prompt::Bool)
  # Checks if folder already exists
    if isdir(save_path)
        if prompt
            inp1 = ""
            opts1 = ["y", "n"]
            while false==(inp1 in opts1)
                print("\n\nFolder $save_path already exists. Remove? (y/n) ")
                inp1 = readline()[1:end]
            end
            if inp1=="y"
                rm(save_path, recursive=true, force=true)
                println("\n")
            else
                return
            end
        else
            rm(save_path, recursive=true, force=true)
        end
    end
    mkdir(save_path)
end


function timeformat(time_beg, time_end)
    time_delta = Dates.value(time_end)-Dates.value(time_beg)
    hrs = Int(floor(time_delta/1000/60/60))
    mins = Int(floor((time_delta-hrs*60*60*1000)/1000/60))
    secs = Int(floor((time_delta-hrs*60*60*1000-mins*1000*60)/1000))
    return hrs,mins,secs
end
