### The ODE formatting and solving function was removed; it was essentially a dead end in the code development. However, I might grab the function for writing output to a file since it's designed to work with general vector inputs. At least, in theory - it may not have been working.

function save(self, s, file_name::String; path::String="",
    add_num::Bool=true, num::Int64=-1, createpath::Bool=false,
    overwrite_time=nothing,np=-1,nt=-1)
    
    # prevents saving on the reverse pass
    if !(typeof(s) <: SolverSettings)
        return nothing
    end

    if np <= 0
        np = Int(get_np(s))
    end
    if nt <= 0
        nt = get_nt(s)
    end
    
    # Save a field with one dummy particle if field is empty. The dummy settings struct makes sure that the particle count is 1 on the next run.
    if np==0
        dummy_pfield = ones(21)
        dummy_s = copy(s)
        dummy_s[1] = 1
        return save(dummy_pfield,dummy_s,file_name,path=path,add_num=add_num,num=num,createpath=createpath,overwrite_time=overwrite_time)
    end

    if createpath
        create_path(path, true)
    end

    fname = file_name*(add_num ? num==-1 ? ".$(nt)" : ".$num" : "")
    h5fname = fname*".h5"

    time = overwrite_time !== nothing ? overwrite_time : nt

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

    #=h5["X"] = [get_X(self,P)[i] for i in 1:3, P in 1:Int(np)]
    h5["Gamma"] = [get_Gamma(self,P)[i] for i in 1:3, P in 1:Int(np)]
    h5["sigma"] = [get_sigma(self,P) for P in 1:Int(np)]
    h5["circulation"] = [get_circulation(self,P) for P in 1:Int(np)]
    h5["vol"] = [get_vol(self,P) for P in 1:Int(np)]
    h5["i"] = [get_index(self,P) for P in 1:Int(np)]=#

    i0 = ([0:np-1...].*size(Particle)) # Starting indices for each particle (minus 1)

    h5["X"] = [self[i0[P]+i] for i in 1:3, P in 1:Int(np)]
    h5["Gamma"] = [self[i0[P]+i] for i in 4:6, P in 1:Int(np)]
    h5["sigma"] = [self[i0[P]+7] for P in 1:Int(np)]
    h5["circulation"] = [self[i0[P]+8] for P in 1:Int(np)]
    h5["vol"] = [self[i0[P]+9] for P in 1:Int(np)]
    h5["i"] = [P for P in 1:Int(np)]

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