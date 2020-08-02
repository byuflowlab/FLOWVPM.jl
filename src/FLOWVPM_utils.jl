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
  `save(pfield, file_name; path="")`

Saves the particle field in HDF5 format and a XDMF file especifying its the
attributes. This format can be opened in Paraview for post-processing and
visualization.
"""
function save(self::ParticleField, file_name::String; path::String="",
                add_num::Bool=true, num::Int64=-1, createpath=false)

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
    h5["X"] = [get_X(self, pi)[i] for i in 1:3, pi in 1:np]
    h5["Gamma"] = [get_Gamma(self, pi)[i] for i in 1:3, pi in 1:np]
    h5["sigma"] = [get_sigma(self, pi) for pi in 1:np]
    h5["vol"] = [get_vol(self, pi) for pi in 1:np]
    h5["i"] = [get_index(self, pi) for pi in 1:np]

    # Connectivity information
    h5["connectivity"] = [i%3!=0 ? 1 : Int(i/3)-1 for i in 1:3*np]

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
                            " Dimensions=\"", np,
                            "\" Format=\"HDF\" Precision=\"4\">",
                            h5fname, ":sigma</DataItem>\n")
              print(xmf, "\t\t\t\t</Attribute>\n")

              # Attribute: vol
              print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                          " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                          " Name=\"vol\" Type=\"Scalar\">\n")
                print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                            " Dimensions=\"", np,
                            "\" Format=\"HDF\" Precision=\"4\">",
                            h5fname, ":vol</DataItem>\n")
              print(xmf, "\t\t\t\t</Attribute>\n")


              # Attribute: index
              print(xmf, "\t\t\t\t<Attribute Center=\"Node\" ElementCell=\"\"",
                          " ElementDegree=\"0\" ElementFamily=\"\" ItemType=\"\"",
                          " Name=\"i\" Type=\"Scalar\">\n")
                print(xmf, "\t\t\t\t\t<DataItem DataType=\"Float\"",
                            " Dimensions=\"", np,
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
