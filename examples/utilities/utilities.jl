#=##############################################################################
# DESCRIPTION
    Utilities and examples for processing VPM simulations.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Oct 2021
=###############################################################################

import Printf: @printf

import GeometricTools as gt

import FLOWVPM as vpm

header_path = splitdir(@__FILE__)[1]      # Path to this header

for header_name in ["fluiddomain"]
    include(joinpath( header_path, "utilities_"*header_name*".jl" ))
end
