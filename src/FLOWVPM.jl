"""
# DESCRIPTION
    Implementation of the three-dimensional viscous Vortex Particle Method
    written in Julia 1.4.2.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : 2019
  * Copyright : Eduardo J Alvarez. All rights reserved. No licensing, use, or
        modification of this code is allowed without written consent.
"""
module FLOWVPM

# ------------ GENERIC MODULES -------------------------------------------------
import HDF5
import SpecialFunctions

# ------------ FLOW CODES ------------------------------------------------------
import FLOWExaFMM
const fmm = FLOWExaFMM

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]      # Path to this module

# ------------ HEADERS ---------------------------------------------------------
for header_name in ["particlefield", "utils"]
    include(joinpath( module_path, "FLOWVPM_"*header_name*".jl" ))
end

end # END OF MODULE
