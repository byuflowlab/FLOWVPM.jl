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

# TODO
* [x] Run leapfrogging test (I dind't seem to find the singuar transposed scheme to be more stable).
* [ ] Implement the kernel that the italian group used.
* [ ] Implement UJ_fmm.
* [ ] Incorporate viscous diffusion and RBF.
"""
module FLOWVPM

# ------------ GENERIC MODULES -------------------------------------------------
import HDF5
import SpecialFunctions
import Dates

# ------------ FLOW CODES ------------------------------------------------------
import FLOWExaFMM
const fmm = FLOWExaFMM

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]      # Path to this module

# Determine the floating point precision of ExaFMM
const exafmm_single_precision = fmm.getPrecision()
const RealFMM = exafmm_single_precision ? Float32 : Float64

# ------------ HEADERS ---------------------------------------------------------
for header_name in ["kernel", "fmm", "particle", "particlefield", "UJ",
                    "timeintegration", "utils"]
    include(joinpath( module_path, "FLOWVPM_"*header_name*".jl" ))
end

# Available Kernels
const kernel_sing = Kernel(zeta_sing, g_sing, dgdr_sing, g_dgdr_sing, 1, 1)
const kernel_gauserf = Kernel(zeta_gauserf, g_gauserf, dgdr_gauserf, g_dgdr_gauserf, 5, 1)
const kernel_gaus = Kernel(zeta_gaus, g_gaus, dgdr_gaus, g_dgdr_gaus, -1, 1)
const kernel_wnklmns = Kernel(zeta_wnklmns, g_wnklmns, dgdr_wnklmns, g_dgdr_wnklmns, 3, 1)

# Aliases
const kernel_singular = kernel_sing
const kernel_gaussianerf = kernel_gauserf
const kernel_gaussian = kernel_gaus
const kernel_winckelmans = kernel_wnklmns
const singular = kernel_sing
const gaussianerf = kernel_gauserf
const winckelmans = kernel_wnklmns

end # END OF MODULE
