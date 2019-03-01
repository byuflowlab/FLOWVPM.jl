"""
  Implementation of the three-dimensional viscous Vortex Particle Method written in Julia 1.0.

  # AUTHORSHIP
    * Author    : Eduardo J Alvarez
    * Email     : Edo.AlvarezR@gmail.com
    * Created   : Feb 2019
    * Copyright : Eduardo J Alvarez. All rights reserved.
"""
module FLOWVPM

# ------------ EXPOSED FUNCTIONS -----------------------------------------------
# export Particle, stretching, vorticity
# export ParticleField, addparticle, nextstep
# export Kernel, kernel_sing, kernel_gauserf, kernel_gaus, kernel_wnklmns
# export UJ_directAD, UJ_direct, UJ_directSym, UJ_fmm
# export U, omega_approx
# export FMM, Cell, buildTree, init_cells, upwardPass, horizontalPass, downwardPass
# export run_vpm!, save, load_particlefield

# ------------ GENERIC MODULES -------------------------------------------------
import Dates
import HDF5
import SpecialFunctions
import ForwardDiff
FD = ForwardDiff
using LinearAlgebra
using Distributed

# MultiComplex operations
import Base.*
import Base.conj
import Base.real
import Base.imag
import Base.atan

# GPU capabilities
const GPUenabled = false                        # GPU modules enabled/disabled
if GPUenabled
    using CUDAnative
    using CuArrays
end

# ------------ FLOW MODULES ----------------------------------------------------
# None required

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]      # Path to this module

# ------------ HEADERS ---------------------------------------------------------
for header_name in ["kernel", "particlefield"]
  include(joinpath( module_path, "FLOWVPM_"*header_name*".jl" ))
end



# ------------ MODULE PROPERTIES -----------------------------------------------
# Constant values
const EPS = 1e-15                     # Epsilon
const H_CSDA = 1e-32                  # Complex step

# Available Kernels
# const kernel_sing = Kernel(zeta_sing, g_sing, dgdr_sing, g_dgdr_sing)
# const kernel_gauserf = Kernel(zeta_gauserf, g_gauserf, dgdr_gauserf, g_dgdr_gauserf)
# const kernel_gaus = Kernel(zeta_gaus, g_gaus, dgdr_gaus, g_dgdr_gaus)
# const kernel_wnklmns = Kernel(zeta_wnklmns, g_wnklmns, dgdr_wnklmns, g_dgdr_wnklmns)

end # END OF MODULE
