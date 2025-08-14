"""
# DESCRIPTION
    Implementation of the three-dimensional viscous Vortex Particle Method.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : 2019

# TODO
* [ ] Review time integration routines and SFS models to conform to new GE derivation.
* [ ] Remember to multiply SFS by p.sigma[1]^3/zeta0
* [ ] Save and read C for restart.
* [ ] Remove circulation property.
* [ ] Optimize creating of hdf5s to speed up simulations.
"""
module FLOWVPM

# ------------ GENERIC MODULES -------------------------------------------------
import HDF5
import BSON
import Dates
import Printf
import DataStructures: OrderedDict
import Roots
import SpecialFunctions: erf
import Base: getindex, setindex! # for compatibility with FastMultipole
# using ReverseDiff
using StaticArrays
# using CUDA
# using CUDA: i32
using Primes
using SyntheticEddy

# ------------ FLOW CODES ------------------------------------------------------
# import FLOWExaFMM
# const fmm = FLOWExaFMM
import FastMultipole

#------------- exports --------------------------------------------------------

export ParticleField,
       ClassicVPM, ReformulatedVPM,
       NoSFS, ConstantSFS, DynamicSFS,
       U_INDEX, J_INDEX,
       SIGMA_INDEX, GAMMA_INDEX, X_INDEX, GAMMA_INDEX,
       add_particle, remove_particle, run_vpm!

const fmm = FastMultipole

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]      # Path to this module
const examples_path  = joinpath(module_path, "..", "examples") # Path to examples
const utilities_path = joinpath(examples_path, "utilities") # Path to utilities
const utilities = joinpath(examples_path, "utilities", "utilities.jl") # Utilities

# Determine the floating point precision of ExaFMM
const FLOAT_TYPE = Float64

# miscellaneous constants
const const1 = 1/(2*pi)^1.5
const const2 = sqrt(2/pi)
const const3 = 3/(4*pi)
const const4 = 1/(4*pi)
const sqr2 = sqrt(2)

# ------------ HEADERS ---------------------------------------------------------
for header_name in ["kernel", "viscous", "formulation",
                    "relaxation", "subfilterscale", "inflow_turbulence",
                    "particlefield", "fmm",
                    # "particlefield", "gpu_erf", "gpu", "fmm",
                    "gpu_erf",
                    "UJ", "subfilterscale_models", "timeintegration",
                    "monitors", "utils"]# , "rrules"]
    include(joinpath( module_path, "FLOWVPM_"*header_name*".jl" ))
end


# ------------ AVAILABLE SOLVER OPTIONS ----------------------------------------

# ------------ Available VPM formulations
const formulation_classic = ClassicVPM{FLOAT_TYPE}()
const formulation_cVPM = ReformulatedVPM{FLOAT_TYPE}(0, 0)
const formulation_rVPM = ReformulatedVPM{FLOAT_TYPE}(0, 1/5)

"""
    formulation_tube_continuity
    
Alias for mass conserving tube formulation. Enforces conservation of mass 
for a vortex tube (f=1/2, g=0)
"""
const formulation_tube_continuity = ReformulatedVPM{FLOAT_TYPE}(1/2, 0)

"""
    formulation_tube_momentum
    
Alias for momentum conserving tube formulation. Enforces conservation 
of momentum for a vortex tube (f=1/4, g=1/4)
"""
const formulation_tube_momentum = ReformulatedVPM{FLOAT_TYPE}(1/4, 1/4)
const formulation_sphere_momentum = ReformulatedVPM{FLOAT_TYPE}(0, 1/5 + 1e-8)

# Formulation aliases
"""
    cVPM
    
Alias for the classic VPM formulation.
"""
const cVPM = formulation_cVPM

"""
    rVPM
    
Alias for the reformulated VPM formulation. Enforces conservation of mass 
and momentum for a spherical fluid element and is the default formulation 
(f=0, g=1/5)
"""
const rVPM = formulation_rVPM
const formulation_default = formulation_rVPM

const standard_formulations = ( :formulation_classic,
                                :formulation_cVPM, :formulation_rVPM,
                                :formulation_tube_continuity, :formulation_tube_momentum,
                                :formulation_sphere_momentum
                              )

# ------------ Available Kernels
const kernel_singular = Kernel(zeta_sing, g_sing, dgdr_sing, g_dgdr_sing)
const kernel_gaussian = Kernel(zeta_gaus, g_gaus, dgdr_gaus, g_dgdr_gaus)
const kernel_gaussianerf = Kernel(zeta_gauserf, g_gauserf, dgdr_gauserf, g_dgdr_gauserf)
const kernel_winckelmans = Kernel(zeta_wnklmns, g_wnklmns, dgdr_wnklmns, g_dgdr_wnklmns)
const kernel_default = kernel_gaussianerf

# Kernel aliases
"""
    singular

Alias for the singular kernel.
"""
const singular = kernel_singular

"""
    gaussian

Alias for the Gaussian kernel.
"""
const gaussian = kernel_gaussian

"""
    gaussianerf

Alias for the Gaussian error function kernel.
"""
const gaussianerf = kernel_gaussianerf

"""
    winckelmans

Alias for the Winckelmans kernel.
"""
const winckelmans = kernel_winckelmans

const standard_kernels = (:singular, :gaussian, :gaussianerf, :winckelmans)


# ------------ Available relaxation schemes
const relaxation_none = Relaxation((args...; optargs...)->nothing, -1, FLOAT_TYPE(0.0))
const relaxation_pedrizzetti = Relaxation(relax_pedrizzetti, 1, FLOAT_TYPE(0.3))
const relaxation_correctedpedrizzetti = Relaxation(relax_correctedpedrizzetti, 1, FLOAT_TYPE(0.3))

# Relaxation aliases
"""
    pedrizzetti
    
Alias for the Pedrizzetti relaxation scheme.
"""
const pedrizzetti = relaxation_pedrizzetti

"""
    correctedpedrizzetti
    
Alias for the corrected Pedrizzetti relaxation scheme. Is a modification 
to the pedrizzetti relaxation that preserves the vortex strength magnitude.
"""
const correctedpedrizzetti = relaxation_correctedpedrizzetti

"""
    norelaxation

Alias for the no relaxation scheme.
"""
const norelaxation = relaxation_none
const relaxation_default = pedrizzetti

const standard_relaxations = (:norelaxation, :pedrizzetti, :correctedpedrizzetti)

# ------------ Subfilter-scale models
# SFS procedure aliases
const pseudo3level_beforeUJ = dynamicprocedure_pseudo3level_beforeUJ
const pseudo3level_afterUJ = dynamicprocedure_pseudo3level_afterUJ
const pseudo3level_positive_afterUJ(args...; optargs...) = pseudo3level_afterUJ(args...; force_positive=true, optargs...)
const pseudo3level = (pseudo3level_beforeUJ, pseudo3level_afterUJ)
const pseudo3level_positive = (pseudo3level_beforeUJ, pseudo3level_positive_afterUJ)
const sensorfunction = dynamicprocedure_sensorfunction

# SFS Schemes
const SFS_none = NoSFS{FLOAT_TYPE}()

"""
    `SFS_Cs_nobackscatter = ConstantSFS(Estr_fmm; Cs=1.0, clippings=(clipping_backscatter,))`

Alias for the Constant SFS model with no backscatter.
"""
const SFS_Cs_nobackscatter = ConstantSFS(Estr_fmm; Cs=1.0, clippings=(clipping_backscatter,))

"""
    `SFS_Cd_twolevel_nobackscatter = DynamicSFS(Estr_fmm, pseudo3level_beforeUJ, pseudo3level_positive_afterUJ; alpha=0.999, clippings=(clipping_backscatter,))`

Alias for the Dynamic SFS model with two levels and no backscatter.
This is the recommended SFS model for high fidelity modeling.
"""
const SFS_Cd_twolevel_nobackscatter = DynamicSFS(Estr_fmm, pseudo3level_beforeUJ, pseudo3level_positive_afterUJ; alpha=0.999, clippings=(clipping_backscatter,))

"""
    `SFS_Cd_threelevel_nobackscatter = DynamicSFS(Estr_fmm, pseudo3level_beforeUJ, pseudo3level_positive_afterUJ; alpha=0.667, clippings=(clipping_backscatter,))`

Alias for the Dynamic SFS model with three levels and no backscatter.
This is similar to the two level version but uses a lower value of alpha (0.667).
"""
const SFS_Cd_threelevel_nobackscatter = DynamicSFS(Estr_fmm, pseudo3level_beforeUJ, pseudo3level_positive_afterUJ; alpha=0.667, clippings=(clipping_backscatter,))

# SFS aliases
"""
    `noSFS = NoSFS{FLOAT_TYPE}()`

Alias for the no subfilter-scale model.
"""
const noSFS = SFS_none
const SFS_default = SFS_none

const standard_SFSs = (
                        :SFS_none, :SFS_Cs_nobackscatter,
                        # :SFS_Cd_twolevel_nobackscatter,
                        # :SFS_Cd_threelevel_nobackscatter
                        )

# ------------ Other default functions
const nofreestream(t) = SVector{3,Float64}(0,0,0)
const Uinf_default = nofreestream
# const runtime_default(pfield, t, dt) = false
const monitor_enstrophy = monitor_enstrophy_Gammaomega
# const runtime_default = monitor_enstrophy
const runtime_default(pfield, t, dt; vprintln=nothing) = false
const static_particles_default(pfield, t, dt) = nothing


# ------------ Compatibility between kernels and viscous schemes
function _kernel_compatibility(viscous_scheme::Inviscid)
    return [singular, gaussian, gaussianerf, winckelmans,
    kernel_singular, kernel_gaussian,
    kernel_gaussianerf, kernel_winckelmans]
end

function _kernel_compatibility(viscous_scheme::CoreSpreading)
    return [gaussianerf, kernel_gaussianerf]
end

function _kernel_compatibility(viscous_scheme::ParticleStrengthExchange)
    return [gaussianerf, winckelmans,
    kernel_gaussianerf, kernel_winckelmans]
end


# ------------ INTERNAL DATA STRUCTURES ----------------------------------------

# Field inside the Particle type where the SFS contribution is stored (make sure
# this is consistent with ExaFMM and functions under FLOWVPM_subfilterscale.jl)
# const _SFS = :S

# ----- Instructions on how to save and print solver settings ------------------
# Settings that are functions
const _pfield_settings_functions = (:Uinf, :UJ, :integration, :kernel,
                                            :relaxation, :SFS, :viscous)

# Hash table between functions that are solver settings and their symbol
const _keys_standardfunctions = (:nofreestream, :UJ_direct, :UJ_fmm, :euler,
                                 :rungekutta3, standard_kernels...,
                                               standard_relaxations...,
                                               standard_SFSs...)
const _fun2key = Dict( (eval(sym), sym) for sym in _keys_standardfunctions )
const _key2fun = Dict( (sym, fun) for (fun, sym) in _fun2key )
const _standardfunctions = Tuple(keys(_fun2key))
const _key_userfun = Symbol("*userfunction")

# Hash table between standard options that are too lengthy to describe in print
const _keys_lengthyoptions = (standard_formulations..., standard_kernels...)
const _lengthy2key = Dict( (eval(sym), sym) for sym in _keys_lengthyoptions )
const _lengthyoptions = Tuple(keys(_lengthy2key))

# Relevant solver settings in a given particle field
const _pfield_settings = (sym for sym in fieldnames(ParticleField)
                          if !( sym in (:particles, :bodies, :np, :nt, :t, :M) )
                        )

# ------------------------------------------------------------------------------


export rVPM, cVPM,
       formulation_tube_continuity, formulation_tube_momentum
       singular, gaussian, gaussianerf, winckelmans,
       pedrizzetti, correctedpedrizzetti, norelaxation, 
       Inviscid, CoreSpreading, ParticleStrengthExchange,
       noSFS, SFS_Cs_nobackscatter, SFS_Cd_twolevel_nobackscatter,
       SFS_Cd_threelevel_nobackscatter, FMM
end # END OF MODULE
