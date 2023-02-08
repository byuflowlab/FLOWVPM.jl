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
* [ ] RBF testing: Point and ring test cases.
* [ ] Print and save setting at beginning of simulation.
* [ ] Feature of probing the fluid domain.
* [ ] Optimize creating of hdf5s to speed up simulations.
"""
module FLOWVPM

# ------------ GENERIC MODULES -------------------------------------------------
import HDF5
import JLD
import SpecialFunctions
import Dates
import Printf
import DataStructures: OrderedDict

# ------------ FLOW CODES ------------------------------------------------------
#import FLOWExaFMM
#const fmm = FLOWExaFMM

# ------------ GLOBAL VARIABLES ------------------------------------------------
const module_path = splitdir(@__FILE__)[1]      # Path to this module

### The ExaFMM part needs to be revised. For now it should be disabled; once Ryan finishes his VPM then it will likely interface with that instead.
# Determine the floating point precision of ExaFMM
#const exafmm_single_precision = fmm.getPrecision()
#const RealFMM = exafmm_single_precision ? Float64 : Float64
const RealFMM = Float64

# ------------ HEADERS ---------------------------------------------------------
for header_name in ["kernel", "viscous", "formulation",
                    "particle", "particlefield",
                    "UJ", "sgsmodels", "timeintegration",
                    "monitors", "utils", "settings", "run",
                    "fileIO"]
    include(joinpath( module_path, "FLOWVPM_"*header_name*".jl" ))
end


# ------------ AVAILABLE SOLVER OPTIONS ----------------------------------------
# Available VPM formulations
const formulation_classic = ClassicVPM{Float64}()
const formulation_tube_classic = ReformulatedVPM{Float64}(0, 0)
const formulation_tube_continuity = ReformulatedVPM{Float64}(1/2, 0)
const formulation_tube_momentum = ReformulatedVPM{Float64}(1/4, 1/4)
const formulation_tube = formulation_tube_continuity
const formulation_sphere_momentum = ReformulatedVPM{Float64}(0, 1/5)
const formulation_sphere = formulation_sphere_momentum
const formulation_reclassic = formulation_tube_classic
const formulation_default = formulation_sphere_momentum

const standard_formulations = (:formulation_classic, :formulation_tube_classic,
                               :formulation_tube_continuity, :formulation_tube_momentum,
                               :formulation_sphere_momentum)

# Available Kernels
const kernel_singular = Kernel(zeta_sing, g_sing, dgdr_sing, g_dgdr_sing, 1, 1)
const kernel_gaussian = Kernel(zeta_gaus, g_gaus, dgdr_gaus, g_dgdr_gaus, -1, 1)
const kernel_gaussianerf = Kernel(zeta_gauserf, g_gauserf, dgdr_gauserf, g_dgdr_gauserf, 5, 1)
const kernel_winckelmans = Kernel(zeta_wnklmns, g_wnklmns, dgdr_wnklmns, g_dgdr_wnklmns, 3, 1)
const kernel_default = kernel_gaussianerf

# Kernel aliases
const singular = kernel_singular
const gaussian = kernel_gaussian
const gaussianerf = kernel_gaussianerf
const winckelmans = kernel_winckelmans

const standard_kernels = (:singular, :gaussian, :gaussianerf, :winckelmans)

# Relaxation aliases
const pedrizzetti = relaxation_pedrizzetti
const correctedpedrizzetti = relaxation_correctedpedrizzetti
const norelaxation(args...) = nothing
const relaxation_default = pedrizzetti

const standard_relaxations = (:norelaxation, :pedrizzetti, :correctedpedrizzetti)

# Subgrid-scale models
const sgs_none(args...) = nothing
const sgs_stretching1_fmm_directionlow = generate_sgs_directionfiltered(generate_sgs_lowfiltered(sgs_stretching1_fmm))
const sgs_default = sgs_none

const standard_sgsmodels = (:sgs_none,
                            :sgs_stretching0_fmm,
                            :sgs_stretching1_direct, :sgs_stretching1_fmm,
                            :sgs_stretching2_direct, :sgs_stretching2_fmm,
                            # This one won't be recognized by save_settings due to scope of definition
                            # :sgs_stretching1_fmm_directionlow
                            )

# Subgrid-scale scaling functions
const sgs_scaling_none(args...) = 1
const sgs_scaling_default = sgs_scaling_none
const standard_sgsscalings = (:sgs_scaling_none, )

# Other default functions
const nofreestream(t) = zeros(3)
const Uinf_default = nofreestream
# const runtime_default(pfield, t, dt) = false
const monitor_enstrophy = monitor_enstrophy_Gamma2
const runtime_default = monitor_enstrophy
const static_particles_default(pfield, t, dt) = nothing


# Compatibility between kernels and viscous schemes
const _kernel_compatibility = Dict( # Viscous scheme => kernels
        Inviscid.body.name      => [singular, gaussian, gaussianerf, winckelmans,
                                        kernel_singular, kernel_gaussian,
                                        kernel_gaussianerf, kernel_winckelmans],
        CoreSpreading.body.name => [gaussianerf, kernel_gaussianerf],
        ParticleStrengthExchange.body.name => [gaussianerf, winckelmans,
                                        kernel_gaussianerf, kernel_winckelmans],
    )


# Default enstrophy monitor



# ------------ INTERNAL DATA STRUCTURES ----------------------------------------

# Field inside the Particle type where the SGS contribution is stored (make sure
# this is consistent with ExaFMM and functions under FLOWVPM_sgsmodels.jl)
const _SGS = :Jexa

# ----- Instructions on how to save and print solver settings ------------------
# Settings that are functions
const _pfield_settings_functions = (:Uinf, :UJ, :integration, :kernel,
                                            :relaxation, :sgsmodel, :sgsscaling, :viscous)

# Hash table between functions that are solver settings and their symbol
const _keys_standardfunctions = (:nofreestream, :UJ_direct, :UJ_fmm, :euler,
                                 :rungekutta3, standard_kernels...,
                                               standard_relaxations...,
                                               standard_sgsmodels...,
                                               standard_sgsscalings...)
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

end # END OF MODULE
