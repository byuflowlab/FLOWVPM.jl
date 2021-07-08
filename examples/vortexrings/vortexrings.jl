#=##############################################################################
# DESCRIPTION
    Vortex ring simulations

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################

import FLOWVPM
vpm = FLOWVPM

import Printf: @printf
import Roots
import Cubature
import Elliptic
import LinearAlgebra: I
import DifferentialEquations

import CSV
import DataFrames
import PyPlot
import PyPlot: @L_str
plt = PyPlot


try
    # If this variable exist, we know we are running this as a unit test
    this_is_a_test
catch e
    nothing
end

# Plot formatting
plt.rc("font", family="Times New Roman") # Text font
plt.rc("mathtext", fontset="stix")       # Math font

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
ANNOT_SIZE = 10
plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title



header_path = splitdir(@__FILE__)[1]      # Path to this header

for header_name in ["functions", "simulation", "postprocessing"]
    include(joinpath( header_path, "vortexrings_"*header_name*".jl" ))
end
