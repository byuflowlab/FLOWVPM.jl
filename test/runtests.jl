# activate test environment
# if splitpath(Base.active_project())[end-1] == "FLOWVPM"
#     import TestEnv
#     TestEnv.activate()
# end

using Test
import FLOWVPM
# using FLOWVPM.CUDA

# Run tests on CPU
const test_using_GPU = fill(0)
include("runtests_singlevortexring.jl")
include("runtests_leapfrog.jl")
include("runtests_merging.jl")
include("runtests_filament_edge_graph.jl")
include("runtests_filament_calibration.jl")
include("runtests_subfilterscale.jl")
include("runtests_vorticity_storage.jl")
include("runtests_relaxation_filter.jl")
include("runtests_expint.jl")

# Run tests on GPU if device is functional
# if CUDA.functional()
#     test_using_GPU[] = 1
#     include("runtests_singlevortexring.jl")
#     include("runtests_leapfrog.jl")
# end
