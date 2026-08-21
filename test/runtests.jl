# activate test environment
# if splitpath(Base.active_project())[end-1] == "FLOWVPM"
#     import TestEnv
#     TestEnv.activate()
# end

using Test
import FLOWVPM

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

# Also run the GPU direct-sum kernel tests, if a functional CUDA-capable GPU
# is available. CUDA is an optional (weak) dependency of FLOWVPM -- it must
# be added to a *separate* environment (never FLOWVPM's own `--project=.`
# environment) for this to activate; otherwise this is a no-op and only the
# CPU tests above run, same as before.
cuda_functional = try
    import CUDA
    CUDA.functional()
catch
    false
end

if cuda_functional
    include("runtests_gpu.jl")
end

# Radix FMM coupling tests (task 034): Part A (host-resident transfer path)
# runs on CPU whenever the installed FastMultipole provides the radix device
# interface (self-skips with an @info otherwise); Part B (device-resident
# lifecycle) self-gates on functional CUDA (or requires it under
# FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1).
include("runtests_gpu_fmm.jl")
