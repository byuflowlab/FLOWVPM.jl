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
