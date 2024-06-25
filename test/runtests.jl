# activate test environment
if splitpath(Base.active_project())[end-1] == "FLOWVPM"
    import TestEnv
    TestEnv.activate()
end

using Test
import FLOWVPM
using FLOWVPM.CUDA

const test_using_GPU = fill(false)
# include("runtests_singlevortexring.jl")
# include("runtests_leapfrog.jl")

if CUDA.functional()
    test_using_GPU[] = true
    # include("runtests_singlevortexring.jl")
    include("runtests_leapfrog.jl")
end
