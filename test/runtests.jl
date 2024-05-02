# activate test environment
if splitpath(Base.active_project())[end-1] == "FLOWVPM"
    import TestEnv
    TestEnv.activate()
end

using Test
import FLOWVPM

include("runtests_singlevortexring.jl")
include("runtests_leapfrog.jl")
