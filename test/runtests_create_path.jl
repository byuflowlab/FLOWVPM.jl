using Test
import FLOWVPM

@testset "create_path never deletes silently" begin
    dir = mktempdir()
    path = joinpath(dir, "run")
    FLOWVPM.create_path(path, true)                      # new folder: created
    @test isdir(path)
    write(joinpath(path, "keep.txt"), "x")
    @test_throws ErrorException FLOWVPM.create_path(path, true)   # non-interactive: refuses
    @test isfile(joinpath(path, "keep.txt"))
    FLOWVPM.create_path(path, false)                     # explicit overwrite
    @test isdir(path) && isempty(readdir(path))
end
