#!/usr/bin/env julia
# Architecture compatibility probe. This is intentionally stricter than a
# package instantiate: every manifest JLL must load on the allocated CPU/GPU.

using InteractiveUtils
using LinearAlgebra
using Pkg
using TOML

length(ARGS) == 2 || error("usage: fm052_arch_probe.jl ARCH OUTPUT_TOML")
arch, output = ARGS
expected_arch = arch == "gh200" ? :aarch64 : :x86_64
Sys.ARCH == expected_arch || error("Julia CPU architecture mismatch: expected $expected_arch, observed $(Sys.ARCH)")
VERSION == v"1.11.7" || error("task 052 requires Julia 1.11.7 exactly; observed $VERSION")

using CUDA
CUDA.functional() || error("CUDA.jl is not functional: $(CUDA.functional(true))")
using FastMultipole
using FLOWPanel
using FLOWVPM
using VSPGeom

jll_loaded = String[]
jll_failures = String[]
for (uuid, package) in sort!(collect(Pkg.dependencies()); by=x -> something(last(x).name, ""))
    name = package.name
    name === nothing && continue
    endswith(name, "_jll") || continue
    try
        Base.require(Base.PkgId(uuid, name))
        push!(jll_loaded, name)
    catch err
        push!(jll_failures, "$name: $(sprint(showerror, err))")
    end
end
isempty(jll_failures) || error("JLL load failures:\n" * join(jll_failures, '\n'))

A = reshape(Float64.(1:16), 4, 4)
x = Float64.(1:4)
y = A * x
y == [90.0, 100.0, 110.0, 120.0] || error("Float64 BLAS probe failed: $y")

device = only(collect(CUDA.devices()))
runtime = string(CUDA.runtime_version())
driver = string(CUDA.driver_version())
blas_config = sprint(show, BLAS.get_config())

mkpath(dirname(output))
open(output, "w") do io
    println(io, "architecture = ", repr(arch))
    println(io, "status = \"pass\"")
    println(io, "julia_version = ", repr(string(VERSION)))
    println(io, "julia_cpu_architecture = ", repr(string(Sys.ARCH)))
    println(io, "julia_threads = ", Threads.nthreads())
    println(io, "cuda_runtime = ", repr(runtime))
    println(io, "cuda_driver = ", repr(driver))
    println(io, "cuda_device = ", repr(string(CUDA.name(device))))
    println(io, "cuda_compute_capability = ", repr(string(CUDA.capability(device))))
    println(io, "blas_vendor = ", repr(string(BLAS.vendor())))
    println(io, "blas_threads = ", BLAS.get_num_threads())
    println(io, "blas_config = ", repr(blas_config))
    println(io, "jll_count = ", length(jll_loaded))
    println(io, "jll_packages = ", repr(join(jll_loaded, ',')))
    println(io, "fastmultipole_path = ", repr(string(pathof(FastMultipole))))
    println(io, "flowpanel_path = ", repr(string(pathof(FLOWPanel))))
    println(io, "flowvpm_path = ", repr(string(pathof(FLOWVPM))))
    println(io, "vspgeom_path = ", repr(string(pathof(VSPGeom))))
end

println("architecture compatibility probe passed: $output")
versioninfo(verbose=true)
CUDA.versioninfo()
println("BLAS vendor=$(BLAS.vendor()) threads=$(BLAS.get_num_threads()) config=$blas_config")
