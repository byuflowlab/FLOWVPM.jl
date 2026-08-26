# 052a Phase A (A2): x86-side aarch64 instantiation into the gh200 slug depot.
# Run with JULIA_DEPOT_PATH pointed at the slug depot. Platform tags match the
# official aarch64 Julia 1.11 binary (glibc, libgfortran5, cxx11). The cuda
# tags must be explicit: cross-platform instantiate does not run the JLLs'
# augmented-selection hooks, so without them CUDA_Runtime_jll/CUDA_Compiler_jll
# download nothing (observed on the first prepare attempt). Their aarch64
# artifacts are tagged cuda=<major.minor> plus cuda_platform=sbsa|jetson;
# GH200 Grace is server-class ARM, i.e. sbsa. allow_autoprecomp=false: only
# the ARM Julia may write compile caches into this depot.
using Pkg, Base.BinaryPlatforms

pin = get(ENV, "FP052_CUDA_PIN", "12.6")
platform = Platform("aarch64", "linux";
    libc = "glibc",
    libgfortran_version = "5.0.0",
    cxxstring_abi = "cxx11",
    julia_version = "1.11.7",
    cuda = pin,
    cuda_platform = "sbsa")

try
    Pkg.instantiate(; platform, allow_autoprecomp = false)
catch err
    # a fresh depot has no registry; instantiate from a complete Manifest
    # normally needs none, so add General only on demonstrated failure
    @warn "instantiate failed; adding the General registry to the slug depot and retrying" err
    Pkg.Registry.add("General")
    Pkg.instantiate(; platform, allow_autoprecomp = false)
end
