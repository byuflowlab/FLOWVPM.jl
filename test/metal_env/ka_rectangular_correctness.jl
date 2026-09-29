# Correctness check for RectangularGaussianErfVortex on Metal: the device
# FastMultipole.direct_rectangular! (FastMultipole's KernelAbstractions
# extension, reaching FLOWVPM's rect_pair) against the threaded host method.
using FLOWVPM
using Metal
using KernelAbstractions
import Random
const FM = FLOWVPM.FastMultipole

Random.seed!(7)
k = FLOWVPM.RectangularGaussianErfVortex()
src = rand(Float32, 7, 400) .- 0.5f0; src[7, :] .= 0.05f0
tgt = rand(Float32, 3, 300) .- 0.5f0
nfail = 0
for grad in (false, true)
    rows = FM.rect_output_rows(grad)
    ref = zeros(Float32, rows, 300)
    FM.direct_rectangular!(ref, tgt, k, src; gradient=grad)
    out = MtlArray(zeros(Float32, rows, 300))
    FM.direct_rectangular!(out, MtlArray(tgt), k, MtlArray(src); gradient=grad)
    e = maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref))
    ok = e < 1e-5
    global nfail += !ok
    println(ok ? "PASS" : "FAIL", "  gradient=$grad relerr=$e")
end
nfail == 0 || error("RectangularGaussianErfVortex Metal check failed")
