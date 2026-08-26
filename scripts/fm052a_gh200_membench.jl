# 052a Phase C: GH200 memory-effectiveness measurement (no library changes).
# Measures (1) staging bandwidth pinned-vs-pageable at 1 MB..4 GB, H2D and D2H
# [covers plan items 1 and 4 in one sweep], (2) a one-shot S-sized (10.805 GB)
# upload wall time, (3) post-upload free HBM with the S-sized array resident.
# Output: CSV rows to FM052A_MEMBENCH_CSV plus a summary block on stdout.
using CUDA, Printf, Statistics

const CSV_PATH = get(ENV, "FM052A_MEMBENCH_CSV", "fm052a_membench.csv")
const S_BYTES = round(Int, 10.805e9)          # 052 one-shot S upload size
const SWEEP_MB = [1, 4, 16, 64, 256, 1024, 4096]
const REPS = 10; const WARMUP = 2

CUDA.functional() || error("CUDA not functional")
dev = CUDA.device()
println("device = ", CUDA.name(dev))

# CUDA.Mem.info() is gone in the cuda63-era package split; MemoryInfo lives
# (unexported) in CUDACore there (same fallback as FLOWPanel_solver.jl:424)
function _mem_info()
    if isdefined(CUDA, :Mem) && isdefined(CUDA.Mem, :info)
        return CUDA.Mem.info()
    end
    T = isdefined(CUDA, :MemoryInfo) ? CUDA.MemoryInfo : CUDA.CUDACore.MemoryInfo
    info = Base.invokelatest(T)
    return (Int(info.free_bytes), Int(info.total_bytes))
end

free0, total0 = _mem_info()
@printf "hbm_total_gib = %.3f\nhbm_free_initial_gib = %.3f\n" total0/2^30 free0/2^30

function bw_mbs(nbytes, pinned::Bool, h2d::Bool)
    n = nbytes ÷ 8
    h = rand(Float64, n)
    pinned && CUDA.pin(h)
    d = CuArray{Float64}(undef, n)
    ts = Float64[]
    for r in 1:(WARMUP + REPS)
        t = CUDA.@elapsed begin
            h2d ? copyto!(d, h) : copyto!(h, d)
        end
        r > WARMUP && push!(ts, t)
    end
    CUDA.unsafe_free!(d)
    med = median(ts)
    (nbytes / med) / 1e6, med
end

open(CSV_PATH, "w") do io
    println(io, "test,size_bytes,pinned,direction,median_s,mb_per_s")
    for mb in SWEEP_MB, pinned in (false, true), h2d in (true, false)
        nbytes = mb * 2^20
        mbs, med = bw_mbs(nbytes, pinned, h2d)
        dirs = h2d ? "h2d" : "d2h"
        println(io, "sweep,$nbytes,$pinned,$dirs,$med,$mbs")
        @printf "sweep %5d MB pinned=%-5s %s : %10.1f MB/s (median of %d)\n" mb pinned dirs mbs REPS
        flush(io); flush(stdout)
    end

    # one-shot S-sized upload (pinned, as the library's _pin_host_array path does)
    n = S_BYTES ÷ 8
    h = rand(Float64, n)
    CUDA.pin(h)
    d = CuArray{Float64}(undef, n)
    t_first = CUDA.@elapsed copyto!(d, h)      # includes any first-touch cost
    t_re = CUDA.@elapsed copyto!(d, h)
    println(io, "s_upload,$S_BYTES,true,h2d,$t_first,$((S_BYTES/t_first)/1e6)")
    println(io, "s_upload_repeat,$S_BYTES,true,h2d,$t_re,$((S_BYTES/t_re)/1e6)")
    @printf "s_upload one-shot %.3f GB: %.3f s (%.1f MB/s); repeat %.3f s (%.1f MB/s)\n" S_BYTES/1e9 t_first (S_BYTES/t_first)/1e6 t_re (S_BYTES/t_re)/1e6

    # capacity probe with S resident
    free1, _ = _mem_info()
    println(io, "free_hbm_with_s_resident,$free1,,,,")
    @printf "hbm_free_with_s_resident_gib = %.3f\n" free1/2^30
    CUDA.unsafe_free!(d)
end
println("membench complete; csv = ", CSV_PATH)
