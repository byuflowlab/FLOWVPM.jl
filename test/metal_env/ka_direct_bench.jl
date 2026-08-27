# KernelAbstractions.jl port of the brute-force direct-sum kernel (mirrors
# ext/FLOWVPMMetalExt.jl's gpu_direct_kernel!, generalized to any KA backend)
# -- standalone script, does NOT touch FLOWVPM.gpu_direct! dispatch. This is
# a speed/correctness check only: does a backend-agnostic KA kernel cost
# anything vs the hand-written @metal kernel, before considering it as a
# replacement for the (tiled, H200-validated) CUDA gpu_atomic_square! path.
# NOTE: swapping CUDA over to a KA implementation is out of scope here --
# this machine has no CUDA GPU to check for regression against the existing
# ~800x-at-1M-particles CUDA baseline. This only validates the Metal side.
using FLOWVPM
using Metal
using KernelAbstractions
import Random

const KA = KernelAbstractions

@kernel function ka_direct_kernel!(out, @Const(s), n::Int32, kernel)
    j_target = @index(Global)
    if j_target <= n
        T = eltype(s)
        @inbounds tx = s[1, j_target]
        @inbounds ty = s[2, j_target]
        @inbounds tz = s[3, j_target]

        U1, U2, U3 = zero(T), zero(T), zero(T)
        J1, J2, J3, J4, J5, J6, J7, J8, J9 = zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T)

        const4 = T(0.25 / pi)

        i::Int32 = 1
        while i <= n
            @inbounds dX1 = tx - s[1, i]
            @inbounds dX2 = ty - s[2, i]
            @inbounds dX3 = tz - s[3, i]
            r2 = dX1^2 + dX2^2 + dX3^2
            r = sqrt(r2)

            @inbounds sigma = s[7, i]

            if r2 > zero(T) && abs(sigma) > zero(T)
                c4 = -const4 / (r*r2)
                @inbounds gam1 = c4 * s[4, i]
                @inbounds gam2 = c4 * s[5, i]
                @inbounds gam3 = c4 * s[6, i]

                g_sgm, dg_sgmdr = kernel(r/sigma)

                aux = dg_sgmdr/(sigma*r) - 3*g_sgm/r2

                crss1 = dX2*gam3 - dX3*gam2
                crss2 = dX3*gam1 - dX1*gam3
                crss3 = dX1*gam2 - dX2*gam1

                U1 += g_sgm * crss1
                U2 += g_sgm * crss2
                U3 += g_sgm * crss3

                gam1 *= g_sgm; gam2 *= g_sgm; gam3 *= g_sgm
                dX1 *= aux; dX2 *= aux; dX3 *= aux

                J1 += crss1 * dX1
                J2 += crss2 * dX1 - gam3
                J3 += crss3 * dX1 + gam2
                J4 += crss1 * dX2 + gam3
                J5 += crss2 * dX2
                J6 += crss3 * dX2 - gam1
                J7 += crss1 * dX3 - gam2
                J8 += crss2 * dX3 + gam1
                J9 += crss3 * dX3
            end
            i += Int32(1)
        end

        @inbounds out[1, j_target] = U1
        @inbounds out[2, j_target] = U2
        @inbounds out[3, j_target] = U3
        @inbounds out[4, j_target] = J1
        @inbounds out[5, j_target] = J2
        @inbounds out[6, j_target] = J3
        @inbounds out[7, j_target] = J4
        @inbounds out[8, j_target] = J5
        @inbounds out[9, j_target] = J6
        @inbounds out[10, j_target] = J7
        @inbounds out[11, j_target] = J8
        @inbounds out[12, j_target] = J9
    end
end

function ka_gpu_direct!(pfield, backend; workgroup=256)
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = KA.zeros(backend, T, 12, n)

    ka_direct_kernel!(backend, workgroup)(out, s, Int32(n), pfield.kernel.g_dgdr; ndrange=n)
    KA.synchronize(backend)

    view(P, FLOWVPM.U_INDEX, 1:n) .+= view(out, 1:3, :)
    view(P, FLOWVPM.J_INDEX, 1:n) .+= view(out, 4:12, :)
    return nothing
end

function build_pfield(n, R; seed=1)
    Random.seed!(seed)
    pfield = FLOWVPM.ParticleField(n, R)
    for i in 1:n
        X = rand(R, 3) .* 2 .- 1
        Gamma = rand(R, 3) .* 2 .- 1
        sigma = R(0.1) + R(0.05) * rand(R)
        FLOWVPM.add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

function build_metal_from(cpu_pfield)
    n = cpu_pfield.maxparticles
    R = eltype(cpu_pfield.particles)
    mfield = FLOWVPM.ParticleField(n, R; arraytype=Metal.MtlArray, np=cpu_pfield.np)
    mfield.particles .= Metal.MtlArray(cpu_pfield.particles)
    return mfield
end

function relerr(a, b)
    a, b = Array(a), Array(b)
    return maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(eltype(b)))
end

# --- correctness check: KA-Metal vs CPU reference ---
println("=== Correctness: KA-Metal vs CPU direct ===")
for n in (64, 200, 1000)
    cpu = build_pfield(n, Float32)
    mfield = build_metal_from(cpu)
    FLOWVPM.UJ_direct(cpu)
    ka_gpu_direct!(mfield, MetalBackend())
    eU = relerr(view(cpu.particles, FLOWVPM.U_INDEX, :), view(mfield.particles, FLOWVPM.U_INDEX, :))
    eJ = relerr(view(cpu.particles, FLOWVPM.J_INDEX, :), view(mfield.particles, FLOWVPM.J_INDEX, :))
    println("n=$n  max U relerr=$eU  max J relerr=$eJ")
end

# --- speed: KA-Metal vs existing @metal ext (FLOWVPM.gpu_direct!) vs CPU-FMM ---
const NS = (1_000, 5_000, 20_000, 50_000, 100_000)

function timeit_cpu_fmm(n; reps=3)
    t = Inf
    for _ in 1:reps
        pfield = build_pfield(n, Float64)
        dt = @elapsed FLOWVPM.UJ_fmm(pfield)
        t = min(t, dt)
    end
    return t
end

function timeit_metal_ext_direct(n; reps=3)
    t = Inf
    for _ in 1:reps
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        FLOWVPM.UJ_direct(mfield) # warm up
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        dt = @elapsed (FLOWVPM.UJ_direct(mfield); Metal.synchronize())
        t = min(t, dt)
    end
    return t
end

function timeit_ka_metal_direct(n; reps=3)
    backend = MetalBackend()
    t = Inf
    for _ in 1:reps
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        ka_gpu_direct!(mfield, backend) # warm up (compile)
        cpu = build_pfield(n, Float32)
        mfield = build_metal_from(cpu)
        dt = @elapsed ka_gpu_direct!(mfield, backend)
        t = min(t, dt)
    end
    return t
end

println()
println("=== Speed: CPU-FMM vs @metal-ext direct vs KA-Metal direct ===")
println("n, CPU-FMM (s), metal-ext (s), KA-metal (s), speedup FMM/metal-ext, speedup FMM/KA, KA/metal-ext")
for n in NS
    t_cpu = timeit_cpu_fmm(n)
    t_ext = timeit_metal_ext_direct(n)
    t_ka = timeit_ka_metal_direct(n)
    println("$n, $(round(t_cpu, sigdigits=4)), $(round(t_ext, sigdigits=4)), $(round(t_ka, sigdigits=4)), " *
            "$(round(t_cpu/t_ext, sigdigits=3))x, $(round(t_cpu/t_ka, sigdigits=3))x, $(round(t_ka/t_ext, sigdigits=3))x")
end
