#=##############################################################################
# DESCRIPTION
    KernelAbstractions.jl + Metal.jl package extension: GPU kernels for the
    direct (no-FMM) N-body sum, basis-function evaluation, and SFS
    vortex-stretching contribution, backend-agnostic via KernelAbstractions
    (see .claude/plans/we-need-to-make-idempotent-spark.md side-track).
    Loaded automatically whenever FLOWVPM, KernelAbstractions, and Metal are
    all loaded in the same environment.

    Replaces the hand-written ext/FLOWVPMMetalExt.jl @metal kernel: proven in
    test/metal_env/ka_direct_bench.jl to match its correctness (~1e-7 relerr
    vs CPU reference) and speed (within 0-4%) for the direct-sum kernel on
    Metal. Brute-force per-target loop (no shared-memory tiling/atomics,
    unlike FLOWVPMCUDAExt.jl's tiled kernels) -- same tradeoff the CUDA ext's
    zeta/estr kernels make, chosen for auditability over peak perf.

    NOTE: CUDA is intentionally NOT dispatched through this KA path.
    FLOWVPMCUDAExt.jl keeps its tiled, H200-validated kernels; replacing them
    with KA equivalents requires a real non-regression benchmark against
    gpu_atomic_square! on H200 hardware, unavailable in this environment (see
    project decision in the metal-testing side-track). KA is scoped as a
    Metal-only migration for this package until that gate is run.
=###############################################################################
module FLOWVPMKAMetalExt

using FLOWVPM
using Metal
using KernelAbstractions

const KA = KernelAbstractions
const fmm = FLOWVPM.fmm

#------- radix/FMM device-resident coupling -------#
#
# MOVED. `device_backend`, `source_to_buffer!`, `buffer_to_target!` and
# `sfs_to_target!` used to be defined here, typed to `MtlArray`, as verbatim
# copies of FLOWVPMCUDAExt.jl's `AnyCuArray` versions. Nothing in their bodies
# was device code -- they are broadcasts over views of `pfield.particles` -- so
# they now live once in ext/FLOWVPMGPUExt.jl, dispatched on
# `GPUArraysCore.AnyGPUMatrix`, which covers Metal and every other GPU backend.
# What remains below is the part that genuinely is Metal-and-KA: the kernels.

# Each thread handles one target and loops directly over all sources in
# global (device) memory. Math mirrors FLOWVPMCUDAExt.jl's gpu_interaction!.
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

"""
    FLOWVPM.gpu_direct!(pfield::ParticleField)

KA-Metal implementation of the direct (no-FMM) O(N²) N-body sum: overloads
the stub declared in `FLOWVPM_UJ.jl`, dispatched to from `UJ_direct`
whenever `pfield.particles isa MtlArray`. Brute-force (no tiling) -- see
module docstring above.
"""
function FLOWVPM.gpu_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                             ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:MtlArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = KA.zeros(MetalBackend(), T, 12, n)

    ka_direct_kernel!(MetalBackend(), 256)(out, s, Int32(n), pfield.kernel.g_dgdr; ndrange=n)
    KA.synchronize(MetalBackend())

    view(P, FLOWVPM.U_INDEX, 1:n) .+= view(out, 1:3, :)
    view(P, FLOWVPM.J_INDEX, 1:n) .+= view(out, 4:12, :)

    return nothing
end

# Each thread handles one target and brute-force loops over every source
# directly from global memory. Mirrors FLOWVPMCUDAExt.jl's
# gpu_zeta_direct_kernel!.
@kernel function ka_zeta_direct_kernel!(out, @Const(s), n::Int32, zeta)
    j_target = @index(Global)
    if j_target <= n
        @inbounds tx = s[1, j_target]
        @inbounds ty = s[2, j_target]
        @inbounds tz = s[3, j_target]

        T = eltype(s)
        acc1, acc2, acc3 = zero(T), zero(T), zero(T)

        i::Int32 = 1
        while i <= n
            @inbounds dX1 = tx - s[1, i]
            @inbounds dX2 = ty - s[2, i]
            @inbounds dX3 = tz - s[3, i]
            r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

            @inbounds sigma = s[7, i]
            zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)

            @inbounds acc1 += s[4, i]*zeta_sgm
            @inbounds acc2 += s[5, i]*zeta_sgm
            @inbounds acc3 += s[6, i]*zeta_sgm

            i += Int32(1)
        end

        @inbounds out[1, j_target] += acc1
        @inbounds out[2, j_target] += acc2
        @inbounds out[3, j_target] += acc3
    end
end

"""
    FLOWVPM.gpu_zeta_direct!(pfield::ParticleField)

KA-Metal implementation of `zeta_direct`'s O(N²) direct-sum basis-function
evaluation, overloading the stub declared in `FLOWVPM_viscous.jl`. Unlike
most direct-sum call sites, `zeta_direct` includes ALL particles (even
static ones) as both source and target -- matching the CPU version's
`iterator(pfield; include_static=true)` on both sides -- so no active-particle
masking is applied here.
"""
function FLOWVPM.gpu_zeta_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:MtlArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = KA.zeros(MetalBackend(), T, 3, n)

    ka_zeta_direct_kernel!(MetalBackend(), 256)(out, s, Int32(n), pfield.kernel.zeta; ndrange=n)
    KA.synchronize(MetalBackend())

    # CPU `zeta_direct` zeroes VORTICITY_INDEX before accumulating (over ALL
    # particles, per the include_static=true above), so this is an
    # assignment, not an accumulation, to match.
    view(P, FLOWVPM.VORTICITY_INDEX, 1:n) .= view(out, 1:3, :)

    return nothing
end

# Each thread handles one target and brute-force loops over every source
# directly from global memory. Mirrors FLOWVPMCUDAExt.jl's
# gpu_estr_direct_kernel!.
@kernel function ka_estr_direct_kernel!(sfs_out, @Const(P), n::Int32, zeta, transposed::Bool,
                                         static_row::Int32, j1::Int32, j2::Int32, j3::Int32,
                                         j4::Int32, j5::Int32, j6::Int32, j7::Int32, j8::Int32, j9::Int32)
    j_target = @index(Global)
    T = eltype(P)
    if j_target <= n
        @inbounds target_is_static = P[static_row, j_target]
        if target_is_static == 0
            @inbounds tx = P[1, j_target]
            @inbounds ty = P[2, j_target]
            @inbounds tz = P[3, j_target]
            @inbounds JT1 = P[j1, j_target]; @inbounds JT2 = P[j2, j_target]; @inbounds JT3 = P[j3, j_target]
            @inbounds JT4 = P[j4, j_target]; @inbounds JT5 = P[j5, j_target]; @inbounds JT6 = P[j6, j_target]
            @inbounds JT7 = P[j7, j_target]; @inbounds JT8 = P[j8, j_target]; @inbounds JT9 = P[j9, j_target]

            acc1, acc2, acc3 = zero(T), zero(T), zero(T)

            i::Int32 = 1
            while i <= n
                @inbounds source_is_static = P[static_row, i]
                if source_is_static == 0
                    @inbounds sx = P[1, i]
                    @inbounds sy = P[2, i]
                    @inbounds sz = P[3, i]
                    dX1 = tx - sx
                    dX2 = ty - sy
                    dX3 = tz - sz
                    r = sqrt(dX1*dX1 + dX2*dX2 + dX3*dX3)

                    @inbounds sigma = P[7, i]
                    zeta_sgm = zeta(r/sigma) / (sigma*sigma*sigma)

                    @inbounds GS1 = P[4, i]; @inbounds GS2 = P[5, i]; @inbounds GS3 = P[6, i]
                    @inbounds JS1 = P[j1, i]; @inbounds JS2 = P[j2, i]; @inbounds JS3 = P[j3, i]
                    @inbounds JS4 = P[j4, i]; @inbounds JS5 = P[j5, i]; @inbounds JS6 = P[j6, i]
                    @inbounds JS7 = P[j7, i]; @inbounds JS8 = P[j8, i]; @inbounds JS9 = P[j9, i]

                    if transposed
                        S1 = (JT1-JS1)*GS1 + (JT2-JS2)*GS2 + (JT3-JS3)*GS3
                        S2 = (JT4-JS4)*GS1 + (JT5-JS5)*GS2 + (JT6-JS6)*GS3
                        S3 = (JT7-JS7)*GS1 + (JT8-JS8)*GS2 + (JT9-JS9)*GS3
                    else
                        S1 = (JT1-JS1)*GS1 + (JT4-JS4)*GS2 + (JT7-JS7)*GS3
                        S2 = (JT2-JS2)*GS1 + (JT5-JS5)*GS2 + (JT8-JS8)*GS3
                        S3 = (JT3-JS3)*GS1 + (JT6-JS6)*GS2 + (JT9-JS9)*GS3
                    end

                    acc1 += zeta_sgm*S1
                    acc2 += zeta_sgm*S2
                    acc3 += zeta_sgm*S3
                end
                i += Int32(1)
            end

            @inbounds sfs_out[1, j_target] += acc1
            @inbounds sfs_out[2, j_target] += acc2
            @inbounds sfs_out[3, j_target] += acc3
        end
    end
end

"""
    FLOWVPM.gpu_estr_direct!(pfield::ParticleField)

KA-Metal implementation of `Estr_direct!`'s O(N²) direct-sum SFS
vortex-stretching contribution, overloading the stub declared in
`FLOWVPM_subfilterscale_models.jl`. Both source and target loops skip static
particles (matching `Estr_direct_singlethreaded`/`_multithreaded`'s use of
the default `iterator(pfield)`, which excludes them), and results are
accumulated (`+=`) into `SFS_INDEX`, matching the CPU version, which never
resets SFS itself (that's done separately via `_reset_particles_sfs`, gated
by the `reset_sfs` kwarg upstream in `UJ_direct`/`UJ_fmm`).
"""
function FLOWVPM.gpu_estr_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:MtlArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    out = KA.zeros(MetalBackend(), T, 3, n)

    jrows = Int32.(FLOWVPM.J_INDEX)

    ka_estr_direct_kernel!(MetalBackend(), 256)(
        out, P, Int32(n), pfield.kernel.zeta, pfield.transposed,
        Int32(FLOWVPM.STATIC_INDEX),
        jrows[1], jrows[2], jrows[3], jrows[4], jrows[5], jrows[6], jrows[7], jrows[8], jrows[9];
        ndrange=n)
    KA.synchronize(MetalBackend())

    view(P, FLOWVPM.SFS_INDEX, 1:n) .+= view(out, 1:3, :)

    return nothing
end

end # module FLOWVPMKAMetalExt
