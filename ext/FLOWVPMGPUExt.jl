#=##############################################################################
# DESCRIPTION
    Vendor-neutral device extension: the device-resident coupling between
    FLOWVPM's ParticleField and FastMultipole's radix FMM, plus the direct-sum
    kernels.

    These four hooks are what FastMultipole demands of any DeviceResident
    system: name the backend to allocate on, pack sources into the framework's
    persistent device buffer, and accumulate the U/J and SFS results back. Their
    bodies contain no device code at all -- every one is a broadcast over views
    of `pfield.particles` -- so there was never anything vendor-specific about
    them.

    `GPUArraysCore.AnyGPUMatrix` is the dispatch handle that removes the
    duplication: it is the wrapper-aware GPU-array union (so a `SubArray` of a
    device matrix still matches, which the plain `AbstractGPUArray` bound does
    not), it covers CUDA, Metal, AMDGPU and oneAPI alike, and a host `Array` is
    NOT one of them. That last point is what keeps this safe: FastMultipole's
    own host `source_to_buffer!(buffer::Matrix, system, sort_index)`
    (compatibility.jl:581) and FLOWVPM's host `sfs_to_target!` take plain
    `Matrix`, whose intersection with `AnyGPUMatrix` is empty, so no ambiguity
    is possible and the host paths are untouched.

    The direct (no-FMM) N-body sum, basis-function evaluation and SFS
    vortex-stretching kernels live here too, written with KernelAbstractions
    `@kernel` and launched on `KA.get_backend(pfield.particles)`, so one
    extension serves CUDA, Metal and any other KA backend. They are brute-force
    per-target loops (no shared-memory tiling); the former hand-written CUDA
    extension's tiled kernels were removed together with FastMultipole's native
    CUDA FMM lifecycle once the KA path reached parity (2026-09-02).
=###############################################################################
module FLOWVPMGPUExt

using FLOWVPM
using GPUArraysCore
using KernelAbstractions

const KA = KernelAbstractions
const fmm = FLOWVPM.fmm

if FLOWVPM._FMM_HAS_RADIX

# `residency(pfield)` is already backend-agnostic (`particles isa Array`,
# FLOWVPM_fmm_radix.jl:51), so any GPU-backed field takes the DeviceResident
# path and FastMultipole then requires all four of these.
const GPUField{R} = FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,
    TRelaxation,AT} where {F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,
                           AT<:AbstractGPUArray{R}}

# Which KA backend the field's storage lives on; FastMultipole's non-CUDA radix
# backend registry needs this BEFORE it may touch any device buffer.
fmm.device_backend(pfield::GPUField) = KA.get_backend(pfield.particles)

# Framework-owned persistent device source buffer, filled as a 9 x np view
# (live prefix, identity sort index). Packed layout (integration-api-spec §3):
#   rows 1:3  position            (X_INDEX)
#   row  4    MAC/error radius    rho_sigma * sigma
#   rows 5:7  vector strength     (GAMMA_INDEX)
#   row  8    raw smoothing sigma (SIGMA_INDEX; read by RegularizedVortex)
#   row  9    SFS active mask     (1 non-static, 0 static)
# Steady-state allocation-free: broadcasts into the existing buffer only.
function fmm.source_to_buffer!(buf::AnyGPUMatrix, pfield::GPUField{R}, sort_index) where R
    np = pfield.np
    (first(sort_index) == 1 && last(sort_index) == np) || error(
        "FLOWVPM device source_to_buffer! expects the identity sort index over " *
        "the live particle prefix (got $(first(sort_index)):$(last(sort_index)) for np=$np)")
    size(buf, 1) >= 9 && size(buf, 2) == np || error(
        "unexpected device source buffer shape $(size(buf)) for np=$np")
    P = pfield.particles
    rho_sigma = R(pfield.fmm.default_rho_over_sigma)
    # One flat kernel, lane per particle. The previous five broadcasts over
    # strided row views cost 0.6 ms/step on Metal for 450 kB (2026-09-01).
    kernel = _source_to_buffer_kernel!(KA.get_backend(P), 256)
    kernel(buf, P, rho_sigma, first(FLOWVPM.X_INDEX), first(FLOWVPM.GAMMA_INDEX),
           FLOWVPM.SIGMA_INDEX, FLOWVPM.STATIC_INDEX, np; ndrange=np)
    return buf
end

@kernel function _source_to_buffer_kernel!(buf, @Const(P), rho_sigma, ix, ig, isig, istat, np)
    i = @index(Global)
    @inbounds if i <= np
        buf[1, i] = P[ix, i]
        buf[2, i] = P[ix + 1, i]
        buf[3, i] = P[ix + 2, i]
        sig = P[isig, i]
        buf[4, i] = rho_sigma * sig
        buf[5, i] = P[ig, i]
        buf[6, i] = P[ig + 1, i]
        buf[7, i] = P[ig + 2, i]
        buf[8, i] = sig
        buf[9, i] = P[istat, i] == 0
    end
end

# Framework-owned per-system device output buffer, switch-relative rows, in
# global (unsorted) particle order. ACCUMULATE (.+=): FLOWVPM zeroes U/J in
# `_reset_particles` at the top of each evaluation and the framework delivers
# the evaluation's total influence. Accumulating into all live particles
# (static included) matches the legacy `buffer_to_target_system!`.
function fmm.buffer_to_target!(pfield::GPUField, buf::AnyGPUMatrix,
        derivatives_switch, sort_index)
    np = pfield.np
    size(buf, 2) == np || error(
        "unexpected device output buffer shape $(size(buf)) for np=$np")
    P = pfield.particles
    grange = fmm.gradient_range(derivatives_switch)
    isempty(grange) ||
        (view(P, FLOWVPM.U_INDEX, 1:np) .+= view(buf, grange, :))
    hrange = fmm.hessian_range(derivatives_switch)
    isempty(hrange) ||
        (view(P, FLOWVPM.J_INDEX, 1:np) .+= view(buf, hrange, :))
    return pfield
end

# Task 048: framework-owned per-system 3 x np device SFS buffer (E_str, global
# particle order). ACCUMULATE into SFS_INDEX (rows 40:42), mirroring the
# Estr_direct/Estr_fmm! += convention; the caller (UJ_fmm) owns the SFS reset.
function fmm.sfs_to_target!(pfield::GPUField, buf::AnyGPUMatrix,
        sort_index=1:pfield.np)
    np = pfield.np
    size(buf, 2) == np || error(
        "unexpected device SFS buffer shape $(size(buf)) for np=$np")
    view(pfield.particles, FLOWVPM.SFS_INDEX, 1:np) .+= buf
    return pfield
end

#------- core spreading on the device: ζ reconstruction + RBF conjugate gradient -------#

# 3 x maxparticles sorted-order accumulator and global-order delivery buffer,
# allocated once per field (radix_zeta! owns neither).
const _zeta_buffers = IdDict{Any,Any}()

function _zeta_buffers_for(pfield::GPUField{R}) where R
    bufs = get(_zeta_buffers, pfield, nothing)
    if bufs === nothing || size(bufs[1], 2) < pfield.maxparticles
        P = pfield.particles
        bufs = (similar(P, R, 3, pfield.maxparticles), similar(P, R, 3, pfield.maxparticles))
        _zeta_buffers[pfield] = bufs
    end
    return bufs
end

# ζ arrives in global particle order; the host zeta_fmm assigns (after zeroing).
function fmm.zeta_to_target!(pfield::GPUField, buf::AnyGPUMatrix, sort_index=1:pfield.np)
    np = pfield.np
    view(pfield.particles, FLOWVPM.VORTICITY_INDEX, 1:np) .= view(buf, 1:3, 1:np)
    return pfield
end

function FLOWVPM.zeta_fmm(pfield::GPUField)
    st = FLOWVPM._radix_fmm_coupling!(pfield)
    om, out = _zeta_buffers_for(pfield)
    fmm.radix_zeta!(st.cache, (pfield,), om, out)
    return nothing
end

# Same algorithm and storage as the host rbf_conjugategradient (x = M[1:3],
# r = M[4:6], b = M[7:9], A p = vorticity rows, p = Gamma rows), with the
# per-particle loops as whole-field broadcasts and the dot products as
# device reductions. Static particles are excluded from the solve exactly as
# `iterator(pfield)` excludes them on the host.
function FLOWVPM.rbf_conjugategradient(pfield::GPUField{R}, cs::FLOWVPM.CoreSpreading) where R
    P = pfield.particles; np = pfield.np
    X = view(P, FLOWVPM.M_INDEX[1:3], 1:np)      # solution
    Rr = view(P, FLOWVPM.M_INDEX[4:6], 1:np)     # residual
    B = view(P, FLOWVPM.M_INDEX[7:9], 1:np)      # target vorticity
    G = view(P, FLOWVPM.GAMMA_INDEX, 1:np)       # search direction p
    W = view(P, FLOWVPM.VORTICITY_INDEX, 1:np)   # A p
    vol = view(P, FLOWVPM.VOL_INDEX:FLOWVPM.VOL_INDEX, 1:np)
    act = view(P, FLOWVPM.STATIC_INDEX:FLOWVPM.STATIC_INDEX, 1:np) .== 0   # 1 x np mask
    dots(A, Bm) = [sum(view(A, i:i, :) .* view(Bm, i:i, :) .* act) for i in 1:3]

    cs.rr0s .= 0; cs.rrs .= 0; cs.flags .= false
    X .= ifelse.(act, B .* vol, X)
    G .= ifelse.(act, X, G)
    cs.zeta(pfield)
    Rr .= ifelse.(act, B .- W, Rr)               # r0 = b - A x0
    G .= ifelse.(act, Rr, G)                     # p0 = r0
    cs.rr0s .= dots(Rr, Rr)
    cs.rrs .= cs.rr0s
    for i in 1:3
        cs.flags[i] = sqrt(cs.rr0s[i]) > cs.tol || sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
    end
    for it in 1:cs.itmax
        true in cs.flags || break
        cs.zeta(pfield)                          # A p -> W
        cs.pAps .= dots(G, W)
        for i in 1:3
            cs.alphas[i] = cs.rrs[i] / cs.pAps[i] * cs.flags[i]
        end
        cs.prev_rrs .= cs.rrs
        al = _rowvec(P, cs.alphas)
        X .= ifelse.(act, X .+ al .* G, X)
        Rr .= ifelse.(act, Rr .- al .* W, Rr)
        cs.rrs .= dots(Rr, Rr)
        cs.betas .= cs.rrs ./ cs.prev_rrs
        for i in 1:3
            abs(cs.prev_rrs[i]) <= 2 * eps() && (cs.betas[i] = 1)
        end
        be = _rowvec(P, cs.betas)
        G .= ifelse.(act, Rr .+ be .* G, G)
        for i in 1:3
            cs.flags[i] *= abs(cs.rr0s[i]) <= 2 * eps() ? false : sqrt(cs.rrs[i] / cs.rr0s[i]) > cs.tol
        end
        if it == cs.itmax && true in cs.flags
            msg = "Maximum number of iterations $(cs.itmax) reached before convergence." *
                  " Errors: $(sqrt.(cs.rrs ./ cs.rr0s)), tolerance:$(cs.tol)"
            cs.iterror ? error(msg) : (cs.verbose && @warn(msg))
        end
    end
    G .= ifelse.(act, X, G)                      # Gamma = solution
    return nothing
end

# 3 x 1 device column of per-dimension scalars, for broadcasting against 3 x np views
_rowvec(P, v) = (c = similar(P, eltype(P), 3, 1); copyto!(c, reshape(eltype(P).(v), 3, 1)); c)

end # FLOWVPM._FMM_HAS_RADIX


#------- direct-sum kernels (KernelAbstractions, any GPU backend) -------#

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

KernelAbstractions implementation of the direct (no-FMM) O(N²) N-body sum: overloads
the stub declared in `FLOWVPM_UJ.jl`, dispatched to from `UJ_direct`
whenever `pfield.particles` is a GPU array. Brute-force (no tiling) -- see
module docstring above.
"""
function FLOWVPM.gpu_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                             ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:AbstractGPUArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = KA.zeros(KA.get_backend(pfield.particles), T, 12, n)

    ka_direct_kernel!(KA.get_backend(pfield.particles), 256)(out, s, Int32(n), pfield.kernel.g_dgdr; ndrange=n)
    KA.synchronize(KA.get_backend(pfield.particles))

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

KernelAbstractions implementation of `zeta_direct`'s O(N²) direct-sum basis-function
evaluation, overloading the stub declared in `FLOWVPM_viscous.jl`. Unlike
most direct-sum call sites, `zeta_direct` includes ALL particles (even
static ones) as both source and target -- matching the CPU version's
`iterator(pfield; include_static=true)` on both sides -- so no active-particle
masking is applied here.
"""
function FLOWVPM.gpu_zeta_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:AbstractGPUArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = KA.zeros(KA.get_backend(pfield.particles), T, 3, n)

    ka_zeta_direct_kernel!(KA.get_backend(pfield.particles), 256)(out, s, Int32(n), pfield.kernel.zeta; ndrange=n)
    KA.synchronize(KA.get_backend(pfield.particles))

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

KernelAbstractions implementation of `Estr_direct!`'s O(N²) direct-sum SFS
vortex-stretching contribution, overloading the stub declared in
`FLOWVPM_subfilterscale_models.jl`. Both source and target loops skip static
particles (matching `Estr_direct_singlethreaded`/`_multithreaded`'s use of
the default `iterator(pfield)`, which excludes them), and results are
accumulated (`+=`) into `SFS_INDEX`, matching the CPU version, which never
resets SFS itself (that's done separately via `_reset_particles_sfs`, gated
by the `reset_sfs` kwarg upstream in `UJ_direct`/`UJ_fmm`).
"""
function FLOWVPM.gpu_estr_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                                  ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:AbstractGPUArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    out = KA.zeros(KA.get_backend(pfield.particles), T, 3, n)

    jrows = Int32.(FLOWVPM.J_INDEX)

    ka_estr_direct_kernel!(KA.get_backend(pfield.particles), 256)(
        out, P, Int32(n), pfield.kernel.zeta, pfield.transposed,
        Int32(FLOWVPM.STATIC_INDEX),
        jrows[1], jrows[2], jrows[3], jrows[4], jrows[5], jrows[6], jrows[7], jrows[8], jrows[9];
        ndrange=n)
    KA.synchronize(KA.get_backend(pfield.particles))

    view(P, FLOWVPM.SFS_INDEX, 1:n) .+= view(out, 1:3, :)

    return nothing
end


end # module
