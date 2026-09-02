#=##############################################################################
# DESCRIPTION
    Vendor-neutral device-resident coupling between FLOWVPM's ParticleField and
    FastMultipole's radix FMM.

    These four hooks are what FastMultipole demands of any DeviceResident
    system: name the backend to allocate on, pack sources into the framework's
    persistent device buffer, and accumulate the U/J and SFS results back. Their
    bodies contain no device code at all -- every one is a broadcast over views
    of `pfield.particles` -- so there was never anything vendor-specific about
    them, yet they existed twice, once typed to `CUDA.AnyCuArray` in
    FLOWVPMCUDAExt.jl and once to `MtlArray` in FLOWVPMKAMetalExt.jl, byte for
    byte identical apart from the array type. A third backend would have made it
    three copies.

    `GPUArraysCore.AnyGPUMatrix` is the dispatch handle that removes the
    duplication: it is the wrapper-aware GPU-array union (so a `SubArray` of a
    device matrix still matches, which the plain `AbstractGPUArray` bound does
    not), it covers CUDA, Metal, AMDGPU and oneAPI alike, and a host `Array` is
    NOT one of them. That last point is what keeps this safe: FastMultipole's
    own host `source_to_buffer!(buffer::Matrix, system, sort_index)`
    (compatibility.jl:581) and FLOWVPM's host `sfs_to_target!` take plain
    `Matrix`, whose intersection with `AnyGPUMatrix` is empty, so no ambiguity
    is possible and the host paths are untouched.

    CUDA keeps its own copies in FLOWVPMCUDAExt.jl. `AnyCuArray` is strictly
    more specific than `AnyGPUMatrix`, so a CuArray-backed field still resolves
    to exactly the methods H200 was validated against and this extension cannot
    change its behaviour. Those copies are now redundant and should be deleted
    once someone can re-run the H200 gate; doing it here would be an untested
    change to the only path that has production hardware behind it.
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

end # module
