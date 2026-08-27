#=##############################################################################
# DESCRIPTION
    Metal.jl package extension: minimal GPU kernel for the direct (no-FMM)
    N-body sum, for local Apple-Silicon speed testing (see
    .claude/plans/we-need-to-make-idempotent-spark.md side-track). Brute-force
    per-target loop (no shared-memory tiling/atomics, unlike
    FLOWVPMCUDAExt.jl's gpu_atomic_square!) -- enough to get a real speed
    number before investing in an optimized tiled kernel.
=###############################################################################
module FLOWVPMMetalExt

using FLOWVPM
using Metal

# Each thread handles one target and loops directly over all sources in
# global (device) memory. Math mirrors FLOWVPMCUDAExt.jl's gpu_interaction!.
function gpu_direct_kernel!(out, s, n::Int32, kernel)
    j_target = thread_position_in_grid_1d()
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
    return
end

"""
    FLOWVPM.gpu_direct!(pfield::ParticleField)

Metal implementation of the direct (no-FMM) O(N²) N-body sum: overloads the
stub declared in `FLOWVPM_UJ.jl`, dispatched to from `UJ_direct` whenever
`pfield.particles isa MtlArray`. Brute-force (no tiling) -- see module
docstring above.
"""
function FLOWVPM.gpu_direct!(pfield::FLOWVPM.ParticleField{R,F,V,TUinf,S,Tkernel,TUJ,Tintegration,TRelaxation,AT}
                             ) where {R,F<:FLOWVPM.Formulation,V<:FLOWVPM.ViscousScheme,TUinf,S<:FLOWVPM.SubFilterScale,Tkernel,TUJ,Tintegration,TRelaxation,AT<:MtlArray{R}}
    n = pfield.np
    n == 0 && return nothing

    P = pfield.particles
    T = eltype(P)

    s = view(P, 1:7, 1:n)
    out = Metal.mtl(zeros(T, 12, n))

    nthreads = 256
    ngroups = cld(n, nthreads)

    Metal.@metal threads=nthreads groups=ngroups gpu_direct_kernel!(out, s, Int32(n), pfield.kernel.g_dgdr)

    view(P, FLOWVPM.U_INDEX, 1:n) .+= view(out, 1:3, :)
    view(P, FLOWVPM.J_INDEX, 1:n) .+= view(out, 4:12, :)

    return nothing
end

end # module FLOWVPMMetalExt
