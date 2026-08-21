#=##############################################################################
# DESCRIPTION
    Task 052 stage A: GPU/broadcast implementations of the pseudo3level
    dynamic-SFS procedures for CuArray-backed particle fields.

    `dynamicprocedure_pseudo3level_beforeUJ` / `_afterUJ` are per-particle
    host loops (scalar indexing) interleaved with a `pfield.UJ(...)` N-body
    evaluation at the test filter width. On a device-resident field the UJ
    call already dispatches to the radix GPU lifecycle (`UJ_fmm_gpu!`), so the
    only host-only pieces are the elementwise loops. These are ported here as
    whole-field masked broadcasts, replicating the host arithmetic exactly
    (same operation order, same clamping branches via `ifelse`), so a CPU and
    a GPU run of the same configuration follow the same SFS model math.

    Deviations from the host loops (all diagnostic-only):
      - the SFS_WATCH_* CSV diagnostics are not emitted on the device path;
      - the NaN error does not print the offending particle's state (the
        check itself is preserved as a device reduction over C1).

    Scratch usage: rows 1..7 of `pfield.scratch` (the same scratch matrix the
    relaxation broadcasts use; the procedures never run concurrently with
    relaxation, which happens inside the integrator).

# AUTHORSHIP
  * Created by  : task 052 stage A (agent), Aug 2026
  * License     : MIT License
=###############################################################################

"""
    _pseudo3level_beforeUJ_broadcast!(pfield, SFS, alpha)

GPU-safe equivalent of the per-particle loops in
`dynamicprocedure_pseudo3level_beforeUJ`. Called from that function when
`pfield.particles` is not a plain `Array`.
"""
function _pseudo3level_beforeUJ_broadcast!(pfield, SFS, alpha::Real)
    R = eltype(pfield.particles)
    np = pfield.np
    np == 0 && return nothing

    P = view(pfield.particles, :, 1:np)
    Sc = view(pfield.scratch, :, 1:np)
    row(r) = view(P, r:r, :)

    # active (non-static) mask as 1.0/0.0
    act = view(Sc, 1:1, :)
    act .= row(STATIC_INDEX) .== 0

    αR = R(alpha)
    one_R = one(R)

    # ---- test filter width ----
    σ = row(SIGMA_INDEX)
    σ .*= ifelse.(act .> 0, αR, one_R)

    # UJ with test filter (device radix path; resets U/J and SFS rows)
    pfield.UJ(pfield; sfs=true, reset=true, reset_sfs=true)

    # M := 0 for active particles (all M rows, matching the host loop's
    # `particles[M_INDEX, i] .= 0`), then store test-filter stretching in
    # M[1:3] and test-filter SFS (E_str) in M[4:6].
    Mall = view(P, M_INDEX, :)
    Mall .*= (act .== 0)

    G1, G2, G3 = row(GAMMA_INDEX[1]), row(GAMMA_INDEX[2]), row(GAMMA_INDEX[3])
    J1, J2, J3 = row(J_INDEX[1]), row(J_INDEX[2]), row(J_INDEX[3])
    J4, J5, J6 = row(J_INDEX[4]), row(J_INDEX[5]), row(J_INDEX[6])
    J7, J8, J9 = row(J_INDEX[7]), row(J_INDEX[8]), row(J_INDEX[9])
    M1, M2, M3 = row(M_INDEX[1]), row(M_INDEX[2]), row(M_INDEX[3])
    M4, M5, M6 = row(M_INDEX[4]), row(M_INDEX[5]), row(M_INDEX[6])
    S1, S2, S3 = row(SFS_INDEX[1]), row(SFS_INDEX[2]), row(SFS_INDEX[3])

    if pfield.transposed
        # Transposed scheme (Γ⋅∇')U
        M1 .= ifelse.(act .> 0, J1 .* G1 .+ J2 .* G2 .+ J3 .* G3, M1)
        M2 .= ifelse.(act .> 0, J4 .* G1 .+ J5 .* G2 .+ J6 .* G3, M2)
        M3 .= ifelse.(act .> 0, J7 .* G1 .+ J8 .* G2 .+ J9 .* G3, M3)
    else
        # Classic scheme (Γ⋅∇)U
        M1 .= ifelse.(act .> 0, J1 .* G1 .+ J4 .* G2 .+ J7 .* G3, M1)
        M2 .= ifelse.(act .> 0, J2 .* G1 .+ J5 .* G2 .+ J8 .* G3, M2)
        M3 .= ifelse.(act .> 0, J3 .* G1 .+ J6 .* G2 .+ J9 .* G3, M3)
    end
    M4 .= ifelse.(act .> 0, S1, M4)
    M5 .= ifelse.(act .> 0, S2, M5)
    M6 .= ifelse.(act .> 0, S3, M6)

    # ---- restore domain filter width ----
    σ ./= ifelse.(act .> 0, αR, one_R)

    return nothing
end

"""
    _pseudo3level_afterUJ_broadcast!(pfield, SFS, alpha, rlxf, minC, maxC;
                                     force_positive=false)

GPU-safe equivalent of the per-particle loops in
`dynamicprocedure_pseudo3level_afterUJ` (branching replicated with `ifelse`
in the host loop's operation order). Called from that function when
`pfield.particles` is not a plain `Array`.
"""
function _pseudo3level_afterUJ_broadcast!(pfield, SFS, alpha::Real, rlxf::Real,
        minC::Real, maxC::Real; force_positive::Bool=false)
    R = eltype(pfield.particles)
    np = pfield.np
    np == 0 && return nothing

    P = view(pfield.particles, :, 1:np)
    Sc = view(pfield.scratch, :, 1:np)
    row(r) = view(P, r:r, :)

    act = view(Sc, 1:1, :)
    act .= row(STATIC_INDEX) .== 0

    G1, G2, G3 = row(GAMMA_INDEX[1]), row(GAMMA_INDEX[2]), row(GAMMA_INDEX[3])
    J1, J2, J3 = row(J_INDEX[1]), row(J_INDEX[2]), row(J_INDEX[3])
    J4, J5, J6 = row(J_INDEX[4]), row(J_INDEX[5]), row(J_INDEX[6])
    J7, J8, J9 = row(J_INDEX[7]), row(J_INDEX[8]), row(J_INDEX[9])
    M1, M2, M3 = row(M_INDEX[1]), row(M_INDEX[2]), row(M_INDEX[3])
    M4, M5, M6 = row(M_INDEX[4]), row(M_INDEX[5]), row(M_INDEX[6])
    S1, S2, S3 = row(SFS_INDEX[1]), row(SFS_INDEX[2]), row(SFS_INDEX[3])

    # subtract domain-filter stretching / SFS from the test-filter values
    # stored under M[1:3] / M[4:6]
    if pfield.transposed
        M1 .-= act .* (J1 .* G1 .+ J2 .* G2 .+ J3 .* G3)
        M2 .-= act .* (J4 .* G1 .+ J5 .* G2 .+ J6 .* G3)
        M3 .-= act .* (J7 .* G1 .+ J8 .* G2 .+ J9 .* G3)
    else
        M1 .-= act .* (J1 .* G1 .+ J4 .* G2 .+ J7 .* G3)
        M2 .-= act .* (J2 .* G1 .+ J5 .* G2 .+ J8 .* G3)
        M3 .-= act .* (J3 .* G1 .+ J6 .* G2 .+ J9 .* G3)
    end
    M4 .-= act .* S1
    M5 .-= act .* S2
    M6 .-= act .* S3

    # ---- model coefficient ----
    zeta0 = R(pfield.kernel.zeta(zero(R)))
    σ = row(SIGMA_INDEX)
    C1r, C2r, C3r = row(C_INDEX[1]), row(C_INDEX[2]), row(C_INDEX[3])

    nume_raw = view(Sc, 2:2, :)
    deno_raw = view(Sc, 3:3, :)
    C3i  = view(Sc, 4:4, :)
    nume = view(Sc, 5:5, :)
    deno = view(Sc, 6:6, :)
    big  = view(Sc, 7:7, :)

    fac = R(3 * alpha - 2)
    rlxfR = R(rlxf)
    minCR, maxCR = R(minC), R(maxC)

    nume_raw .= (M1 .* G1 .+ M2 .* G2 .+ M3 .* G3) .* fac
    # host: deno /= zeta0 / sigma^3  ==  deno *= sigma^3 / zeta0
    deno_raw .= (M4 .* G1 .+ M5 .* G2 .+ M6 .* G3) .* (σ .^ 3 ./ zeta0)

    # initialize denominator memory when zero (host: C_p[3] = deno, then eps())
    C3i .= ifelse.(C3r .== 0,
                   ifelse.(deno_raw .== 0, R(eps()), deno_raw),
                   C3r)

    # Lagrangian average
    nume .= rlxfR .* nume_raw .+ (1 - rlxfR) .* C2r
    deno .= rlxfR .* deno_raw .+ (1 - rlxfR) .* C3i

    # clamping, replicating the host branch order:
    #   if |nume/deno| > maxC:
    #       if |deno| < |C3i|: deno = sign(deno)*|C3i|
    #       if |nume/deno| >= maxC (with updated deno): nume = sign(nume)*|deno|*maxC
    #   elseif |nume/deno| < minC: nume = sign(nume)*|deno|*minC
    big .= abs.(nume ./ deno) .> maxCR
    deno .= ifelse.((big .> 0) .& (abs.(deno) .< abs.(C3i)),
                    sign.(deno) .* abs.(C3i), deno)
    nume .= ifelse.((big .> 0) .& (abs.(nume ./ deno) .>= maxCR),
                    sign.(nume) .* abs.(deno) .* maxCR,
                    ifelse.((big .== 0) .& (abs.(nume ./ deno) .< minCR),
                            sign.(nume) .* abs.(deno) .* minCR,
                            nume))

    # commit (active particles only)
    C2r .= ifelse.(act .> 0, nume, C2r)
    C3r .= ifelse.(act .> 0, deno, C3r)
    C1r .= ifelse.(act .> 0, C2r ./ C3r, C1r)
    if force_positive
        # host: C1 *= sign(C1)^force_positive == |C1|
        C1r .= ifelse.(act .> 0, abs.(C1r), C1r)
    end

    any(isnan, C1r) && error("NaN in dynamicprocedure_pseudo3level_afterUJ " *
        "(GPU broadcast path, np=$(np))")

    # flush temporal memory (all M rows, active particles)
    Mall = view(P, M_INDEX, :)
    Mall .*= (act .== 0)

    return nothing
end
