#=##############################################################################
# DESCRIPTION
    BRAINSTORM 026 Phase 2 Stage 8.3: ring collective test for the resolution
    splitting kernels.

    A thin-cored circular vortex ring is marched ~1 convective time (the time
    for the ring to self-advect one radius) with direct N-body evaluation and
    forward Euler. Arms:

      control      — no splitting
      tetra4       — force-split ALL particles with the viscous 4-child kernel
      tri3_gamma   — compression 3-child kernel, split plane ⟂ Γ̂
      tri3_axis    — compression 3-child kernel, plane ⟂ accumulated stretch axis
      pair2_gamma  — elongation 2-child kernel, offsets along Γ̂
      pair2_axis   — elongation 2-child kernel, offsets along the stretch axis

    The forced split happens after a short warm-up (so the axis variants have
    an accumulated `ResolutionSplitState` axis to use), then every arm marches
    the same schedule. Reported per sample: axial ring position and speed,
    total circulation Σ|Γ|, linear impulse ½Σx×Γ, and an enstrophy proxy
    Σ|Γ|²/σ³. One CSV per arm + a drift summary table on stdout.

    Self-contained: no deps beyond FLOWVPM (ring seeded analytically; CSVs
    written with plain IO). Run from the FLOWVPM root:

        julia --project=. examples/p026_ring_split_test.jl

    Output: examples/p026_ring_split_test_out/<arm>.csv

# AUTHORSHIP
  * Created by  : BRAINSTORM 026 Phase 2 Session 2 (agent), Sep 2026
=###############################################################################

import FLOWVPM
vpm = FLOWVPM

import LinearAlgebra: norm, cross

# ------------------------------------------------------------------ parameters
const RING_R       = 1.0     # ring radius
const RING_GAMMA   = 1.0     # ring circulation
const NPHI         = 100     # particles along the centerline
const OVERLAP      = 2.4     # sigma / inter-particle spacing
const NWARM        = 20      # warm-up steps before the forced split
const NSTEPS       = 200     # steps after the split (~1 convective time)
const NSAMPLE      = 10      # sample cadence for the CSV

const DS      = 2pi * RING_R / NPHI          # centerline spacing
const SIGMA   = OVERLAP * DS                 # particle core size
# thin-ring self-induced speed (Kelvin): U = Γ/(4πR) (ln(8R/a) − 1/4)
const U_RING  = RING_GAMMA / (4pi * RING_R) * (log(8RING_R / SIGMA) - 0.25)
const T_CONV  = RING_R / U_RING              # ~1 convective time
const DT      = T_CONV / NSTEPS

# ---------------------------------------------------------------------- set up
function make_ring_pfield(; maxp=8 * NPHI)
    pfield = vpm.ParticleField(maxp;
        UJ=vpm.UJ_direct,
        integration=vpm.euler,
        relaxation=vpm.relaxation_none,
        SFS=vpm.noSFS,
        viscous=vpm.Inviscid())
    for j in 1:NPHI
        theta = 2pi * (j - 1) / NPHI
        X = (RING_R * cos(theta), RING_R * sin(theta), 0.0)
        # filament strength Γ ds t̂
        G = RING_GAMMA * DS .* (-sin(theta), cos(theta), 0.0)
        vpm.add_particle(pfield, X, G, SIGMA;
            vol=4 / 3 * pi * SIGMA^3, circulation=RING_GAMMA)
    end
    return pfield
end

function metrics(pfield)
    np = vpm.get_np(pfield)
    circ = 0.0; z = 0.0; ens = 0.0
    I = zeros(3)
    for i in 1:np
        X = vpm.get_X(pfield, i)
        G = vpm.get_Gamma(pfield, i)
        s = vpm.get_sigma(pfield, i)[]
        g = norm(G)
        circ += g
        z += g * X[3]
        I .+= 0.5 .* cross(X, G)
        ens += g^2 / s^3
    end
    return (; np, circ, z=z / circ, I, ens)
end

march!(pfield, nsteps) = vpm.run_vpm!(pfield, DT, nsteps;
    verbose=false, prompt=false, save_path=nothing)

# force-split every current particle with one kernel; direction resolved the
# same way split_particles! would (accumulated axis or Γ̂ per opts)
function force_split_all!(pfield, kernel::Symbol, use_stretch_axis::Bool)
    rs = pfield.resolution_split
    opts = vpm.ResolutionSplitOpts(; use_stretch_axis,
        enable_viscous_split=true, enable_stretch_split=true)
    np0 = vpm.get_np(pfield)
    for i in 1:np0
        if kernel === :tetra4
            vpm._split_viscous_tetra4!(pfield, rs, i, opts.viscous_offset_ratio)
        else
            ex, ey, ez = vpm._rsplit_direction(rs, i, opts, pfield)
            if kernel === :tri3
                vpm._split_compress_tri3!(pfield, rs, i, ex, ey, ez,
                    opts.compress_offset_ratio)
            elseif kernel === :pair2
                vpm._split_elongate_pair2!(pfield, rs, i, ex, ey, ez,
                    opts.elongate_offset_ratio)
            else
                error("unknown kernel $(kernel)")
            end
        end
    end
    return pfield
end

# ------------------------------------------------------------------------- run
const OUTDIR = joinpath(@__DIR__, "p026_ring_split_test_out")
mkpath(OUTDIR)

const ARMS = (
    (name="control",     kernel=:none,   axis=false),
    (name="tetra4",      kernel=:tetra4, axis=false),
    (name="tri3_gamma",  kernel=:tri3,   axis=false),
    (name="tri3_axis",   kernel=:tri3,   axis=true),
    (name="pair2_gamma", kernel=:pair2,  axis=false),
    (name="pair2_axis",  kernel=:pair2,  axis=true),
)

results = NamedTuple[]
for arm in ARMS
    pfield = make_ring_pfield()
    vpm.enable_resolution_split!(pfield)   # warm-up accumulates the axis
    march!(pfield, NWARM)
    arm.kernel === :none || force_split_all!(pfield, arm.kernel, arm.axis)

    m0 = metrics(pfield)
    rows = [(step=0, t=0.0, m0...)]
    for block in 1:(NSTEPS ÷ NSAMPLE)
        march!(pfield, NSAMPLE)
        push!(rows, (step=block * NSAMPLE, t=block * NSAMPLE * DT,
                     metrics(pfield)...))
    end

    open(joinpath(OUTDIR, arm.name * ".csv"), "w") do io
        println(io, "step,t,np,ring_z,ring_speed,circulation,Ix,Iy,Iz,enstrophy_proxy")
        for (k, r) in enumerate(rows)
            speed = k == 1 ? U_RING :
                (r.z - rows[k-1].z) / (r.t - rows[k-1].t)
            println(io, join((r.step, r.t, r.np, r.z, speed, r.circ,
                              r.I[1], r.I[2], r.I[3], r.ens), ","))
        end
    end

    mN = last(rows)
    speed_avg = (mN.z - m0.z) / (mN.t - 0.0)
    push!(results, (; name=arm.name, np=mN.np, speed=speed_avg,
        dcirc=mN.circ / m0.circ - 1, dI=norm(mN.I - m0.I) / norm(m0.I),
        dens=mN.ens / m0.ens - 1))
    println("arm $(rpad(arm.name, 12)) np=$(lpad(mN.np, 4)) " *
        "speed=$(round(speed_avg, sigdigits=5)) (Kelvin $(round(U_RING, sigdigits=5))) " *
        "d_circ=$(round(results[end].dcirc, sigdigits=3)) " *
        "d_impulse=$(round(results[end].dI, sigdigits=3)) " *
        "d_enstrophy=$(round(results[end].dens, sigdigits=3))")
end

println("\nCSV output in $(OUTDIR)")
