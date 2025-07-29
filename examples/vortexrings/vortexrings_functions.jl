#=##############################################################################
# DESCRIPTION
    Automatic generation and processing of vortex rings.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################

dot(A, B) = A[1]*B[1] + A[2]*B[2] + A[3]*B[3]
norm(X) = sqrt(dot(X, X))
function cross(A::AbstractVector{TF},B::AbstractVector{TF}) where TF
    out = zeros(TF, 3)
    cross!(out, A, B)
    return out
end
function cross!(out::AbstractVector{TF}, A::AbstractVector{TF}, B::AbstractVector{TF}) where TF
    out[1] = A[2]*B[3] - A[3]*B[2]
    out[2] = A[3]*B[1] - A[1]*B[3]
    out[3] = A[1]*B[2] - A[2]*B[1]
end

function mean(X)
    val = 0
    for x in X
        val += x
    end
    return val/length(X)
end

"Number of particles used to discretized a ring"
number_particles(Nphi, nc; extra_nc=0) = Int( Nphi * ( 1 + 8*sum(1:(nc+extra_nc)) ) )

"Intervals of each ring"
function calc_ring_invervals(nrings, Nphis, ncs, extra_ncs)
    intervals = [0]
    for ri in 1:nrings
        Np = number_particles(Nphis[ri], ncs[ri]; extra_nc=extra_ncs[ri])
        push!(intervals, intervals[end] + Np)
    end
    return intervals
end

"Analytic self-induced velocity of an inviscid ring"
Uring(circulation, R, Rcross, beta) = circulation/(4*pi*R) * ( log(8*R/Rcross) - beta )

"""
  `addvortexring(pfield, circulation, R, AR, Rcross, Nphi, nc, sigma;
extra_nc=0, O=zeros(3), Oaxis=eye(3))`

Adds a vortex ring to the particle field `pfield`. The ring is discretized as
described in Winckelmans' 1989 doctoral thesis (Topics in Vortex Methods...),
where the ring is an ellipse of equivalent radius `R=sqrt(a*b)`, aspect ratio
`AR=a/b`, and cross-sectional radius `Rcross`, where `a` and `b` are the
semi-major and semi-minor axes, respectively. Hence, `AR=1` defines a circle of
radius `R`.

The ring is discretized with `Nphi` cross section evenly spaced and the
thickness of the toroid is discretized with `nc` layers, using particles with
smoothing radius `sigma`. Here, `nc=0` means that the ring is represented only
with particles centered along the centerline, and `nc>0` is the number of layers
around the centerline extending out from 0 to `Rcross`.

Additional layers of empty particles (particle with no strength) beyond `Rcross`
can be added with the optional argument `extra_nc`.

The ring is placed in space at the position `O` and orientation `Oaxis`,
where `Oaxis[:, 1]` is the major axis, `Oaxis[:, 2]` is the minor axis, and
`Oaxis[:, 3]` is the line of symmetry.
"""
function addvortexring(
        pfield::vpm.ParticleField{TF,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
        circulation::Real,
                            R::Real, AR::Real, Rcross::Real,
                            Nphi::Int, nc::Int, sigma::Real; extra_nc::Int=0,
                            O::Vector{<:Real}=zeros(3), Oaxis=I,
                            verbose=true, v_lvl=0
    ) where TF

    # ERROR CASE
    if AR < 1
        error("Invalid aspect ratio AR < 1 (AR = $(AR))")
    end

    a = R*sqrt(AR)                      # Semi-major axis
    b = R/sqrt(AR)                      # Semi-minor axis

    fun_S(phi, a, b) = a * EllipticFunctions.ellipticE(phi, 1-(b/a)^2) # Arc length from 0 to a given angle
    Stot = fun_S(2*pi, a, b)            # Total perimeter length of centerline

                                        # Non-dimensional arc length from 0 to a given value <=1
    fun_s(phi, a, b) = fun_S(phi, a, b)/fun_S(2*pi, a, b)
                                        # Angle associated to a given non-dimensional arc length
    fun_phi(s, a, b) = abs(s) <= eps() ? 0 :
                     abs(s-1) <= eps() ? 2*pi :
                     Roots.fzero( phi -> fun_s(phi, a, b) - s, (0, 2*pi-eps()), atol=1e-16, rtol=1e-16)

                                        # Length of a given filament in a
                                        # cross section cell
    function fun_length(r, tht, a, b, phi1, phi2)
        S1 = fun_S(phi1, a + r*cos(tht), b + r*cos(tht))
        S2 = fun_S(phi2, a + r*cos(tht), b + r*cos(tht))

        return S2-S1
    end
                                        # Function to be integrated to calculate
                                        # each cell's volume
    function fun_dvol(r, args...)
        return r * fun_length(r, args...)
    end
                                        # Integrate cell volume
    function fun_vol(dvol_wrap, r1, tht1, r2, tht2)
        (val, err) = HCubature.hcubature(dvol_wrap,  [r1, tht1], [r2, tht2];
                                           rtol=1e-8, atol=0, maxevals=1000)
        return val
    end

    invOaxis = inv(Oaxis)               # Add particles in the global coordinate system
    function addparticle(pfield, X, Gamma, sigma, vol, circulation)
        X_global = Oaxis*X + O
        Gamma_global = Oaxis*Gamma

        vpm.add_particle(pfield, X_global, Gamma_global, sigma;
                                             vol=vol, circulation=circulation)
    end

    rl = Rcross/(2*nc + 1)              # Radial spacing between cross-section layers
    dS = Stot/Nphi                      # Perimeter spacing between cross sections
    ds = dS/Stot                        # Non-dimensional perimeter spacing

    omega = circulation / (pi*Rcross^2) # Average vorticity

    org_np = vpm.get_np(pfield)

    # Discretization of torus into cross sections
    for N in 0:Nphi-1

        # Non-dimensional arc-length position of cross section along centerline
        sc1 = ds*N                      # Lower bound
        sc2 = ds*(N+1)                  # Upper bound
        sc = (sc1 + sc2)/2              # Center

        # Angle of cross section along centerline
        phi1 = fun_phi(sc1, a, b)       # Lower bound
        phi2 = fun_phi(sc2, a, b)       # Upper bound
        phic = fun_phi(sc, a, b)        # Center

        Xc = [a*sin(phic), b*cos(phic), 0]  # Center of the cross section
        T = [a*cos(phic), -b*sin(phic), 0]  # Unitary tangent of this cross section
        T ./= norm(T)
        T .*= -1                            # Flip to make +circulation travel +Z
                                        # Local coordinate system of section

        zvec = zeros(eltype(T), 3)
        zvec[3] = one(eltype(T))
        Naxis = hcat(T, cross(zvec, T), zvec)

                                        # Volume of each cell in the cross section
        dvol_wrap(X) = fun_dvol(X[1], X[2], a, b, phi1, phi2)


        # Discretization of cross section into layers
        for n in 0:nc+extra_nc

            if n==0           # Particle in the center

                r1, r2 = zero(TF), rl              # Lower and upper radius
                tht1, tht2 = zero(TF), 2*pi*one(TF)        # Left and right angle
                vol = fun_vol(dvol_wrap, r1, tht1, r2, tht2) # Volume
                X = Xc                      # Position
                Gamma = omega*vol*T         # Vortex strength

                # Filament length
                length = fun_length(zero(TF), zero(TF), a, b, phi1, phi2)

                # Circulation
                crcltn = norm(Gamma) / length

                addparticle(pfield, X, Gamma, sigma, vol, crcltn)

            else              # Layers

                rc = (1 + 12*n^2)/(6*n)*rl  # Center radius
                r1 = (2*n-1)*rl             # Lower radius
                r2 = (2*n+1)*rl             # Upper radius
                ncells = 8*n                # Number of cells
                deltatheta = 2*pi/ncells    # Angle of cells

                # Discretize layer into cells around the circumference
                for j in 0:(ncells-1)

                    tht1 = deltatheta*j     # Left angle
                    tht2 = deltatheta*(j+1) # Right angle
                    thtc = (tht1 + tht2)/2  # Center angle
                    vol = fun_vol(dvol_wrap, r1, tht1, r2, tht2) # Volume
                                            # Position
                    X = Xc + Naxis*[0, rc*cos(thtc), rc*sin(thtc)]

                                            # Vortex strength
                    if n<=nc    # Ring particles
                        Gamma = omega*vol*T
                    else        # Particles for viscous diffusion
                        Gamma = eps()*T
                    end
                                            # Filament length
                    length = fun_length(rc, thtc, a, b, phi1, phi2)
                                            # Circulation
                    crcltn = norm(Gamma) / length


                    addparticle(pfield, X, Gamma, sigma, vol, crcltn)
                end

            end

        end

    end

    if verbose
        println("\t"^(v_lvl)*"Number of particles: $(vpm.get_np(pfield) - org_np)")
    end

    return nothing
end



"""
Calculate centroid, radius, and cross-section radius of all rings from the
position of all (unweighted) particles.
"""
function calc_rings_unweighted!(outZ, outR, outsgm, pfield, nrings, intervals)

    # Iterate over each ring
    for ri in 1:nrings

        Np = intervals[ri+1] - intervals[ri]

        # Calculate centroid
        outZ[ri] .= zero(eltype(outZ[1]))
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)
            outZ[ri] .+= vpm.get_X(P)

        end
        outZ[ri] ./= Np

        # Calculate ring radius and cross-section radius
        outR[ri], outsgm[ri] = 0, 0
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)

            outR[ri] += sqrt((vpm.get_X(P)[1] - outZ[ri][1])^2 + (vpm.get_X(P)[2] - outZ[ri][2])^2 + (vpm.get_X(P)[3] - outZ[ri][3])^2)
            outsgm[ri] += P[7]

        end
        outR[ri] /= Np
        outsgm[ri] /= Np

    end

    return nothing
end


"""
Calculate centroid, radius, and cross-section radius of all rings from the
position of particles weighted by vortex strength.
"""
function calc_rings_weighted!(outZ, outR, outsgm,
        pfield::vpm.ParticleField{TF,<:Any, <:Any, <:Any,<:Any, <:Any, <:Any,<:Any, <:Any, <:Any},
        nrings, intervals) where TF

    # Iterate over each ring
    for ri in 1:nrings

        # Calculate centroid
        outZ[ri] .= zero(eltype(outZ[1]))
        magGammatot = zero(TF)
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)
            normGamma = norm(vpm.get_Gamma(P))
            magGammatot += normGamma

            outZ[ri][1] += vpm.get_X(P)[1] * normGamma
            outZ[ri][2] += vpm.get_X(P)[2] * normGamma
            outZ[ri][3] += vpm.get_X(P)[3] * normGamma
        end
        outZ[ri] ./= magGammatot

        # Calculate ring radius and cross-section radius
        outR[ri] = zero(eltype(outR[1]))
        outsgm[ri] = zero(eltype(outsgm[1]))
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)
            normGamma = norm(vpm.get_Gamma(P))

            outR[ri] += normGamma*sqrt((vpm.get_X(P)[1] - outZ[ri][1])^2 + (vpm.get_X(P)[2] - outZ[ri][2])^2 + (vpm.get_X(P)[3] - outZ[ri][3])^2)
            outsgm[ri] += normGamma*vpm.get_sigma(P)[]

        end
        outR[ri] /= magGammatot
        outsgm[ri] /= magGammatot

    end

    return nothing
end


"""
Calculate centroid, radius, and cross-section radius of all rings from the
position of particles weighted by local vorticity.
"""
function calc_rings_weightedW2!(outZ, outR, outsgm, pfield, nrings, intervals; zdir=3)

    # Iterate over each ring
    for ri in 1:nrings

        # Calculate centroid
        outZ[ri] .= 0
        magW2tot = 0
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)
            W2 = vpm.get_W1(P)^2 + vpm.get_W2(P)^2 + vpm.get_W3(P)^2
            magW2tot += W2

            for i in 1:3
                outZ[ri][i] += W2*vpm.get_X(P)[i]
            end

        end
        outZ[ri] ./= magW2tot

        # Calculate the ring's first and second radial moment radius
        # NOTE: This assumes that the ring travels in the z-direction
        outR[ri]= 0
        Wthttot = 0
        R22 = 0
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)

            r = 0
            for i in 1:3; r += (i!=zdir)*(vpm.get_X(P)[i] - outZ[ri][i])^2; end;
            r = sqrt(r)

            tht = zdir==1 ? atan(vpm.get_X(P)[3], vpm.get_X(P)[2]) :
                  zdir==2 ? atan(vpm.get_X(P)[1], vpm.get_X(P)[3]) :
                            atan(vpm.get_X(P)[2], vpm.get_X(P)[1])

            Wtht = zdir==1 ? vpm.get_W2(P)*cos(tht) + vpm.get_W3(P)*sin(tht) :
                   zdir==2 ? vpm.get_W3(P)*cos(tht) + vpm.get_W1(P)*sin(tht) :
                             vpm.get_W1(P)*cos(tht) + vpm.get_W2(P)*sin(tht)

            Wthttot += Wtht

            outR[ri] += r*Wtht
            R22 += r^2*Wtht
        end
        outR[ri] /= Wthttot
        R22 /= Wthttot

        # Calculate the ring's w-weighted core size
        outsgm[ri] = sqrt(2*abs(R22 - outR[ri]^2))

    end

    return nothing
end


"""
Calculate ring radius in x and y from all rings from the position of particles
weighted by vortex strength in y and x, respectively.
"""
function calc_elliptic_radius(outRm, outRp, Z, pfield, nrings, intervals;
                                                   unitx=(1,0,0), unity=(0,1,0))

    # Iterate over each ring
    for ri in 1:nrings

        outRm[ri] .= 0             # - directions
        outRp[ri] .= 0             # + directions
        weightxmtot, weightymtot = 0, 0
        weightxptot, weightyptot = 0, 0
        for pi in (intervals[ri]+1):(intervals[ri+1])

            P = vpm.get_particle(pfield, pi)
            weightx = dot(vpm.get_Gamma(P), unity)
            weighty = dot(vpm.get_Gamma(P), unitx)

            if vpm.get_X(P)[1]-Z[ri][1] < 0
                outRm[ri][1] -= abs(weightx*vpm.get_X(P)[1])
                weightxmtot += abs(weightx)
            else
                outRp[ri][1] += abs(weightx*vpm.get_X(P)[1])
                weightxptot += abs(weightx)
            end

            if vpm.get_X(P)[2]-Z[ri][2] < 0
                outRm[ri][2] -= abs(weighty*vpm.get_X(P)[2])
                weightymtot += abs(weighty)
            else
                outRp[ri][2] += abs(weighty*vpm.get_X(P)[2])
                weightyptot += abs(weighty)
            end

        end
        outRm[ri][1] /= weightxmtot
        outRm[ri][2] /= weightymtot
        outRp[ri][1] /= weightxptot
        outRp[ri][2] /= weightyptot

    end

    return nothing
end

"""
    Generate a runtime function for monitoring ring metrics. This monitor
outputs the centroid position, ring radius, and core radius calculated through
three different approaches: (1) Average of all particles, (2) weighted by the
strength of every particles, and (3) weighted by the strength in each
transversal direction.
"""
function generate_monitor_vortexring(nrings, Nphis, ncs, extra_ncs;
                                        TF=Float64,
                                        save_path=nothing,
                                        fname_pref="vortexring",
                                        unitx=(1,0,0), unity=(0,1,0),
                                        out1=nothing, out2=nothing,
                                        out3=nothing, out4=nothing,
                                        overwrite_intervals=nothing)

    # File names
    fnames = Tuple(joinpath(save_path, fname_pref*"-dynamics$(fi).csv")
                                      for fi in 1:(save_path!=nothing ? 4 : -1))

    # Pre-allocate memory
    Z1, Z2, Z4 = ([zeros(TF, 3) for ri in 1:nrings] for i in 1:3)
    R1, R2, R4 = (zeros(TF, nrings) for i in 1:3)
    sgm1, sgm2, sgm4 = (zeros(TF, nrings) for i in 1:3)
    R3p = [zeros(TF, 2) for ri in 1:nrings]
    R3m = [zeros(TF, 2) for ri in 1:nrings]
    outs = (out1, out2, out3, out4)

    intervals = calc_ring_invervals(nrings, Nphis, ncs, extra_ncs)
    intervals = overwrite_intervals != nothing ? overwrite_intervals : intervals

    function monitor_vortexring(pfield, t, args...; optargs...)

        # Open output files
        if pfield.nt==0

            fs = Tuple(open(fname, "w") for fname in fnames)

            # Initiate output files with a header
            for (fi, f) in enumerate(fs)
                print(f, "t")                   # t = time stamp
                for ri in 1:nrings              # Z = centroid
                    print(f, ",Zx$(ri),Zy$(ri),Zz$(ri)")
                    if fi==1 || fi==2 || fi==4
                        print(f, ",R$(ri)")     # R = ring radius
                    else                        # - and + radius in each dim
                        print(f, ",Rxm$(ri),Rym$(ri)")
                        print(f, ",Rxp$(ri),Ryp$(ri)")
                    end
                    print(f, ",a$(ri)")         # a = core radius
                end
                print(f, "\n")
            end

        else
            fs = Tuple(open(fname, "a") for fname in fnames)
        end

        # Calculate the centroid, radius, and thickness of the ring through
        # two approaches:
        # (1) From the position of all (unweighted) particles
        # (2) From the position of particles weighted by vortex strength
        calc_rings_unweighted!(Z1, R1, sgm1, pfield, nrings, intervals)
        calc_rings_weighted!(Z2, R2, sgm2, pfield, nrings, intervals)

        # Calculate ring radius in x and y from particles weighted by
        # vortex strength in y and x, respectively
        calc_elliptic_radius(R3m, R3p, Z2, pfield, nrings, intervals;
                                                       unitx=unitx, unity=unity)

       # Calculate the centroid, radius, and thickness of the ring from
       # the position of all particles weighted by local vorticity
       calc_rings_weightedW2!(Z4, R4, sgm4, pfield, nrings, intervals)

        # Write ring to file and/or save to output arrays
        for (fi, Z, R, a) in ((1, Z1, R1, sgm1), (2, Z2, R2, sgm2),
                                (3, Z2, (R3m, R3p), sgm2), (4, Z4, R4, sgm4))
            if save_path != nothing
                f = fs[fi]
                print(f, t)
                for ri in 1:nrings
                    for dim in 1:3; print(f, ",", Z[ri][dim]); end;
                    if fi==1 || fi==2 || fi==4
                        print(f, ",", R[ri])
                    else
                        for dim in 1:2; print(f, ",", R[1][ri][dim]); end;
                        for dim in 1:2; print(f, ",", R[2][ri][dim]); end;
                    end
                    print(f, ",", a[ri])
                end
                print(f, "\n")
            end

            if outs[fi] != nothing
                push!(outs[fi], Any[t])
                push!(outs[fi][end], deepcopy(Z), deepcopy(R), deepcopy(a))
            end
        end

        # Close output files
        for f in fs
            close(f)
        end

        return false
    end

    return monitor_vortexring
end


function calc_vorticity!(
        pfield::vpm.ParticleField{TF,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any},
        ws, Xs, xoRs, nrings, Z, R, probedir;
        Gamma=TF(1e-10), sigma=one(TF), zdir=3) where TF

    org_np = vpm.get_np(pfield)        # Original number of particles
    nprobes = length(xoRs)
    X = zeros(TF, 3)

    # Add probes
    for ri in 1:nrings
        for pi in 1:nprobes

            X .= Z[ri]
            for i in 1:3; X[i] += xoRs[pi]*R[ri]*probedir[i]; end;

            vpm.add_particle(pfield, X, Gamma, sigma)
        end
    end

    # Evaluate UJ
    pfield.UJ(pfield; reset=true)

    # Save vorticity at probes
    for ri in 1:nrings
        for pi in 1:nprobes
            P = vpm.get_particle(pfield, org_np + nprobes*(ri-1) + pi)

            ws[1, pi, ri] = vpm.get_W1(P)
            ws[2, pi, ri] = vpm.get_W2(P)
            ws[3, pi, ri] = vpm.get_W3(P)
            Xs[:, pi, ri] .= vpm.get_X(P)
        end
    end

    # Remove probes
    for pi in vpm.get_np(pfield):-1:org_np+1
        vpm.remove_particle(pfield, pi)
    end

end

"""
    Generate a runtime function for monitoring vorticity distribution along
lines of probes centered around each ring.
"""
function generate_monitor_ringvorticity(nrings, Nphis, ncs, extra_ncs, TF=Float64;
                                            nprobes=100,
                                            linefactor=1.5, probedir=[1,0,0],
                                            save_path=nothing, fname_pref="vortexring",
                                            outs=nothing,
                                            overwrite_intervals=nothing)

    # File names
    fnames = [joinpath(save_path, fname_pref*"-vorticity-ring$(ri)")
                                for ri in 1:(save_path!=nothing ? nrings : -1)]

    # Position of probe line
    xoRs = linefactor*range(-1, 1, length=nprobes)

    # Pre-allocate memory
    Z = [zeros(TF,3) for ri in 1:nrings]            # Centroid of each ring
    R = zeros(TF,nrings)                            # Radius of each ring
    sgm = zeros(TF,nrings)                          # Average smoothing of each ring
    ws = zeros(TF, 3, nprobes, nrings)               # Probed vorticity on each ring
    Xs = zeros(TF, 3, nprobes, nrings)               # Probe position on each ring

    # VTK-related memory
    points = [[view(Xs, 1:3, pi, ri) for pi in 1:nprobes] for ri in 1:nrings]
    lines = [[collect(0:nprobes-1)] for ri in 1:nrings]
    data = [[
            Dict(
                  "field_name" => "W",
                  "field_type" => "vector",
                  "field_data" => [view(ws, 1:3, pi, ri) for pi in 1:nprobes]
                )
            ] for ri in 1:nrings]

    # Particle-count intervals of each ring
    intervals = calc_ring_invervals(nrings, Nphis, ncs, extra_ncs)
    intervals = overwrite_intervals != nothing ? overwrite_intervals : intervals

    function monitor_ringvorticity(pfield, t, args...; optargs...)

        # Calculate the centroid, radius, and thickness of the rings
        calc_rings_weighted!(Z, R, sgm, pfield, nrings, intervals)

        # Calculate vorticity vectors
        calc_vorticity!(pfield, ws, Xs, xoRs, nrings, Z, R, probedir; sigma=mean(sgm))

        # Write output vtk file and/or save to output arrays
        for ri in 1:nrings

            # Save to VTK file
            if save_path != nothing

                gt.generateVTK(fnames[ri], points[ri];
                                    lines=lines[ri], point_data=data[ri],
                                    path=save_path, num=pfield.nt)
            end

            # Save to array
            if outs != nothing
                push!(outs[1], deepcopy(Xs))
                push!(outs[2], deepcopy(ws))
            end

        end

        return false

    end

    return monitor_ringvorticity
end
