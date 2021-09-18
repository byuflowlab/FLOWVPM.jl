#=##############################################################################
# DESCRIPTION
    Automatic generation and processing of round jets.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################


dot(A, B) = A[1]*B[1] + A[2]*B[2] + A[3]*B[3]
norm(X) = sqrt(dot(X, X))
function cross(A,B)
    out = zeros(3)
    cross!(out, A, B)
    return out
end
function cross!(out, A, B)
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


"""
  `addannulus(pfield, circulation, R, AR, Nphi, sigma, area;
O=zeros(3), Oaxis=eye(3))`

Adds a vortex ring to the particle field `pfield` representing an annulus. The
ring is discretized as with a single layer of particles, where the ring is an
ellipse of equivalent radius `R=sqrt(a*b)`, and aspect ratio `AR=a/b`where `a`
and `b` are the semi-major and semi-minor axes, respectively. Hence, `AR=1`
defines a circle of radius `R`.

The ring is discretized with `Nphi` cross section evenly spaced in the perimeter
of the ellipse, using particles with smoothing radius `sigma`.

The ring is placed in space at the position `O` and orientation `Oaxis`,
where `Oaxis[:, 1]` is the major axis, `Oaxis[:, 2]` is the minor axis, and
`Oaxis[:, 3]` is the line of symmetry.
"""
function addannulus(pfield::vpm.ParticleField, circulation::Real,
                            R::Real, AR::Real,
                            Nphi::Int, sigma::Real, area::Real;
                            O::Vector{<:Real}=zeros(3), Oaxis=I,
                            verbose=true, v_lvl=0
                            )

    # ERROR CASE
    if AR < 1
        error("Invalid aspect ratio AR < 1 (AR = $(AR))")
    end

    a = R*sqrt(AR)                      # Semi-major axis
    b = R/sqrt(AR)                      # Semi-minor axis

    fun_S(phi, a, b) = a * Elliptic.E(phi, 1-(b/a)^2) # Arc length from 0 to a given angle
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

    invOaxis = inv(Oaxis)               # Add particles in the global coordinate system
    function addparticle(pfield, X, Gamma, sigma, vol, circulation)
        X_global = Oaxis*X + O
        Gamma_global = Oaxis*Gamma

        vpm.add_particle(pfield, X_global, Gamma_global, sigma;
                                             vol=vol, circulation=circulation)
    end

    dS = Stot/Nphi                      # Perimeter spacing between cross sections
    ds = dS/Stot                        # Non-dimensional perimeter spacing

    # Discretization of annulus into cross sections
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
        T .*= -1                        # Flip to make +circulation travel +Z
                                        # Local coordinate system of section
        Naxis = hcat(T, cross([0,0,1], T), [0,0,1])

        X = Xc                          # Position
                                        # Filament length
        length = fun_length(0, 0, a, b, phi1, phi2)
        Gamma = circulation*length*T    # Vortex strength

        addparticle(pfield, X, Gamma, sigma, area*length, circulation)

    end

    return nothing
end



"""
    `probeline_vorticity!(pfield, ws::Array{Real, 3},
lines::Array{Array{Array{Real, 1}, 1}, 1})`

Probes both the velocity and vorticity field at multiple lines of probes where
`lines[i][j]` is the vector position of the `j`-th probe in the `i`-th line. The
velocity and vorticity of such probe get stored under `U[:, j, i]` and
`W[:, j, i]`, respectively.
"""
function probeline_UW!(pfield, U, W, lines; Gamma=1e-10, sigma=1)

    org_np = vpm.get_np(pfield)        # Original number of particles

    # Add probes
    for line in lines
        for X in line
            vpm.add_particle(pfield, X, Gamma, sigma)
        end
    end

    # Evaluate UJ
    vpm._reset_particles(pfield)
    pfield.UJ(pfield)

    # Save vorticity at probes
    pi = org_np + 1
    for (li, line) in enumerate(lines)
        for (xi, X) in enumerate(line)
            P = vpm.get_particle(pfield, pi)

            U[:, xi, li] .= P.U
            W[1, xi, li] = vpm.get_W1(P)
            W[2, xi, li] = vpm.get_W2(P)
            W[3, xi, li] = vpm.get_W3(P)

            pi += 1
        end
    end

    # Remove probes
    for pi in vpm.get_np(pfield):-1:org_np+1
        vpm.remove_particle(pfield, pi)
    end

end



"""
    Generate a runtime function for monitoring velocity and vorticity along
lines of probes.
"""
function generate_probelines(lines; nprobes=100,
                                    save_path=nothing, fname_pref="roundjet",
                                    outs=nothing)


    # Generate lines of probes from the initial and end points given by the user
    fs = range(0, 1, length=nprobes)
    lines = [ [Xlo + f*(Xup-Xlo) for f in fs] for (Xlo, Xup) in lines ]

    nlines = length(lines)

    # Pre-allocate memory
    U = zeros(3, nprobes, nlines)               # Velocity at each probe
    W = zeros(3, nprobes, nlines)               # Vorticity at each probe

    # File names
    fnames = [joinpath(save_path, fname_pref*"-probeline$(li)")
                                for li in 1:(save_path!=nothing ? nlines : -1)]

    # VTK-related memory
    points_vtk = lines
    lines_vtk = [[collect(0:nprobes-1)] for li in 1:nlines]
    data = [[
            Dict(
                  "field_name" => "U",
                  "field_type" => "vector",
                  "field_data" => [view(U, 1:3, pi, li) for pi in 1:nprobes]
                ),
            Dict(
                  "field_name" => "W",
                  "field_type" => "vector",
                  "field_data" => [view(W, 1:3, pi, li) for pi in 1:nprobes]
                )
            ] for li in 1:nlines]

    function probelines(pfield, t, args...; optargs...)

        meansigma = 0
        Gammatot = 0
        for P in vpm.iterate(pfield)
            meansigma += P.sigma[1]*norm(P.Gamma)
            Gammatot += norm(P.Gamma)
        end
        meansigma /= Gammatot


        # Calculate vorticity vectors
        probeline_UW!(pfield, U, W, lines; Gamma=1e-10, sigma=meansigma)

        # Write output vtk file and/or save to output arrays
        for li in 1:nlines

            # Save to VTK file
            if save_path != nothing

                gt.generateVTK(fnames[li], points_vtk[li];
                                    lines=lines_vtk[li], point_data=data[li],
                                    path=save_path, num=pfield.nt)
            end

            # Save to array
            if outs != nothing
                push!(outs[1], deepcopy(U))
                push!(outs[2], deepcopy(W))
            end

        end

        return false

    end

    return probelines
end
