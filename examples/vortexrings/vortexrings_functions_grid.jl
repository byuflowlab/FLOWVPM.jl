#=##############################################################################
# DESCRIPTION
    Automatic generation and processing of vortex rings initiated from a
    Cartesian grid.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Jul 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################


"""
  `addvortexring(pfield, circulation, R, AR, Rcross, dxoRcross, sigma, minmagGamma;
O=zeros(3), Oaxis=eye(3), zeta)`

Adds a vortex ring to the particle field `pfield`. The ring is discretized
by probing an analytic vorticity field on a uniform Cartesian grid, where the
ring is an ellipse of equivalent radius `R=sqrt(a*b)`, aspect ratio `AR=a/b`,
and cross-sectional radius `Rcross`, where `a` and `b` are the semi-major and
semi-minor axes, respectively. Hence, `AR=1` defines a circle of radius `R`. The
vorticity distribution inside the ring core is given by `zeta` (which defaults
to a Gaussian distribution).

The ring is discretized with cells of length `dx = dxoRcross*Rcross`, where each
cell is converted into a particle of smoothing radius `sigma`. To reduce
computation, only particles with a vortex strength larger than `minmagGamma`
get added to the particle field. This means that in a Gaussian distribution,
95% of the vorticity distribution is captured by setting
`minmagGamma = 5e-2 * circulation/(pi*Rcross^2)*dx^3`.

The ring is placed in space at the position `O` and orientation `Oaxis`,
where `Oaxis[:, 1]` is the major axis, `Oaxis[:, 2]` is the minor axis, and
`Oaxis[:, 3]` is the line of symmetry.
"""
function addvortexring(pfield::vpm.ParticleField, circulation::Real,
                            R::Real, AR::Real, Rcross::Real,
                            dxoRcross::Real, sigma::Real, minmagGamma::Real;
                            O::Vector{<:Real}=zeros(3), Oaxis=I,
                            fx=1.25, fy=1.25, fz=1.75,
                            zeta=(r,Rcross) -> 1/(pi*Rcross^2) * exp(-r^2/Rcross^2),
                            verbose=true, v_lvl=0
                            )

    # ERROR CASE
    if AR < 1
        error("Invalid aspect ratio AR < 1 (AR = $(AR))")
    end

    a = R*sqrt(AR)                      # Semi-major axis
    b = R/sqrt(AR)                      # Semi-minor axis

    # Generate analytic vorticity field
    Wfun! = generate_Wfun(circulation, R, AR, Rcross; O=O, Oaxis=Oaxis, zeta=zeta)

    # Pre-allocate memory
    W, X, Gamma = zeros(3), zeros(3), zeros(3)
    Q, N = zeros(3), zeros(3)
    ni, nj = zeros(Int, 3), zeros(Int, 3)
    Xlo, Xup, Xc, X = zeros(3), zeros(3), zeros(3), zeros(3)

    # Define grid
    Lx = fx*2*(a+Rcross)
    Ly = fy*2*(b+Rcross)
    Lz = fz*2*Rcross

    dx = dxoRcross*Rcross
    dx, dy, dz = dx, dx, dx
    nx, ny, nz = ceil(Int, Lx/dx), ceil(Int, Ly/dy), ceil(Int, Lz/dz)
    ds = [dx, dy, dz]
    ns = [nx, ny, nz]

    P_min         = [-Lx/2, -Ly/2, -Lz/2]
    P_max         = [Lx/2, Ly/2, Lz/2]

    if verbose; println("\t"^(v_lvl)*"Number of nodes: $(prod(ns))"); end
    Nphi = 0

    # Add particles at center of each cell
    for xi in 1:nx
        for yi in 1:ny
            for zi in 1:nz

                ni[1] = xi
                ni[2] = yi
                ni[3] = zi

                # Lower and upper bounds and center in ring's Cartesian coordinates
                for i in 1:3
                    Xlo[i] = P_min[i] + (ni[i]-1)*ds[i]
                    Xup[i] = P_min[i] + ni[i]*ds[i]
                    Xc[i]  = P_min[i] + (ni[i]-0.5)*ds[i]
                end

                # Volume of this cell
                vol = (Xup[1]-Xlo[1])*(Xup[2]-Xlo[2])*(Xup[3]-Xlo[3])

                # Convert to global Cartesian coordinates
                for i in 1:3
                    Xlo[i] = Xlo[1]*Oaxis[i, 1] + Xlo[2]*Oaxis[i, 2] + Xlo[3]*Oaxis[i, 3] + O[i]
                    Xup[i] = Xup[1]*Oaxis[i, 1] + Xup[2]*Oaxis[i, 2] + Xup[3]*Oaxis[i, 3] + O[i]
                    Xc[i]  = Xc[1]*Oaxis[i, 1]  + Xc[2]*Oaxis[i, 2]  + Xc[3]*Oaxis[i, 3]  + O[i]
                end

                # Calculate average vorticity between all eight nodes of the cell
                W.= 0
                for xj in 1:2
                    for yj in 1:2
                        for zj in 1:2

                            nj[1] = xj
                            nj[2] = yj
                            nj[3] = zj

                            # Build coordinates of this node
                            for j in 1:3
                                X[j] = nj[j]==1 ? Xlo[j] : Xup[j]
                            end

                            # Vorticity at this node
                            Wfun!(W, X)

                        end
                    end
                end
                W ./= 8

                # Vortex strength
                Gamma .= W
                Gamma .*= vol
                magGamma = norm(Gamma)

                # Add particle
                if magGamma >= abs(minmagGamma)
                    vpm.add_particle(pfield, Xc, Gamma, sigma; vol=vol)
                    Nphi += 1
                end

            end
        end
    end

    if verbose; println("\t"^(v_lvl)*"Number of particles: $(Nphi)"); end

    return Nphi
end





"""
Return the function of the analytic vorticity field of an elliptic ring with
a given distribution `zeta` (default is Gaussian).

NOTE: Use AR=1 to define a circular ring.
"""
function generate_Wfun(circulation::Real,
                        R::Real, AR::Real, Rcross::Real;
                        O::Vector{<:Real}=zeros(3), Oaxis=I,
                        zeta=(r,Rcross) -> 1/(pi*Rcross^2) * exp(-r^2/Rcross^2))


    a = R*sqrt(AR)                      # Semi-major axis
    b = R/sqrt(AR)                      # Semi-minor axis

    across = Rcross*sqrt(AR)
    bcross = Rcross/sqrt(AR)

    function W_fun!(out, X)

        # Project X unto the ring's Cartesian coordinate system
        x = (X[1]-O[1])*Oaxis[1, 1] + (X[2]-O[2])*Oaxis[2, 1] + (X[3]-O[3])*Oaxis[3, 1]
        y = (X[1]-O[1])*Oaxis[1, 2] + (X[2]-O[2])*Oaxis[2, 2] + (X[3]-O[3])*Oaxis[3, 2]
        z = (X[1]-O[1])*Oaxis[1, 3] + (X[2]-O[2])*Oaxis[2, 3] + (X[3]-O[3])*Oaxis[3, 3]

        # Wrap space to make it ellipsoid
        # NOTE: I'm not sure I'm converting things correctly here
        x /= sqrt(AR)
        y *= sqrt(AR)

        # Convert Cartesian to cylindrical coordinates
        rho = sqrt(x^2 + y^2)               # Radius from centroid
        # phi = atan(y, x)                    # Angle from centroid
        phi = atan(x, y)                    # Angle from centroid

        # NOTE: Nor here
        # Reff = sqrt( (a*sin(phi))^2 + (b*cos(phi))^2 ) # Effective radius at this angle
        Reff = R
        Rcrosseff = sqrt( (across*cos(phi))^2 + (bcross*sin(phi))^2 )

        # Iterate over foci
        for f in (-1, 1)

            # Convert cylindrical to toroidal coordinates
            r = sqrt( (rho + f*Reff)^2 + z^2 ) # Radius from centerline
            theta = asin(z/r)              # Angle from centerline

            # Calculate vorticity magnitude
            # NOTE: Nor here
            magW = f*circulation*zeta(r * Rcross/Rcrosseff, Rcross)

            # Calculate vorticity unitary direction in ring's Cartesian coordiantes
            Txr, Tyr, Tzr = cos(phi), -sin(phi), 0

            # Vorticity unitary direction in global coordinates
            Tx = Txr*Oaxis[1, 1] + Tyr*Oaxis[1, 2] + Tzr*Oaxis[1, 3]
            Ty = Txr*Oaxis[2, 1] + Tyr*Oaxis[2, 2] + Tzr*Oaxis[2, 3]
            Tz = Txr*Oaxis[3, 1] + Tyr*Oaxis[3, 2] + Tzr*Oaxis[3, 3]

            out[1] += magW*Tx
            out[2] += magW*Ty
            out[3] += magW*Tz

        end

        return out
    end

    return W_fun!
end

function generate_Wfun(nrings::Int, circulations,
                        Rs, ARs, Rcrosss, Os, Oaxiss; optargs...)

    W_funs = [ generate_Wfun(circulations[ri],
                                Rs[ri], ARs[ri], Rcrosss[ri];
                                O=Os[ri], Oaxis=Oaxiss[ri],
                                optargs...) for ri in 1:nrings ]

    function W_fun!(out, args...; optargs...)
        for W in W_funs
            W(out, args...; optargs...)
        end

        return out
    end

    return W_fun!
end
