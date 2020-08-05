#=##############################################################################
# DESCRIPTION
    Simulation of vortex rings.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

"""
  `addvortexring(pfield, Gamma, R, Nphi, nc, rmax; C=[0,0,0], ring_coord_system)`

Adds a particle discretization of a vortex ring to the particle field `pfield`.

  # Arguments
  * `Gamma::Real`     : Circulation of the ring.
  * `R::Real`         : Radius of the ring.
  * `Nphi::Int64`     : Number of cross-sections used to discretize the torus.
  * `nc::Int64`       : Number of layers used to discretize each cross section.
  * `rmax::Real`      : Cross section's radius.
  (OPTIONAL)
  * `extra_nc::Int64` : Adds this many additional layers on top of `nc` with
                        particles of zero strength for viscous diffusion.
  * `C::Array{Real,1}`: Center of the torus.
  * `ring_coord_system`         : Local coordinate system x,y,z of the ring.
  * `lambda::Real`    : Particle overlap (sigma/h), default is 1.25.
"""
function addvortexring(       pfield::vpm.ParticleField,
                              Gamma::Real, R::Real,
                              Nphi::Int64, nc::Int64, rmax::Real;
                              extra_nc::Int64=0,
                              C::Array{<:Real,1}=zeros(3),
                              ring_coord_system::Array{<:Real,2}=Real[1 0 0; 0 1 0; 0 0 1],
                              lambda::Real=1.25,
                        )

    invM = inv(ring_coord_system)

    deltaphi = 2*pi/Nphi            # Angle between cross sections
    h = deltaphi * R                # Perimeter spacing between cross sections
    sigma = lambda*h                # Core size

    # # Cross section's radius
    # rmax = (2*nc + 1)*rl
    # Radius length used to discretize each cross section
    rl = rmax/(2*nc + 1)
    # Vorticity of the ring
    omega = Gamma / (pi*rmax^2)

    # Wrapper that transforms vector in the local to the global coordinate system
    function addparticle(X, vecGamma, sigma, vol)
        Xglob = invM*X + C
        vecGammaglob = invM*vecGamma
        vpm.add_particle(pfield, Xglob, vecGammaglob, sigma; vol=vol)
    end

    # Torus dicretization into cross sections
    for i in 0:Nphi-1

        phi = deltaphi*i                # Angle of this cross section
        Xc = R*[cos(phi), sin(phi), 0]  # Center of the cross section
        D = [-sin(phi), cos(phi), 0]    # Vectorial circulation direction

        # Cross section discretization into layers
        for n in 0:nc+extra_nc

          if n==0 # Particle in the center
            X = Xc                          # Position
            vol = (pi*rl^2) * (deltaphi*R)  # Volume
            vecGamma = omega*vol*D          # Vectorial circulation
            addparticle(X, vecGamma, sigma, vol)

          else    # Layers
            rc = (1 + 12*n^2)/(6*n)*rl  # Center line radius
            r1 = (2*n-1)*rl             # Lower radius
            r2 = (2*n+1)*rl             # Upper radius
            ncells = 8*n                # Number of cells
            deltatheta = 2*pi/ncells    # Angle of cells
            ds = 2*pi*rc/ncells         # Spacing between cells

            # Layer discretization into cells
            for j in 0:(ncells-1)
              theta1 = deltatheta*j           # Left angle
              theta2 = deltatheta*(j+1)       # Right angle
              theta = theta1 + deltatheta/2   # Center angle
              vol = deltaphi*(r2-r1)*(        # Volume of cell
                          deltatheta*R*(r1+r2)/2
                          + (sin(theta2)-sin(theta1))*(r1^2+r1*r2+r2^2)/3
                        )                     # Position of cell
              X = Xc + rc*[cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]

              if n>nc
                vecGamma = 1e-16*D # Particles for viscous diffusion, adds a small
                                   # strength so as to be able to visualize
                                   # smoothing radii
              else
                vecGamma = omega*vol*D
              end
              addparticle(X, vecGamma, sigma, vol)
            end

          end
        end
    end

    return nothing
end

norm(X) = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
