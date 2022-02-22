#=##############################################################################
# DESCRIPTION
    Fast-multipole parameters.

# AUTHORSHIP
  * Author    : Eduardo J Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Aug 2020
  * Copyright : Eduardo J Alvarez. All rights reserved.
=###############################################################################

################################################################################
# FMM STRUCT
################################################################################
"""
    `FMM(; p::Int=4, ncrit::Int=50, theta::Real=0.4, phi::Real=0.3)`

Parameters for FMM solver.

# Arguments
* `p`       : Order of multipole expansion (number of terms).
* `ncrit`   : Maximum number of particles per leaf.
* `theta`   : Neighborhood criterion. This criterion defines the distance
                where the far field starts. The criterion is that if θ*r < R1+R2
                the interaction between two cells is resolved through P2P, where
                r is the distance between cell centers, and R1 and R2 are each
                cell radius. This means that at θ=1, P2P is done only on cells
                that have overlap; at θ=0.5, P2P is done on cells that their
                distance is less than double R1+R2; at θ=0.25, P2P is done on
                cells that their distance is less than four times R1+R2; at
                θ=0, P2P is done on cells all cells.
* `phi`     : Regularizing neighborhood criterion. This criterion avoid
                approximating interactions with the singular-FMM between
                regularized particles that are sufficiently close to each other
                across cell boundaries. Used together with the θ-criterion, P2P
                is performed between two cells if φ < σ/dx, where σ is the
                average smoothing radius in between all particles in both cells,
                and dx is the distance between cell boundaries
                ( dx = r-(R1+R2) ). This means that at φ = 1, P2P is done on
                cells with boundaries closer than the average smoothing radius;
                at φ = 0.5, P2P is done on cells closer than two times the
                smoothing radius; at φ = 0.25, P2P is done on cells closer than
                four times the smoothing radius.
"""
mutable struct FMM
  # Optional user inputs
  p::Int32                        # Multipole expansion order
  ncrit::Int32                    # Max number of particles per leaf
  theta::RealFMM                  # Neighborhood criterion
  phi::RealFMM                    # Regularizing neighborhood criterion

  FMM(; p=4, ncrit=50, theta=0.4, phi=1/3) = new(p, ncrit, theta, phi)
end
