#=##############################################################################
# DESCRIPTION
    Gaussian-erf vortex particles as a FastMultipole rectangular kernel: every
    particle induces velocity (and optionally its gradient) at a distinct set of
    target points through `FastMultipole.direct_rectangular!`, on host arrays or,
    with a KernelAbstractions backend loaded, on device arrays.
=###############################################################################

"""
    RectangularGaussianErfVortex()

`FastMultipole.direct_rectangular!` kernel for gaussianerf-regularized vortex
particles. Source row layout (7 rows, the leading rows of the particle matrix):

| rows | content |
|---|---|
| 1:3 | particle position |
| 4:6 | vector strength Gamma |
| 7   | smoothing radius sigma (> 0) |

The pair math is the one `fmm.direct!` sums for a `ParticleField` with the
gaussianerf kernel (`g_dgdr_gauserf`, `custom_erf`), in the same order. The only
excluded pair is exact coincidence
(`r2 == 0`). On a device the sources are visited in the same order, but FMA
contraction and the device transcendentals differ by about 1 ulp per pair, so
compare device and host at about 1e-13 (U) and 1e-12 (J) relative in Float64.
Velocity and gradient only; no scalar potential.
"""
struct RectangularGaussianErfVortex <: fmm.AbstractRectangularKernel end

fmm.rect_source_rows(::RectangularGaussianErfVortex) = 7

@inline function fmm.rect_pair(::RectangularGaussianErfVortex, target::SVector{3,T},
        sources, q, ::Val{GRAD}, ::Val{POT}) where {T,GRAD,POT}
    @inbounds begin
        dx = target[1] - sources[1, q]
        dy = target[2] - sources[2, q]
        dz = target[3] - sources[3, q]
        gamma_x = sources[4, q]; gamma_y = sources[5, q]; gamma_z = sources[6, q]
        sigma = sources[7, q]
    end
    r2 = dx*dx + dy*dy + dz*dz
    zu = zero(SVector{3,T}); zJ = zero(SMatrix{3,3,T,9})
    iszero(r2) && return zu, zJ, zero(T)          # exact coincidence only
    r = sqrt(r2)
    g_sgm, dg_sgmdr = g_dgdr_gauserf(r / sigma)
    r3inv = one(T) / (r2 * r)
    c4 = T(const4)
    crss1 = -c4 * r3inv * (dy*gamma_z - dz*gamma_y)
    crss2 = -c4 * r3inv * (dz*gamma_x - dx*gamma_z)
    crss3 = -c4 * r3inv * (dx*gamma_y - dy*gamma_x)
    u = SVector{3,T}(g_sgm * crss1, g_sgm * crss2, g_sgm * crss3)
    GRAD || return u, zJ, zero(T)
    aux = dg_sgmdr / (sigma*r) - 3*g_sgm / r2
    aux2 = -c4 * g_sgm * r3inv
    # column-major J[i, j] = du_i/dx_j
    J = SMatrix{3,3,T,9}(
        aux * crss1 * dx, aux * crss2 * dx - aux2 * gamma_z, aux * crss3 * dx + aux2 * gamma_y,
        aux * crss1 * dy + aux2 * gamma_z, aux * crss2 * dy, aux * crss3 * dy - aux2 * gamma_x,
        aux * crss1 * dz - aux2 * gamma_y, aux * crss2 * dz + aux2 * gamma_x, aux * crss3 * dz)
    return u, J, zero(T)
end
