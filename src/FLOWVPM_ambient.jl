#=
Ambient (background) flow: a velocity field u_amb(X, t) and its gradient that move
and stretch the particles without being carried by them (one-way: the wake does
not change the ambient). A `ParticleField` takes, as `Uinf`, either the classic
uniform function `Uinf(t)` (the update adds it; unchanged) or an
`AbstractAmbient`, which [`add_ambient!`](@ref) adds to every particle's U and J
right before each update, at the stage's time: after the stage's U/J evaluation
and its SFS estimate, so the subfilter-scale model, the relaxation and the next
evaluation see the induced field only, while convection and stretching see the
total.
=#

"""
    AbstractAmbient

An ambient flow for [`add_ambient!`](@ref): [`UniformAmbient`](@ref),
[`AmbientField`](@ref), or an [`AmbientHolder`](@ref) filled at run time.
"""
abstract type AbstractAmbient end

"""
    UniformAmbient(velocity)

A uniform ambient, `velocity(t) -> SVector{3}` (or a constant 3-vector): no
gradient, so it only convects.
"""
struct UniformAmbient{F} <: AbstractAmbient
    velocity::F
end
UniformAmbient(v::AbstractVector) = (u = SVector{3}(v); UniformAmbient(t -> u))

"""
    AmbientField(velocity, gradient)

A spatially varying ambient: `velocity(X, t) -> SVector{3}` and
`gradient(X, t) -> SMatrix{3,3}` with `G[i, j] = ∂u_i/∂x_j`, both evaluated at a
particle position `X::SVector{3}`. The gradient enters the stretching.
"""
struct AmbientField{FU,FG} <: AbstractAmbient
    velocity::FU
    gradient::FG
end

"""
    AmbientHolder(ambient = nothing; gradient = true, t_offset = 0.0)

A slot for an ambient a driver sets at run time (`holder.ambient = ...`):
a particle field's `Uinf` is a type parameter fixed at construction, the
driver's freestream is not known until then. `nothing` is no ambient.
`gradient = false` skips the ambient gradient (a run without stretching).
The ambient is evaluated at `pfield.t + t_offset`, for a driver whose clock
does not start with the field's (`pfield.t` also feeds the SFS time scale, so it
is not reset).
"""
mutable struct AmbientHolder <: AbstractAmbient
    ambient::Any
    gradient::Bool
    t_offset::Float64
end
AmbientHolder(ambient = nothing; gradient = true, t_offset = 0.0) = AmbientHolder(ambient, gradient, t_offset)

# Where the classic code evaluates `pfield.Uinf(t)`: with an ambient, the uniform
# term the update adds itself is zero (the ambient went in through add_ambient!).
(a::AbstractAmbient)(t) = SVector(0.0, 0.0, 0.0)

"""
    add_ambient!(pfield, t; gradient = true)

Add the ambient velocity at each particle to its U and (with `gradient`) the
ambient gradient to its J (`J[(j-1)*3 + i] += ∂u_i/∂x_j`), for the update that
follows. A no-op for a classic `Uinf(t)`, which the update adds itself. Returns
the 12 x np increments when a gradient went into J (so a caller can take it out
again; see `remove_ambient_gradient!`), else `nothing`.
"""
add_ambient!(pfield, t; gradient::Bool = true) = _add_ambient!(pfield, pfield.Uinf, t, gradient)

"""
    remove_ambient_gradient!(pfield, increments)

Subtract the gradient [`add_ambient!`](@ref) added: the Euler relaxation must align
the particles with the vorticity they carry, not with the ambient's.
"""
remove_ambient_gradient!(pfield, ::Nothing) = nothing
function remove_ambient_gradient!(pfield, d)
    np = size(d, 2)
    view(pfield.particles, J_INDEX, 1:np) .-= view(d, 4:12, :)
    return nothing
end

_add_ambient!(pfield, ::Any, t, gradient::Bool) = nothing
_add_ambient!(pfield, a::AmbientHolder, t, gradient::Bool) =
    a.ambient === nothing ? nothing : _add_ambient!(pfield, a.ambient, t + a.t_offset, gradient && a.gradient)

function _add_ambient!(pfield, a::UniformAmbient, t, gradient::Bool)
    np = get_np(pfield)
    np == 0 && return nothing
    R = eltype(pfield.particles)
    u = a.velocity(t)
    view(pfield.particles, U_INDEX, 1:np) .+= (R(u[1]), R(u[2]), R(u[3]))
    return nothing
end

# Evaluated on the host (threaded) and added in one pass; a device field brings its
# positions over and the 12 x np increments back, once per stage.
function _add_ambient!(pfield, a::AmbientField, t, gradient::Bool)
    np = get_np(pfield)
    np == 0 && return nothing
    P = pfield.particles
    R = eltype(P)
    X = Array(view(P, X_INDEX, 1:np))
    inc = Matrix{R}(undef, 12, np)
    Threads.@threads for i in 1:np
        x = SVector{3}(X[1, i], X[2, i], X[3, i])
        u = a.velocity(x, t)
        @inbounds begin
            inc[1, i] = u[1]; inc[2, i] = u[2]; inc[3, i] = u[3]
            if gradient
                G = a.gradient(x, t)
                for j in 1:3, k in 1:3
                    inc[3 + (j - 1) * 3 + k, i] = G[k, j]
                end
            end
        end
    end
    d = P isa Array ? inc : copyto!(similar(P, R, 12, np), inc)
    view(P, U_INDEX, 1:np) .+= view(d, 1:3, :)
    gradient || return nothing
    view(P, J_INDEX, 1:np) .+= view(d, 4:12, :)
    return d
end
