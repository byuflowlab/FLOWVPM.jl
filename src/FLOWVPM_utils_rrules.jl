# utils doesn't have any actual mathematics, but it has a bunch of code sections that AD should not try to differentiate.

# this ensures that doesn't happen.

# Saving a simulation to a file is non-differentiable.
function save(
    self::ParticleField{TF, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any},
    file_name::String; path::String="",
    add_num::Bool=true, num::Int64=-1, createpath::Bool=false,
    overwrite_time=nothing) where TF <: Union{ForwardDiff.Dual, ReverseDiff.TrackedReal}

    return nothing

end