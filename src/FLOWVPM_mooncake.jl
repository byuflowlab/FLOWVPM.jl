# a file for various Mooncake.jl functions that I end up needing.

# functions with adjoints that do not affect the result and that have ill-defined tangent types. mostly output related.
#=
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(create_path), Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(_get_settings), Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(initialize_verbose), Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(joinpath), Vararg}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(close), Any} # file closure is not differentiable
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(Base.close), Any} # file closure is not differentiable
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(save), Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(HDF5.API.try_close_finalizer), Any}
=#
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(Base.close), Vararg}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(close), IOStream}
#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(Base.close), IOStream}

#Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{Mooncake.CoDual{typeof(Core.finalizer), Mooncake.NoFData}, Mooncake.CoDual{typeof(close), Mooncake.NoFData}, Mooncake.CoDual{IOStream, Mooncake.NoFData}} # file closure is not differentiable

# Tuple{Mooncake.CoDual{typeof(Core.finalizer), Mooncake.NoFData}, Mooncake.CoDual{typeof(close), Mooncake.NoFData}, Mooncake.CoDual{IOStream, Mooncake.NoFData}}