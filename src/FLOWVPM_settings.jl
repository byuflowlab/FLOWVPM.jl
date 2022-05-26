# Stores assorted parameters in a way that the sensitivity solver won't complain about it:
# The sensitivity solver requires all parameters to be numerical, so this struct pretends to only have numbers while
# actually storing other data types. Only the parameters for which derivatives make sense are directly exposed.

mutable struct SolverSettings{T} <: AbstractArray{T,1}

    numerical_parameters
    string_parameters
    bool_parameters
    function_parameters
    other_parameters
    
    # Solver settings
    kernel::Kernel
    UJ::Function
    formulation
    viscous
    outside_functions::Union{Array{Function,1},Nothing}        # Any outside functions needed (VLM, etc)
    outside_function_parameters::Union{Array{T,1},Nothing}     # Any parameters that the outside functions use. To differentiate with respect to these,
                                                #    the outside functions should take these as inputs when called.
    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    sgsmodel                          # Subgrid-scale contributions model
    sgsscaling                        # Scaling factor of SGS contributions
    integration                       # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    #fmm::FMM                                    # Fast-multipole settings

    # Internal properties
    nt::Int                                     # Current time step number
    np::Int
    maxparticles::Int
    
    #=
    # Solver settings
    kernel::Kernel                              # Vortex particle kernel
    UJ::Function                                # Particle-to-particle calculation
    formulation::F                              # VPM formulation
    viscous::V                                  # Viscous scheme
    outside_functions::Array{Function,1}        # Any outside functions needed (VLM, etc)
    outside_function_parameters::Array{T,1}     # Any parameters that the outside functions use. To differentiate with respect to these,
                                                #    the outside functions should take these as inputs when called.

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    sgsmodel::Function                          # Subgrid-scale contributions model
    sgsscaling::Function                        # Scaling factor of SGS contributions
    integration::Function                       # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    relaxation::Function                        # Relaxation scheme
    relax::Bool                                 # Enables relaxation scheme
    rlxf::R                                     # Relaxation factor (fraction of dt)
    fmm::FMM                                    # Fast-multipole settings

    # Internal properties
    nt::Int                                     # Current time step number
    t::R                                        # Current time=#

    discrete_CB_condition_functions::Union{Nothing,Array{Function,1}}
    continuous_CB_condition_functions::Union{Nothing,Array{Function,1}}
    discrete_CB_affect_functions::Union{Nothing,Array{Function,1}}
    continuous_CB_affect_functions::Union{Nothing,Array{Function,1}}
    active_discrete_CBs::Union{Nothing,Array{Bool,1}}
    active_continuous_CBs::Union{Nothing,Array{Bool,1}}

    function SolverSettings{T}(;numerical_parameters=nothing,
                                string_parameters=nothing,
                                bool_parameters=nothing,
                                function_parameters=nothing,
                                other_parameters=nothing,
                                discrete_CB_condition_functions=nothing,
                                discrete_CB_affect_functions=nothing,
                                continuous_CB_condition_functions=nothing,
                                continuous_CB_affect_functions=nothing,
                                kernel=kernel_default,
                                UJ=UJ_fmm,
                                formulation=formulation_default,
                                viscous=nothing,
                                outside_functions=nothing,
                                outside_function_parameters=nothing,
                                Uinf=Uinf_default,
                                sgsmodel=nothing,
                                sgsscaling=nothing,
                                integration=nothing,
                                transposed=true,
                                nt=0,
                                np=0,
                                maxparticles=0) where {T}
        if discrete_CB_condition_functions != nothing || discrete_CB_affect_functions != nothing
            if length(discrete_CB_condition_functions) != length(discrete_CB_affect_functions)
                error("Mismatch in number of discrete conditions and affects!")
            end
            IndicatorBoolsDiscrete = zeros(Bool,length(discrete_CB_condition_functions))
        else
            IndicatorBoolsDiscrete = nothing
        end
        if continuous_CB_condition_functions != nothing || continuous_CB_affect_functions != nothing
            if length(continuous_CB_condition_functions) != length(continuous_CB_affect_functions)
                error("Mismatch in number of continuous conditions and affects!")
            end
            IndicatorBoolsContinuous = zeros(Bool,length(continuous_CB_condition_functions))
        else
            IndicatorBoolsContinuous= nothing
        end
        #=if numerical_parameters != nothing
            T = eltype(numerical_parameters)
        else
            T = Float64
        end=#

        if length(numerical_parameters) === nothing
            @warn("Constructing a new settings struct with no numerical values!")
        end
        if length(numerical_parameters) > 1
            for i=1:length(numerical_parameters)
                if numerical_parameters[i] == undef
                    error("")
                end
            end
        end
        
        new{T}(numerical_parameters,string_parameters,bool_parameters,function_parameters,
               other_parameters,kernel,UJ,formulation,viscous,outside_functions,outside_function_parameters,
               Uinf,sgsmodel,sgsscaling,integration,transposed,nt,np,maxparticles,
               discrete_CB_condition_functions,continuous_CB_condition_functions,
               discrete_CB_affect_functions,continuous_CB_affect_functions,IndicatorBoolsDiscrete,IndicatorBoolsContinuous)
    end

end

# handles callbacks by lumping them together into one function in a [hopefully] differentiable way

function DiscreteCBConditions(u,t,integrator)
    dcbcfs = integrator.p.discrete_CB_condition_functions
    if dcbcfs === nothing
        return false
    end
    for i=1:length(dcbcfs)
        integrator.p.active_discrete_CBs[i] = dcbcfs[i](u,t,integrator)
    end
    sum(integrator.p.active_discrete_CBs) > 0 ? true : false
end

function DiscreteCBAffects!(integrator)
    dcbafs = integrator.p.discrete_CB_affect_functions
    for i=1:length(dcbafs)
        integrator.p.active_discrete_CBs[i] == true ? dcbafs[i](integrator) : nothing
    end
end

function ContinuousCBConditions(u,t,integrator)
    ccbcfs = integrator.p.continuous_CB_condition_functions
    CBvals = ccbcfs !== nothing ? zeros(length(ccbcfs)) : return 1.0
    for i=1:length(out)
        CBvals[i] = ccbcfs[i](u,t,integrator)
        integrator.p.active_continuous_CBs[i] = ccbcfs[i](u,t,integrator) <= 0 ? true : false
    end

    return min(CBvals...)
end

function ContinuousCBAffects!(integrator)
    integrator.p.continuous_CB_affect_functions[idx](integrator)
    ccbafs = integrator.p.continuous_CB_affect_functions
    for i=1:length(ccbafs)
        integrator.p.active_discrete_CBs[i] == true ? ccbafs[i](integrator) : nothing
    end
end

# A few common callbacks: saving and runtime output.

function SaveCondition(u,t,integrator)
    save_path = integrator.p.string_parameters[1]
    (save_path != nothing) ? true : false
end

function SaveAffect!(integrator)
    save_path = integrator.p.string_parameters[1]
    run_name = integrator.p.string_parameters[2]  
    pfield = integrator.u
    t = integrator.t
    integrator.u.nt += 1    
    overwrite_time = integrator.t
    save(pfield, run_name; path=save_path, add_num=true,
                        overwrite_time=overwrite_time)
end
UpdateTimestepAffect!(integrator) = integrator.u.nt += 1
VerboseCondition(u,t,integrator) = true
VerboseAffect!(integrator) = println("Time: $(integrator.t)\tTimestep: $(integrator.u.nt)\tParticles: $(integrator.u.np)")
TrueCondition(u,t,integrator) = true

# Additional methods for arithmatic operations so that DE.jl can run time integration. The details of this are taken from the array interface documentation, since there's
# a very similar case in one of their examples (a struct with array-type data as well as other data that needs to be properly carried over)

# Functions defined to enable using a custom array type (from the docs: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting):

"""
    'Base.size(p::SolverParameters)'
Outputs the number of elements in the differentiation-safe numerical array
"""
Base.size(p::SolverSettings) = p.numerical_parameters !== nothing ? (length(p.numerical_parameters),) : 0
Base.size(::Type{SolverSettings}) = error("Size of Type{SolverSettings} is not well-defined!") # this likely doesn't have well-defined behavior anyway.

"""
    'Base.getindex(p::SolverParameters,i)'
Maps a single-index access to the various entries in the differentiation-safe numerical array
"""
#Base.getindex(p::SolverSettings,i::Int) = p.numerical_parameters[i]
Base.getindex(p::SolverSettings,inds::Vararg{Int,1}) = p.numerical_parameters[inds...]

"""
    'Base.setindex!(p::SolverParameters,val,inds)'
Maps single-indexed variable assignment to the various entries contained in the differentiation-safe numerical array
"""
#Base.setindex!(p::SolverSettings,val,inds::Vararg{Int,1}) = p.numerical_parameters[inds...] = isassigned([val],1) ? val : error("Undefined reference at location $inds when writing $val")

function Base.setindex!(p::SolverSettings,val,inds::Vararg{Int,1})
    p.numerical_parameters[inds...] = val
    return nothing
end

Base.showarg(io::IO,p::SolverSettings,toplevel) = print(io,typeof(p)) # Custom printing

### These next few functions should not be used directly; rather, it is used to implement broadcasting any function onto Particle objects.
# The general format is copied from https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting).
Base.BroadcastStyle(::Type{<:SolverSettings}) = Broadcast.ArrayStyle{SolverSettings}()
"""
    'Base.similar(bc::Broadcasted)'
Unpacks broadcasted input and constructs a new SolverSettings of similar structure.
"""
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SolverSettings}},::Type{ElType}) where {ElType}
    p = find_settings(bc)
    #println("Constructing a similar settings struct with size $(length(p)) with the broadcasting interface!")
    #out = SolverSettings{ElType}(;numerical_parameters=zeros(ElType,length(p)))
    out = SolverSettings{ElType}(;numerical_parameters=Array{ElType}(undef,length(p)))
    #out.numerical_parameters[1:end] .= zero(ElType)
    #println(out)
    return out
end
"""
    'find_settings(bc)'
Unpacks broadcasted input. This function should probably not be called directly.
"""
find_settings(bc::Base.Broadcast.Broadcasted) = find_settings(bc.args)
find_settings(args::Tuple) = find_settings(find_settings(args[1],Base.tail(args)))
find_settings(x) = x
find_settings(::Tuple{}) = nothing
find_settings(p::SolverSettings,rest) = p
find_settings(::Any,rest) = find_settings(rest)

# Some additional similar() definitions to avoid unwanted type conversion
"""
    'Base.similar(p::SolverSettings)'
Creates a new settings_fname list similar to p but with the numerical entries used by a differential equations solver initialized to zero.
"""
function Base.similar(p::SolverSettings{T}) where T
    #println("Constructing a similar settings struct with size $(length(p)) with no type change!")
    #out = SolverSettings{T}(;numerical_parameters=zeros(T,length(p)))
    out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,length(p)))
    #out.numerical_parameters[1:end] .= T(0)
    copy_misc_settings!(out,p)
    return out
end

"""
    'Base.similar(p::SolverSettings,T2::Type)'
    Creates a new settings list similar to p but with the numerical entries used by a differential equations solver initialized to zero.
Also converts the base type of the array of differention-safe data to T2.
"""
function Base.similar(p::SolverSettings,T2::Type)

    #println("Constructing a similar settings struct with size $(length(p)) and type $(T2)!")
    #out = SolverSettings{T2}(;numerical_parameters=zeros(T2,length(p)))
    out = SolverSettings{T2}(;numerical_parameters=Array{T2}(undef,length(p)))
    #out.numerical_parameters[1:end] .= T2(0)
    copy_misc_settings!(out,p)
    #=out.numerical_parameters = zeros(T2,length(p))
    out.string_parameters = p.string_parameters
    out.bool_parameters = p.bool_parameters
    out.function_parameters = p.function_parameters
    out.other_parameters = p.other_parameters
    out.discrete_CB_condition_functions = p.discrete_CB_condition_functions
    out.continuous_CB_condition_functions = p.continuous_CB_condition_functions
    out.discrete_CB_affect_functions = p.discrete_CB_affect_functions
    out.continuous_CB_affect_functions = p.continuous_CB_affect_functions
    out.active_discrete_CBs = p.active_discrete_CBs
    out.active_continuous_CBs = p.active_continuous_CBs=#
    return out
end

function copy_misc_settings!(p_target::SolverSettings,p_source::SolverSettings) ### pending: this might need to create copies of the input settings

    p_target.string_parameters = p_source.string_parameters
    p_target.bool_parameters = p_source.bool_parameters
    p_target.function_parameters = p_source.function_parameters
    p_target.other_parameters = p_source.other_parameters
    p_target.discrete_CB_condition_functions = p_source.discrete_CB_condition_functions
    p_target.continuous_CB_condition_functions = p_source.continuous_CB_condition_functions
    p_target.discrete_CB_affect_functions = p_source.discrete_CB_affect_functions
    p_target.continuous_CB_affect_functions = p_source.continuous_CB_affect_functions
    p_target.active_discrete_CBs = p_source.active_discrete_CBs
    p_target.active_continuous_CBs = p_source.active_continuous_CBs

    p_target.kernel = p_source.kernel
    p_target.UJ = p_source.UJ
    p_target.formulation = p_source.formulation
    p_target.viscous = p_source.viscous
    p_target.outside_functions = p_source.outside_functions
    p_target.outside_function_parameters = p_source.outside_function_parameters
    p_target.Uinf = p_source.Uinf
    p_target.sgsmodel = p_source.sgsmodel
    p_target.sgsscaling = p_source.sgsscaling
    p_target.integration = p_source.integration
    p_target.transposed = p_source.transposed
    p_target.nt = p_source.nt
    p_target.np = p_source.np

end

"""
    'SolverSettings(val::T)'
Constructor to handle initializing a SolverSettings struct with a single numerical value; this is done a few times in the DifferentialEquations.jl
internal function calls.
"""
function SolverSettings(N::I) where {I <: Int}
    #println("Constructing a new settings struct with integer size N! Type assumed to be Float64.")
    #SolverSettings{Float64}(;numerical_parameters=zeros(N))
    SolverSettings{Float64}(;numerical_parameters=Array{Float64}(undef,N)) 
end
function SolverSettings(val::T) where {T<:Real}
    error("")
    println("Constructing a new settings struct with size 1 out of a scalar value!")
    SolverSettings{T}(;numerical_parameters=[val])
end

"""
    Base.axes(p::SolverSettings)
Returns a range of allowable indices for accessing SolverPSettings struct data.
"""
Base.axes(p::SolverSettings) = (Base.OneTo(size(p)[1]),)

"""
    'Base.copy(p::SolverSettings)'
Creates a copy of a SolverSettings struct.
"""
function Base.copy(p::SolverSettings{T}) where T
    
    #out = SolverSettings{T}(;numerical_parameters=zeros(T,length(p)))
    #out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,length(p)))
    out = similar(p)
    #println("Constructing a copy settings struct with size $(length(p))!")

    if p.numerical_parameters !== nothing
        out .= p
        #for i=1:length(p.numerical_parameters)
        #    out.numerical_parameters[i] = p.numerical_parameters[i]
        #end
    else
        @warn("Tried to copy from an empty setting struct!")
    end
    copy_misc_settings!(out,p)
    #=out.string_parameters = p.string_parameters
    out.bool_parameters = p.bool_parameters
    out.other_parameters = p.other_parameters
    out.function_parameters = p.function_parameters
    out.discrete_CB_condition_functions = p.discrete_CB_condition_functions
    out.continuous_CB_condition_functions = p.continuous_CB_condition_functions
    out.discrete_CB_affect_functions = p.discrete_CB_affect_functions
    out.continuous_CB_affect_functions = p.continuous_CB_affect_functions
    out.active_discrete_CBs = p.active_discrete_CBs
    out.active_continuous_CBs = p.active_continuous_CBs=#
    return out

end

function Base.zero(p::SolverSettings{T}) where T

    #println("Constructing a zero settings struct with size $(length(p)))!")
    out = SolverSettings{T}(;numerical_parameters=zeros(T,length(p)))
    #out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,length(p)))
    #out.numerical_parameters[1:end] .= T(0)
    copy_misc_settings!(out,p)
    return out

end

function Base.zero(::Type{SolverSettings{T}}) where T

    error("Base.zero(SolverSettings) is not well-defined!")
    #out = SolverSettings{T}()

end

Base.IndexStyle(::Type{<:SolverSettings}) = IndexLinear()

function Base.similar(s::SolverSettings{T},N::Union{Integer,AbstractUnitRange}) where {T}
    #println("Constructing a settings struct with size $(N)!")
    #out = SolverSettings{T}(;numerical_parameters=zeros(T,N))
    out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,N))
    copy_misc_settings!(out,s)
    return out
end

#=function Base.similar(p::SolverSettings{T},len::Int) where {T}
    println("Constructing a similar settings struct with size $len from a given length!")
    #out = SolverSettings{T}(;numerical_parameters=zeros(T,len))
    out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,len))
    #out.numerical_parameters[1:end] .= T2(0)
    copy_misc_settings!(out,p)
    return out
end=#

#=get_np(self::SolverSettings) = self.np
get_tranposed(self::SolverSettings) = self.transposed
get_viscous(self::SolverSettings) = self.viscous
get_Uinf(self::SolverSettings) = self.Uinf=#

#Base.isassigned(s::SolverSettings,i::Integer) = isassigned(s.numerical_parameters,i)