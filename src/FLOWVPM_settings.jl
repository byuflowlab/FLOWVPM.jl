# Stores assorted parameters in a way that the sensitivity solver won't complain about it:
# The sensitivity solver requires all parameters to be numerical, so this struct pretends to only have numbers while
# actually storing other data types. Only the parameters for which derivatives make sense are directly exposed.

### TODO: The active particle count probably needs to be one of the parameters in numerical_parameters.
# This approach will work (or at least, it worked in the MWE) even through the active particle count is not actually a parameter.
# I'll probably reserve entry 1 in numerical_parameters for this. I also need to write the interface
# for passing user-defined parameters back into user-defined functions.

mutable struct SolverSettings{T} <: AbstractArray{T,1}

    # Values exposed to the vector interface. Contains some solver setup values as well as parameters associated with user-defined functions.
    numerical_parameters::Union{Array{T,1},Nothing} # All the numerical parameters that should be exposed to the derivative interface.
                                     # Current storage scheme: [new particle parameters, other runtime function parameters]

    # Solver settings:
    kernel::Kernel
    UJ::Function
    formulation
    viscous

    # User-defined functions:
    new_particle_function::Union{Function,Nothing} # should take (u,p,t) as input. should output ([new_particle_locations],[new_particle_Gammas];optional: [viscosity info])
    npfp_shape                      # shape of npfp inputs
    npfp_ind::Int                        # First entry in parameters for npfp
    other_function::Union{Function,Nothing}        # other functions as needed. should take (u,p,t) as input and should modify u in-place
    ofp_shape                       # shape of ofp inputs
    ofp_ind::Int                         # First entry in parameters for ofp

    # Optional inputs
    Uinf::Function                              # Uniform freestream function Uinf(t)
    sgsmodel                          # Subgrid-scale contributions model
    sgsscaling                        # Scaling factor of SGS contributions
    integration                       # Time integration scheme
    transposed::Bool                            # Transposed vortex stretch scheme
    #fmm::FMM                                    # Fast-multipole settings
    verbose::Int                      # verbosity level
    save_parameters::Union{Array{String,1},Nothing} # Holds information used for saving to a file. Contents: [save_path, run_name]

    # Internal properties
    nt::Int                                     # Current time step number
    np_ind::Int                                      # index of np in parameters list
    # np has to be a parameter. Everything works in the forward direction either way, but the particle count also needs to be decremented in the reverse direction.
    # If it's just an internal variable in SolverSettings, it continues being incremented and this causes out-of-bounds errors.
    #np::Int
    maxparticles::Int

    #=
    # Optional inputs
    relaxation::Function                        # Relaxation scheme
    relax::Bool                                 # Enables relaxation scheme
    rlxf::R                                     # Relaxation factor (fraction of dt)
    =#

    function SolverSettings{T}(;kernel=kernel_default,
                                UJ=UJ_direct_3!,
                                formulation=formulation_default,
                                viscous=nothing,
                                new_particle_function=nothing,
                                new_particle_function_parameters=nothing,
                                other_function=nothing,
                                other_function_parameters=nothing,
                                save_parameters=nothing,
                                Uinf=Uinf_default,
                                sgsmodel=nothing,
                                sgsscaling=nothing,
                                integration=nothing,
                                transposed=true,
                                verbose=0,
                                nt=0,
                                np=0,
                                maxparticles=0) where {T}

        npfp_shape = 0
        ofp_shape = 0
        npfp_length = 0
        ofp_length = 0
        ofp_ind = 0
        npfp_ind = 0
        p_length = 1
        np_ind = 1
        
        if new_particle_function_parameters !== nothing
            npfp_shape = size.(new_particle_function_parameters)
            npfp_length = sum(length.(new_particle_function_parameters))
            p_length += sum(npfp_length)
            println(p_length)
            npfp_ind = 1 + np_ind
        end
        if other_function_parameters !== nothing
            ofp_shape = size.(other_function_parameters)
            ofp_length = sum(length.(other_function_parameters))
            p_length += sum(ofp_length)
            println(p_length)
            ofp_ind = 1 + npfp_ind
        end

        numerical_parameters = zeros(T,p_length)
        numerical_parameters[np_ind] = np

        for i=1:npfp_length
            numerical_parameters[i + (npfp_ind-1)] = new_particle_function_parameters[i]
        end
        for i=1:ofp_length
            numerical_parameters[i + (ofp_ind-1)] = other_function_parameters[i]
        end

        println("allocated a settings struct with $(length(numerical_parameters)) entries!")
        
        new{T}(numerical_parameters,kernel,UJ,formulation,viscous,new_particle_function,npfp_shape,npfp_ind,other_function,ofp_shape,ofp_ind,
               Uinf,sgsmodel,sgsscaling,integration,transposed,verbose,save_parameters,nt,np_ind,maxparticles)
    end

end

# A few common callbacks: saving and runtime output.

function SaveCondition(u,t,integrator)
    save_path = integrator.p.string_parameters[1]
    (save_path != nothing) ? true : false
end

function SaveAffect!(integrator)
    if !(typeof(integrator.p) <: SolverSettings)
        return nothing
    end
    if integrator.p.save_parameters === nothing
        return nothing
    end
    save_path = integrator.p.save_parameters[1]
    run_name = integrator.p.save_parameters[2]  
    pfield = integrator.u
    #integrator.u.nt += 1    
    integrator.p.nt += 1
    overwrite_time = integrator.t
    save(pfield,integrator.p,run_name;path=save_path,add_num=true,overwrite_time=overwrite_time)
    #save(pfield, run_name; path=save_path, add_num=true, overwrite_time=overwrite_time)
end
UpdateTimestepAffect!(integrator) = integrator.u.nt += 1
VerboseAffect!(integrator) = typeof(integrator.p) <: SolverSettings ? println("Time: $(integrator.t)\tTimestep: $(integrator.p.nt)\tParticles: $(get_np(integrator.p))") : println("Time: $(integrator.t.value)")
TrueCondition(u,t,integrator) = true
FalseCondition(u,t,integrator) = false
VerboseCondition(u,t,integrator) = TrueCondition(u,t,integrator)
NullAffect!(integrator) = nothing

# Additional methods for arithmatic operations so that DE.jl can run time integration. The details of this are taken from the array interface documentation, since there's
# a very similar case in one of their examples (a struct with array-type data as well as other data that needs to be properly carried over)

# Functions defined to enable using a custom array type (from the docs: https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting):

"""
    'Base.size(p::SolverParameters)'
Outputs the number of elements in the differentiation-safe numerical array
"""
Base.size(p::SolverSettings) = p.numerical_parameters !== nothing ? (typeof(p) <: SolverSettings ? (length(p.numerical_parameters),) : (length(p.value.numerical_parameters),)) : error("tried to get size of empty parameter set!")
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
    #out = SolverSettings{ElType}(;numerical_parameters=Array{ElType}(undef,length(p)))
    out = SolverSettings{ElType}()
    out.numerical_parameters = Array{ElType}(undef,length(p))
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
    #out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,length(p)))
    out = SolverSettings{T}()
    out.numerical_parameters = Array{T}(undef,length(p))
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
    #out = SolverSettings{T2}(;numerical_parameters=Array{T2}(undef,length(p)))
    out = SolverSettings{T2}()
    out.numerical_parameters = Array{T2}(undef,length(p))
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

    p_target.kernel = p_source.kernel
    p_target.UJ = p_source.UJ
    p_target.formulation = p_source.formulation
    p_target.viscous = p_source.viscous
    p_target.new_particle_function = p_source.new_particle_function
    p_target.npfp_shape = p_source.npfp_shape
    p_target.other_function = p_source.other_function
    p_target.ofp_shape = p_source.ofp_shape
    p_target.Uinf = p_source.Uinf
    p_target.sgsmodel = p_source.sgsmodel
    p_target.sgsscaling = p_source.sgsscaling
    p_target.integration = p_source.integration
    p_target.transposed = p_source.transposed
    p_target.verbose = p_source.verbose
    p_target.save_parameters = p_source.save_parameters
    p_target.nt = p_source.nt
    #p_target.np = p_source.np
    p_target.maxparticles = p_source.maxparticles

    p_target.npfp_ind = p_source.npfp_ind
    p_target.ofp_ind = p_source.ofp_ind
    p_target.np_ind = p_source.np_ind

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
    out = SolverSettings{T}()
    out.numerical_parameters = zeros(T,length(p))
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

### TODO: fix the typos here
function Base.similar(s::SolverSettings{T},N::Union{Integer,AbstractUnitRange}) where {T}
    #println("Constructing a settings struct with size $(N)!")
    #out = SolverSettings{T}(;numerical_parameters=zeros(T,N))
    #out = SolverSettings{T}(;numerical_parameters=Array{T}(undef,N))
    out = SolverSettings{T}()
#    out
    copy_misc_settings!(out,s)
    out.numerical_parameters = Array{T}(undef,N)
    #out .= zero(T)
    #println(out)
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

using ForwardDiff

# functions for variable access:
#get_np(self) = (typeof(self) <: ParticleField) ? Int(self.np) : (typeof(self) <: SolverSettings || typeof(self) <: Array) ? self[1] : get_np(self.value)
function get_np(self)
    if typeof(self) <: ParticleField
        return Int(self.np)
    elseif typeof(self) <: SolverSettings || typeof(self) <: Array
        if typeof(self[1]) <: ForwardDiff.Dual
            return self[1].value
        else
            return self[1]
        end
    else
        return get_np(self.value)
    end
end
get_nt(self) = (typeof(self) <: SolverSettings) ? self.nt : self.value.nt
get_maxparticles(self) = (typeof(self) <: SolverSettings) ? self.maxparticles : self.value.maxparticles
get_transposed(self) = (typeof(self) <: SolverSettings) ? self.transposed : self.value.transposed
get_integration_scheme(self) = (typeof(self) <: SolverSettings) ? self.integration : self.value.integration
get_verbose(self) = (typeof(self) <: SolverSettings) ? self.verbose : self.value.verbose
get_UJ(self) = (typeof(self) <: SolverSettings) ? self.UJ : self.value.UJ
get_Uinf(self) = (typeof(self) <: SolverSettings) ? self.Uinf : self.value.Uinf
get_viscous(self) = (typeof(self) <: SolverSettings) ? self.viscous : self.value.viscous
get_kernel(self) = (typeof(self) <: SolverSettings) ? self.kernel : self.value.kernel

increment_np(self) = self[1] += 1

# fmm not currently compatible (as of 3 June 2022) so return nothing
#get_fmm(self) = (typeof(self) <: SolverSettings) ? self.fmm : self.value.fmm
get_fmm(self) = nothing

##

# Maybe setting broadcasting precedence will prevent the type conversion to a vector?
Base.BroadcastStyle(::Broadcast.Style{Array},::Broadcast.Style{SolverSettings}) = Broadcast.Style{SolverSettings}

# TODO:
# Clean up comments and dev code
# Define the full set of settings access functions here rather than in the ParticleField file # done
# Update the SolverSettings constructor to match the additional data in the SolverSettings struct
# See if I can find a nice way to expose numerical data to the differentiation packages while still keeping the struct structure. i.e. without putting a bunch of parameters into a nameless struct that is hard to use.
#    My current idea for this is to keep the struct structure and have a parameter struct whose variables point to the same place as the variables in the rest of the struct.
#    This is an interface issue, since it will work fine if an opaque vector of parameters is used. However, it would be much less user-friendly.