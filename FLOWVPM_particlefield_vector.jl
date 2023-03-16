# Functions to add:

# Extract state vector + settings from particle field
# Write state vector to particle field
# Methods for compatibility with saving (or just copy old code over)
# Methods for compatibility with existing time integration? I have to rewrite the time integration anyway so this is probably redundant.

function pfield2vec!(vec,settings,pfield::ParticleField{R,F,V,S}) where {R,F,V,S}
    
    error("not currently save to use due to type conversions.")
    plen = 24
    
    # This is a bandaid solution that basically just reallocates the vector as one with the correct type.
    if eltype(vec) !== R
        #vec = R.(vec)
        vec = similar(vec,R)
        #println("overwrote vector state type!")
        println("pfield:$(eltype(pfield))\tvec: $(eltype(vec))")
    end

    for i=1:pfield.np
        i0 = (i-1)*plen
        vec[i0+1:i0+3] .= pfield.particles[i].X[1:3]
        vec[i0+4:i0+6] .= pfield.particles[i].Gamma[1:3]
        vec[i0+7] = pfield.particles[i].sigma[1]
        vec[i0+8] = pfield.particles[i].vol[1]
        vec[i0+9] = pfield.particles[i].circulation[1]
        vec[i0+10:i0+12] .= pfield.particles[i].U[1:3]
        vec[i0+13:i0+21] .= pfield.particles[i].J[1:9]
        vec[i0+22:i0+24] .= pfield.particles[i].C[1:3]
    end
    #=if ForwardDiff.partials(vec[1]) !== ForwardDiff.partials(pfield.particles[1].X[1])
        println("Vector derivative value: $(ForwardDiff.partials(vec[1]))")
        println("Particle field derivative value: $(ForwardDiff.partials(pfield.particles[1].X[1]))")
        println(" ")
    end=#
    #settings .= [pfield.maxparticles,pfield.formulation,pfield.viscous,pfield.np,pfield.nt,pfield.t,pfield.kernel,pfield.UJ,pfield.Uinf,pfield.SFS,pfield.transposed,pfield.relaxation,pfield.t_hist, pfield.np_hist]
    settings .= [pfield.maxparticles,pfield.formulation,pfield.viscous,pfield.np,pfield.nt,pfield.t,pfield.kernel,pfield.UJ,pfield.Uinf,pfield.SFS,pfield.transposed,pfield.t_hist, pfield.np_hist]
    return nothing

end

function pfield2vec(pfield::ParticleField{R,F,V,S}) where {R,F,V,S}

    plen = 24
    len = pfield.np*plen
    vec = Vector{R}(undef,len)
    for i=1:pfield.np
        i0 = (i-1)*plen
        vec[i0+1:i0+3] .= pfield.particles[i].X[1:3]
        vec[i0+4:i0+6] .= pfield.particles[i].Gamma[1:3]
        vec[i0+7] = pfield.particles[i].sigma[1]
        vec[i0+8] = pfield.particles[i].vol[1]
        vec[i0+9] = pfield.particles[i].circulation[1]
        vec[i0+10:i0+12] .= pfield.particles[i].U[1:3]
        vec[i0+13:i0+21] .= pfield.particles[i].J[1:9]
        vec[i0+22:i0+24] .= pfield.particles[i].C[1:3]
    end
    settings = [pfield.maxparticles,pfield.formulation,pfield.viscous,pfield.np,pfield.nt,pfield.t,pfield.kernel,pfield.UJ,pfield.Uinf,pfield.SFS,pfield.transposed,pfield.t_hist, pfield.np_hist]
    return vec,settings

end

get_X(p) = p[1:3]
get_Gamma(p) = p[4:6]
get_sigma(p) = [p[7]]
get_vol(p) = [p[8]]
get_circulation(p) = [p[9]]
get_U(p) = p[10:12]
get_J(p) = p[13:21]
get_J(p,i,j) = p[13 + (i-1)*3 + (j-1)]
get_C(p) = p[22:24]

function vec2pfield!(pfield::ParticleField{R,F,V,S},settings,vec) where {R,F,V,S}
    
    error("not currently save to use due to type conversions.")
    plen = 24
    pfield.np = Int(length(vec)/plen)
    # This is a bandaid type conversion and I'm not sure I trust it. Update: Yeah I think this is an unwanted type conversion.
    if eltype(pfield) !== eltype(vec)
        println("pfield:$(eltype(pfield))\tvec: $(eltype(vec))")
        pfield = similar(pfield,eltype(vec))
    end
    for i=1:pfield.np
        i0 = (i-1)*plen
        #=if !(eltype(vec) <: AbstractFloat)
            #println(ImplicitAD.unpack_dual(vec[i0+1:i0+3])[1])
            pfield.particles[i].X[1:3] .= ImplicitAD.unpack_dual(vec[i0+1:i0+3])[1]
            pfield.particles[i].Gamma[1:3] .= ImplicitAD.unpack_dual(vec[i0+4:i0+6])[1]
            pfield.particles[i].sigma[1] = ImplicitAD.unpack_dual(vec[i0+7])[1]
            pfield.particles[i].vol[1] = ImplicitAD.unpack_dual(vec[i0+8])[1]
            pfield.particles[i].circulation[1] = ImplicitAD.unpack_dual(vec[i0+9])[1]
            pfield.particles[i].U .= ImplicitAD.unpack_dual(vec[i0+10:i0+12])[1]
            pfield.particles[i].J[1:9] .= ImplicitAD.unpack_dual(vec[i0+13:i0+21])[1]
            pfield.particles[i].C .= ImplicitAD.unpack_dual(vec[i0+22:i0+24])[1]
        else
            pfield.particles[i].X[1:3] .= vec[i0+1:i0+3]
            pfield.particles[i].Gamma[1:3] .= vec[i0+4:i0+6]
            pfield.particles[i].sigma[1] = vec[i0+7]
            pfield.particles[i].vol[1] = vec[i0+8]
            pfield.particles[i].circulation[1] = vec[i0+9]
            pfield.particles[i].U .= vec[i0+10:i0+12]
            pfield.particles[i].J[1:9] .= vec[i0+13:i0+21]
            pfield.particles[i].C .= vec[i0+22:i0+24]
        end=#
        pfield.particles[i].X[1:3] .= vec[i0+1:i0+3]
        pfield.particles[i].Gamma[1:3] .= vec[i0+4:i0+6]
        pfield.particles[i].sigma[1] = vec[i0+7]
        pfield.particles[i].vol[1] = vec[i0+8]
        pfield.particles[i].circulation[1] = vec[i0+9]
        pfield.particles[i].U .= vec[i0+10:i0+12]
        pfield.particles[i].J[1:9] .= vec[i0+13:i0+21]
        pfield.particles[i].C .= vec[i0+22:i0+24]
    end
    #=if ForwardDiff.partials(vec[1]) == ForwardDiff.partials(pfield.particles[1].X[1])
        println("Vector derivative value: $(ForwardDiff.partials(vec[1]))")
        println("Particle field derivative value: $(ForwardDiff.partials(pfield.particles[1].X[1]))")
        println(" ")
    end=#
    pfield.maxparticles = settings[1]
    pfield.formulation = settings[2]
    pfield.viscous = settings[3]
    pfield.np = settings[4]
    pfield.nt = settings[5]
    pfield.t = (eltype(settings[6]) <: AbstractFloat) ? settings[6] : settings[6].value
    pfield.kernel = settings[7]
    pfield.UJ = settings[8]
    pfield.Uinf = settings[9]
    pfield.SFS = settings[10]
    pfield.transposed = settings[11]
    #pfield.relaxation = (eltype(settings[12]) == eltype(vec)) ? settings[12] : similar(settings[12],eltype(pfield))
    #pfield.relaxation = similar(settings[12],eltype(pfield))
    #pfield.relaxation = settings[12]
    pfield.t_hist = settings[12]
    pfield.np_hist = settings[13]
    return pfield
    #return nothing
end

function vec2pfield(vec, settings)
    plen = 24
    R = eltype(vec)
    pfield = ParticleField(settings[1];R=R)
    pfield.maxparticles = settings[1]
    pfield.formulation = settings[2]
    pfield.viscous = settings[3]
    pfield.np = settings[4]
    pfield.nt = settings[5]
    pfield.t = settings[6]#(eltype(settings[6]) <: AbstractFloat) ? settings[6] : settings[6].value
    pfield.kernel = settings[7]
    pfield.UJ = settings[8]
    pfield.Uinf = settings[9]
    pfield.SFS = settings[10]
    pfield.transposed = settings[11]
    #pfield.relaxation = settings[12]
    pfield.t_hist = settings[12]
    pfield.np_hist = settings[13]
    for i=1:settings[1]
        i0 = (i-1)*plen
        pfield.particles[i].X[1:3] .= vec[i0+1:i0+3]
        pfield.particles[i].Gamma[1:3] .= vec[i0+4:i0+6]
        pfield.particles[i].sigma[1] = vec[i0+7]
        pfield.particles[i].vol[1] = vec[i0+8]
        pfield.particles[i].circulation[1] = vec[i0+9]
        pfield.particles[i].U .= vec[i0+10:i0+12]
        pfield.particles[i].J[1:9] .= vec[i0+13:i0+21]
        pfield.particles[i].C .= vec[i0+22:i0+24]
    end
    return pfield

end

Base.eltype(pfield::ParticleField{R,F,V,S}) where {R,F,V,S} = R