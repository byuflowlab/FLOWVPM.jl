# Functions to add:

# Extract state vector + settings from particle field
# Write state vector to particle field
# Methods for compatibility with saving (or just copy old code over)
# Methods for compatibility with existing time integration? I have to rewrite the time integration anyway so this is probably redundant.

function pfield2vec!(vec,settings,pfield::ParticleField{R,F,V,S}) where {R,F,V,S}
    
    plen = 24
    # This is a bandaid solution that basically just reallocates the vector as one with the correct type.
    #if eltype(vec) !== R
    #    vec = similar(vec,R)
    #end

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
    unpack_settings!(settings,pfield)
    println("Displaying vector and pfield types and entries:")
    println("vec type: $(typeof(vec))")
    println("pfield type: $(typeof(pfield))")
    println("vec first three entries: $(vec[1:3])")
    println("pfield first three entries: $(pfield.particles[1].X)")
    println(" ")
    return nothing

end

# Essentially a constructor for a vector that contains the particle field data.
function pfield2vec(pfield::ParticleField{R,F,V,S}) where {R,F,V,S}

    plen = 24
    #len = pfield.np*plen
    len = pfield.maxparticles*plen
    #vec = Vector{R}(undef,len)
    vec = zeros(R,len)
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
    settings = unpack_settings(pfield)

    return vec,settings

end

function pfield2vec(pfield::ParticleField{R,F,V,S},settings) where {R,F,V,S}

    plen = 24
    len = pfield.maxparticles*plen
    vec = zeros(R,len)
    for i=1:pfield.np
        i0 = (i-1)*plen
        #vec[i0+1:i0+3] .= pfield.particles[i].X[1:3]
        vec[i0+1] = pfield.particles[i].X[1]
        vec[i0+2] = pfield.particles[i].X[2]
        vec[i0+3] = pfield.particles[i].X[3]
        #vec[i0+4:i0+6] .= pfield.particles[i].Gamma[1:3]
        vec[i0+4] = pfield.particles[i].Gamma[1]
        vec[i0+5] = pfield.particles[i].Gamma[2]
        vec[i0+6] = pfield.particles[i].Gamma[3]
        vec[i0+7] = pfield.particles[i].sigma[1]
        vec[i0+8] = pfield.particles[i].vol[1]
        vec[i0+9] = pfield.particles[i].circulation[1]
        vec[i0+10:i0+12] .= pfield.particles[i].U[1:3]
        vec[i0+13:i0+21] .= pfield.particles[i].J[1:9]
        vec[i0+22:i0+24] .= pfield.particles[i].C[1:3]
    end
    settings = unpack_settings!(settings,pfield)

    return vec

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
    
    plen = 24
    pack_settings!(pfield,settings)
    for i=1:pfield.np
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
    return nothing
end

function vec2pfield(vec, settings)
    plen = 24
    #R = eltype(vec) # does not necessarily return the correct AD type. specifically, the number of partial derivative entries is not read
    R = typeof(vec[1])
    pfield = ParticleField(settings[1];R=R) # out of place, so initialize a new particle field.
    pack_settings!(pfield,settings)
    #if pfield.np > 0
        for i=1:settings[4]
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
    #end
    return pfield

end

function vec2pfield(vec, settings, T::Type)
    plen = 24
    #R = eltype(vec) # does not necessarily return the correct AD type. specifically, the number of partial derivative entries is not read
    #R = typeof(vec[1])
    pfield = ParticleField(settings[1];R=T) # out of place, so initialize a new particle field.
    pack_settings!(pfield,settings)
    #if pfield.np > 0
        for i=1:settings[4]
            i0 = (i-1)*plen
            #pfield.particles[i].X[1:3] .= vec[i0+1:i0+3]
            pfield.particles[i].X[1] = vec[i0+1]
            pfield.particles[i].X[2] = vec[i0+2]
            pfield.particles[i].X[3] = vec[i0+3]
            #pfield.particles[i].Gamma[1:3] .= vec[i0+4:i0+6]
            pfield.particles[i].Gamma[1] = vec[i0+4]
            pfield.particles[i].Gamma[2] = vec[i0+5]
            pfield.particles[i].Gamma[3] = vec[i0+6]
            pfield.particles[i].sigma[1] = vec[i0+7]
            pfield.particles[i].vol[1] = vec[i0+8]
            pfield.particles[i].circulation[1] = vec[i0+9]
            pfield.particles[i].U .= vec[i0+10:i0+12]
            pfield.particles[i].J[1:9] .= vec[i0+13:i0+21]
            pfield.particles[i].C .= vec[i0+22:i0+24]
        end
    #end
    return pfield

end

Base.eltype(pfield::ParticleField{R,F,V,S}) where {R,F,V,S} = R

unpack_settings(pfield) = [pfield.maxparticles,pfield.formulation,pfield.viscous,
                           pfield.np,pfield.nt,pfield.t,pfield.kernel,pfield.UJ,
                           pfield.Uinf,pfield.SFS,pfield.transposed,pfield.np_f]
                           # pfield.np_hist, pfield.relaxation] # uncomment if relaxation is re-enabled.

function unpack_settings!(settings,pfield)

    settings[1] = pfield.maxparticles
    settings[2] = pfield.formulation
    settings[3] = pfield.viscous
    settings[4] = pfield.np
    settings[5] = pfield.nt
    settings[6] = pfield.t
    settings[7] = pfield.kernel
    settings[8] = pfield.UJ
    settings[9] = pfield.Uinf
    settings[10] = pfield.SFS
    settings[11] = pfield.transposed
    settings[12] = pfield.np_f
    # settings[14] = pfield.relaxation # uncomment if relaxation is re-enabled.
    return nothing

end

# out of place version of pack_settings! doesn't make sense to use so I only wrote an in-place one.
function pack_settings!(pfield,settings)

    pfield.maxparticles = settings[1]
    pfield.formulation = settings[2]
    pfield.viscous = settings[3]
    pfield.np = settings[4]
    pfield.nt = settings[5]
    pfield.t = settings[6]
    pfield.kernel = settings[7]
    pfield.UJ = settings[8]
    pfield.Uinf = settings[9]
    pfield.SFS = settings[10]
    pfield.transposed = settings[11]
    pfield.np_f = settings[12]
    # pfield.relaxation = settings[14] # uncomment if relaxation is re-enabled.
    return nothing

end