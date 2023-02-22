# Functions to add:

# Extract state vector + settings from particle field
# Write state vector to particle field
# Methods for compatibility with saving (or just copy old code over)
# Methods for compatibility with existing time integration? I have to rewrite the time integration anyway so this is probably redundant.

function get_state_vector(pfield::T) where T <: ParticleField

    plen = 24
    len = pfield.np*plen
    out = Vector{eltype(pfield.particles[1].X)}(undef,len)
    for i=1:pfield.np
        i0 = (i-1)*plen
        out[i0+1:i0+3] .= pfield.particles[i].X[1:3]
        out[i0+4:i0+6] .= pfield.particles[i].Gamma[1:3]
        out[i0+7] = pfield.particles[i].sigma[1]
        out[i0+8] = pfield.particles[i].vol[1]
        out[i0+9] = pfield.particles[i].circulation[1]
        out[i0+10:i0+12] .= pfield.particles[i].U[1:3]
        out[i0+13:i0+21] .= pfield.particles[i].J[1:9]
        out[i0+22:i0+24] .= pfield.particles[i].C[1:3]
    end
    return out

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

function write_state_vector(state, pfield::ParticleField)

end