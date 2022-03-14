"""
Contains:

* `pfield::ParticleField`: particle field with one particle at each desired ground particle location.
* `A::Matrix{Float64}`: array of size `pfield.np` by `pfield.np` used to solve for circulation strengths in each basis vector direction.
* `b::Vector{Float64}`: vector of length `pfield.np` used to solve for circulation strengths in each basis vector direction as `A Gamma = b`.
* `Gamma_basis::Vector{Vector{Float64}}`: vector containing unit vectors used to solve for circulation strengths.
* `ground_unit_normal::Vector{Float64}`: vector defines the ground normal vector.
"""
struct GroundField{TF,V}
    pfield::ParticleField{TF, V}
    A::Matrix{TF}
    b::Vector{TF}
    Gamma_basis::Vector{Vector{TF}}
    ground_unit_normal::Vector{TF}
    A_epsilon::Vector{TF}
end

function GroundField(pfield::ParticleField, Gamma_basis; ground_unit_normal = [0.0, 0.0, 1.0], A_epsilon=0.0, optargs...)
    # initialize arrays
    n_sources = get_nsources(pfield)
    A = zeros(n_sources, n_sources)
    b = zeros(n_sources)

    # build A matrix
    update_A!(A, pfield, ground_unit_normal; epsilon = A_epsilon)

    return GroundField(pfield, A, b, Gamma_basis, ground_unit_normal, [A_epsilon])
end

function GroundField(xs_source, ys_source, zs_source, xs_cp, ys_cp, zs_cp; Gamma_basis=[[1.0,0,0], [0,1.0,0]], optargs...)
    gfield = build_gfield(xs_source, ys_source, zs_source, xs_cp, ys_cp, zs_cp; Gamma_basis=Gamma_basis, optargs...)

    return GroundField(gfield, Gamma_basis; optargs...)
end

function build_gfield(xs_source, ys_source, zs_source, xs_cp, ys_cp, zs_cp;
        Gamma_basis = [[1.0,0,0], [0,1.0,0]],
        overlap = 2.0,
        sigma_cp = 1.0,
        optargs...
    )
    # check array lengths
    @assert length(xs_cp) == length(xs_source) "Length of control points and sources inconsistent. Length(xs_cp) = $(length(xs_cp)), length(xs_source) = $(length(xs_source))"
    @assert length(ys_cp) == length(ys_source) "Length of control points and sources inconsistent. Length(ys_cp) = $(length(ys_cp)), length(ys_source) = $(length(ys_source))"
    @assert length(zs_cp) == length(zs_source) "Length of control points and sources inconsistent. Length(zs_cp) = $(length(zs_cp)), length(zs_source) = $(length(zs_source))"

    # prepare offset
    dof = length(Gamma_basis)
    lx = length(xs_source)
    ly = length(ys_source)
    lz = length(zs_source)
    offset_x = lx%dof == 0 ? 1 : 0
    offset_y = (lx*ly)%dof == 0 ? 1 : 0

    # build particle field
    gfield = ParticleField(2*lx*ly*lz)

    # add source particles
    counter = 0
    for (zi,z) in enumerate(zs_source)
        if length(zs_source) > 1
            dz = zi > 1 ? zs_source[zi] - zs_source[zi-1] : zs_source[zi+1] - zs_source[zi]
        else
            dz = 0.0
        end
        counter += offset_y
        for (yi,y) in enumerate(ys_source)
            if length(ys_source) > 1
                dy = yi > 1 ? ys_source[yi] - ys_source[yi-1] : ys_source[yi+1] - ys_source[yi]
            else
                dy = 0.0
            end
            counter += offset_x
            for (xi,x) in enumerate(xs_source)
                if length(xs_source) > 1
                    dx = xi > 1 ? xs_source[xi] - xs_source[xi-1] : xs_source[xi+1] - xs_source[xi]
                else
                    dx = 0.0
                end
                # calculate sigma
                sigma = overlap * sqrt(dx^2 + dy^2 + dz^2)

                counter += 1
                bi = counter%dof+1
                add_particle(gfield, [x,y,z], Gamma_basis[bi], sigma)
            end
        end
    end

    # add control point particles
    Gamma_cp = zeros(Float64, 3)
    for z in zs_cp
        for y in ys_cp
            for x in xs_cp
                add_particle(gfield, [x,y,z], Gamma_cp, sigma_cp)
            end
        end
    end

    return gfield
end

get_nsources(gfield::ParticleField) = Int(gfield.np/2)

get_sources_i(gfield::ParticleField) = range(1, stop=get_nsources(gfield), step=1)

get_sources(gfield::ParticleField) = iterator(gfield; start_i=1, end_i=get_nsources(gfield))

get_cps_i(gfield::ParticleField) = range(get_nsources(gfield)+1, stop=gfield.np, step=1)

get_cps(gfield::ParticleField) = iterator(gfield; start_i=get_nsources(gfield)+1, end_i=-1)

"""
gfield, i_source
    update_A!

gfield, i_source:

* `A::Matrix{Float64}`- influence matrix, preallocated for speed
*
* `source_i`- iterable of indices of the source ground particles contained in `pfield`
* `source_i`- iterable of indices of the particles representing collocation points in `pfield`

"""
function update_A!(A, gfield, ground_unit_normal; epsilon = 0.0)
    # check sizes
    n_sources = get_nsources(gfield)
    @assert n_sources == size(A)[1] "Number of ground source particles = $(n_sources) and A matrix size $(size(A)) do not match."

    # reset field velocities
    _reset_particles(gfield)

    # get particle iterators for control points and sources
    sources = get_sources(gfield)
    # source_js = get_sources_i(gfield)
    cps = get_cps(gfield)
    cp_is = get_cps_i(gfield)

    # get influence coefficients
    for (j,source_j) in enumerate(sources) # iterate over source particles
        # get induced velocity
        UJ_direct([source_j], cps, gfield.kernel)

        # iterate over particles
        for (i,cp_i) in enumerate(cp_is)
            U = get_U(gfield, cp_i)
            normal_vi = sum(U .* ground_unit_normal)
            A[i, j] = normal_vi
            # precondition by zeroing small entries
            # A[i, j] = abs(normal_vi) < epsilon ? 0 : normal_vi
        end

        # reset particles
        _reset_particles(gfield)
    end

    return nothing
end

"""
Assumes zero-strength ground particles have already been added to `pfield` at each collocation point, as using `transfer_particles!(pfield, gfield)`.

Note: `b` should contain the number of `gfield` source particles, which is equal to `gfield.pfield.np/2`.
"""
function update_b!(pfield::ParticleField, groundfield::GroundField)
    # get references
    b = groundfield.b
    ground_unit_normal = groundfield.ground_unit_normal
    gfield = groundfield.pfield
    n_cps = get_nsources(gfield) # number of cps should equal number of sources for a unique solution to exist
    @assert length(b) == n_cps "Inconsistent lengths: n_particles = $n_particles and length(b) = $(length(b))"

    # get induced velocity from `pfield` at each particle location
    _reset_particles(pfield)
    pfield.UJ(pfield)

    # update b
    np = pfield.np - n_cps

    for (bi_cp, i_cp) in enumerate(np+1:1:np+n_cps) # iterate over all control points
        local vi = get_U(pfield, i_cp) # get the velocity at the corresponding control point
        b[bi_cp] = -sum(vi .* ground_unit_normal)
    end
    return nothing
end

"""
Transfer source particles from the ground field to the particle field with zero circulation.
"""
function transfer_source_particles!(pfield::ParticleField, gfield::ParticleField; static::Bool=true)
    n_sources = get_nsources(gfield)
    for i_source in 1:n_sources
        local X = deepcopy(get_X(gfield, i_source))
        # local Gamma = deepcopy(get_Gamma(source_p))
        local sigma = get_sigma(gfield, i_source)
        add_particle(pfield, X, zeros(3), sigma; static=static)
    end

    return nothing
end

function remove_ground_particles!(pfield::ParticleField, gfield::GroundField)
    np = pfield.np - get_nsources(gfield)
    for ip = pfield.np:-1:np+1
        remove_particle(pfield, ip)
    end
    return nothing
end

function solve_Gammas(A,b)
    Gammas = A \ b
    resid_vec = A*Gammas .- b
    resid = transpose(resid_vec) * resid_vec
    # condition = cond(A)
    println("resid = $resid")
    return Gammas
end

"""
Adds ground particles of appropriate strength to `pfield`.
"""
function ground_effect!(pfield::ParticleField, groundfield::GroundField; save_field=true, name="", savepath="")
    # get references
    A = groundfield.A
    b = groundfield.b
    gfield = groundfield.pfield
    n_sources = get_nsources(gfield)

    transfer_source_particles!(pfield, gfield; static=true)

    np = pfield.np - n_sources

    update_b!(pfield, groundfield)

    Gammas = solve_Gammas(A, b)

    # update pfield ground Gammas
    for (ig_source, ip_source) in enumerate(np+1:1:pfield.np)
        Gamma = get_Gamma(pfield, ip_source)
        Gamma .= get_Gamma(gfield, ig_source) * Gammas[ig_source]
    end

    # save particle field
    if save_field
        start_i = np+1
        end_i = np+n_sources
        save(pfield, name*"_ground"; path=savepath, start_i=start_i, end_i=end_i)
    end
    return nothing
end

# function v_induced(target_x, source_x, source_sigma, source_g, source_Gamma)
#     r_vec = target_x .- source_x
#     r_mag = LA.norm(r_vec)
#     q = source_g(r_mag / source_sigma)
#     cross = LA.cross(r_vec, source_Gamma)
#     vind = -q * cross / r_mag^3 / (4*pi)
#     return vind
# end

# function unit_induced_velocity!(source, target)
#     for ip = 1:target.np
#         target.particles[ip].U .= 0.0
#         for jp = 1:source.np
#             r_vec = target.particles[ip].X .- source.particles[jp].X
#             r_mag = LA.norm(r_vec)
#             if r_mag > 0
#                 sigma = source.particles[jp].sigma[1]
#                 q, dqdr = source.kernel.g_dgdr(r_mag / sigma)
#                 cross = LA.cross(r_vec, source.particles[jp].Gamma)
#                 vind = -q * cross / r_mag^3  / (4*pi)
#                 # vind = -source.kernel.g(r_mag / sigma) / (4*pi) * LA.cross(r_vec, sourceparticles[jp].Gamma) / r_mag^3
#                 target.particles[ip].U .+= vind
#             end
#         end
#     end
#     return nothing
# end

# function update_A!(A, ground_pfield, Gamma_dof, get_unit_Gamma = (jp, Gamma_i) -> UNIT_VECTORS[Gamma_i])
#     for ip in 1:ground_pfield.np # get induced velocity at the ipth particle
#         for jp in 1:ground_pfield.np # induced by the jpth particle
#             local xi = ground_pfield.particles[ip].X
#             local xj = ground_pfield.particles[jp].X
#             local r_vec = xi .- xj
#             local r_mag = LA.norm(r_vec)
#             sigma = ground_pfield.particles[jp].sigma[1]
#             if r_mag > EPSILON # ignore particles less than this distance apart
#                 local q = ground_pfield.kernel.g(r_mag / sigma) / (4*pi)
#                 for Gamma_i in 1:Gamma_dof # which components to include
#                     local cross = LA.cross(r_vec, get_unit_Gamma(jp, Gamma_i)) # get_unit_Gamma function allows flexibility for setting Gamma orientations
#                     A[Gamma_dof * (ip-1) + Gamma_i, Gamma_dof * (jp-1) + Gamma_i] = LA.dot(-q * cross / r_mag^3, z_hat) # assume zhat is the unit normal for now
#                 end
#             else
#                 A[ip,jp] = 0.0
#             end
#         end
#     end
#     return A
# end

# function update_A!(A, pfield::vpm.ParticleField, index_ground_particles, Gamma_dof::Int, get_unit_Gamma = (jp, Gamma_i) -> UNIT_VECTORS[Gamma_i])
#     for (ip, pfield_ip) in enumerate(index_ground_particles) # get induced velocity at the ipth particle
#         for (jp, pfield_jp) in enumerate(index_ground_particles) # induced by the jpth particle
#             local xi = pfield.particles[pfield_ip].X
#             local xj = pfield.particles[pfield_jp].X
#             local r_vec = xi .- xj
#             local r_mag = LA.norm(r_vec)
#             sigma = pfield.particles[pfield_jp].sigma[1]
#             if r_mag > EPSILON # ignore particles less than this distance apart
#                 local q = pfield.kernel.g(r_mag / sigma) / (4*pi)
#                 for Gamma_i in 1:Gamma_dof # which components to include
#                     local cross = LA.cross(r_vec, get_unit_Gamma(jp, Gamma_i)) # get_unit_Gamma function allows flexibility for setting Gamma orientations
#                     A[Gamma_dof * (ip-1) + Gamma_i, Gamma_dof * (jp-1) + Gamma_i] = LA.dot(-q * cross / r_mag^3, z_hat) # assume zhat is the unit normal for now
#                 end
#             else
#                 A[ip,jp] = 0.0
#             end
#         end
#     end
#     return A
# end

# function update_A(ground_pfield::vpm.ParticleField, Gamma_dof::Int, get_unit_Gamma = (jp, Gamma_i) -> UNIT_VECTORS[Gamma_i])
#     np = ground_pfield.np
#     A = zeros(np,np)
#     update_A!(A, ground_pfield, Gamma_dof, get_unit_Gamma)
#     return A
# end
