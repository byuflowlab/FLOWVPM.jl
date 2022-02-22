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

function GroundField(pfield::ParticleField, Gamma_basis; ground_unit_normal = [0.0, 0.0, 1.0], A_epsilon=1e-3)
    # initialize arrays
    np = pfield.np
    A = zeros(np, np)
    b = zeros(np)

    # build A matrix
    update_A!(A, pfield, ground_unit_normal; epsilon = A_epsilon)

    return GroundField(pfield, A, b, Gamma_basis, ground_unit_normal, [A_epsilon])
end

function GroundField(xs, ys, zs, Gamma_basis, overlap; optargs...)
    gfield = build_gfield(xs, ys, zs, Gamma_basis, overlap)

    return GroundField(gfield, Gamma_basis; optargs...)
end

function build_gfield(xs, ys, zs, Gamma_basis, overlap)
    # prepare offset
    dof = length(Gamma_basis)
    lx = length(xs)
    ly = length(ys)
    offset_x = lx%dof == 0 ? 1 : 0
    offset_y = (lx*ly)%dof == 0 ? 1 : 0

    # build particle field
    gfield = ParticleField(length(xs) * length(ys) * length(zs))

    # add particles
    counter = 0
    for (zi,z) in enumerate(zs)
        if length(zs) > 1
            dz = zi > 1 ? zs[zi] - zs[zi-1] : zs[zi+1] - zs[zi]
        else
            dz = 0.0
        end
        counter += offset_y
        for (yi,y) in enumerate(ys)
            if length(ys) > 1
                dy = yi > 1 ? ys[yi] - ys[yi-1] : ys[yi+1] - ys[yi]
            else
                dy = 0.0
            end
            counter += offset_x
            for (xi,x) in enumerate(xs)
                if length(xs) > 1
                    dx = xi > 1 ? xs[xi] - xs[xi-1] : xs[xi+1] - xs[xi]
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

    return gfield
end

function update_A!(A, pfield, ground_unit_normal; epsilon = 1e-3)
    # check sizes
    np = pfield.np
    @assert np == size(A)[1] "Ground pfield.np = $(pfield.np) and A matrix size $(size(A)) do not match."

    # reset field velocities
    _reset_particles(pfield)

    for (j,pj) in enumerate(iterator(pfield))
        # update circulation
        # pj.Gamma .= Gamma_basis[(j-1)%dof + 1]

        # get induced velocity
        UJ_direct([pj], iterator(pfield), pfield.kernel)

        # iterate over particles
        for (i,pi) in enumerate(iterator(pfield))
            normal_vi = sum(pi.U .* ground_unit_normal)
            # precondition by zeroing small entries
            A[i, j] = abs(normal_vi) < epsilon ? 0 : normal_vi
        end

        # reset particles
        _reset_particles(pfield)
    end
    # # cycle through basis vectors
    # for (i_Gamma, Gamma_hat) in enumerate(Gamma_basis)
    #     # reset particle field
    #     _reset_particles(pfield)

    #     for (j,pj) in enumerate(iterator(pfield))
    #         # update circulation
    #         pj.Gamma .= Gamma_hat

    #         # get induced velocity
    #         UJ_direct([pj], iterator(pfield), pfield.kernel)

    #         # iterate over particles
    #         for (i,pi) in enumerate(iterator(pfield))
    #             normal_vi = sum(pi.U .* ground_unit_normal)
    #             # precondition by zeroing small entries
    #             A[(i-1)*dof + i_Gamma, (j-1)*dof + i_Gamma] = abs(normal_vi) < epsilon ? 0 : normal_vi
    #         end

    #         # reset particles
    #         _reset_particles(pfield)
    #     end
    # end
    return nothing
end

"""
Assumes zero-strength ground particles have already been added to `pfield`, as using `transfer_particles!(pfield, gfield)`.

Note: `b` should contain the number of `gfield` particles, which is equal to `gfield.pfield.np`.
"""
function update_b!(pfield::ParticleField, gfield::GroundField)
    # get references
    b = gfield.b
    ground_unit_normal = gfield.ground_unit_normal
    n_locations = gfield.pfield.np
    @assert length(b) == n_locations "Inconsistent lengths: n_particles = $n_particles and length(b) = $(length(b))"

    # get induced velocity from `pfield` at each particle location
    _reset_particles(pfield)
    pfield.UJ(pfield)

    # update b
    np = pfield.np - n_locations
    for i_location=1:n_locations # iterate over all ground particle locations
        local vi = pfield.particles[np+i_location].U # get the velocity at the corresponding merged particle
        b[i_location] = -sum(vi .* ground_unit_normal)
    end
    return nothing
end

"""
Transfer particles from the ground field to the particle field with zero circulation.
"""
function transfer_zeroed_particles!(target::ParticleField, source::ParticleField; static::Bool=true)
    for p in iterator(source)
        local X = deepcopy(p.X)
        local Gamma = deepcopy(p.Gamma)
        local sigma = p.sigma[1]
        add_particle(target, X, zeros(3), sigma; static=static)
    end
    # # get number of particle locations
    # dof = length(gfield.Gamma_basis)
    # n_locations = Int(gfield.pfield.np / dof) # number of ground particle locations

    # # loop over particle locations (there are `dof` particles per location)
    # for i_location=1:n_locations
    #     local X = deepcopy(gfield.pfield.particles[(i_location-1) * dof + 1].X)
    #     local sigma = deepcopy(gfield.pfield.particles[(i_location-1) * dof + 1].sigma)
    #     add_particle(pfield, X, zeros(3), sigma)
    #     # for i_dof = 1:dof
    #     #     pfield.particles[pfield.np].Gamma .+= gfield.pfield[(i_location-1) * dof + i_dof].Gamma
    #     # end
    # end

    return nothing
end

function remove_ground_particles!(pfield::ParticleField, gfield::GroundField)
    np = pfield.np - gfield.pfield.np
    for ip = pfield.np:-1:np+1
        remove_particle(pfield, ip)
    end
    return nothing
end

function solve_Gammas(A,b)
    Gammas = A \ b
    return Gammas
end

"""
Adds ground particles of appropriate strength to `pfield`.
"""
function ground_effect!(pfield::ParticleField, gfield::GroundField; save=true, name="", savepath="")
    # get references
    A = gfield.A
    b = gfield.b
    ngp = gfield.pfield.np

    transfer_zeroed_particles!(pfield, gfield.pfield; static=true)

    np = pfield.np - ngp

    update_b!(pfield, gfield)

    Gammas = solve_Gammas(A, b)

    # update pfield ground Gammas
    for (i_location, p) in enumerate(pfield.particles[np+1:pfield.np])
        p.Gamma .= gfield.pfield.particles[i_location].Gamma .* Gammas[i_location]
    end

    # save particle field
    if save
        start_i = np+1
        end_i = np+1+ngp
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
