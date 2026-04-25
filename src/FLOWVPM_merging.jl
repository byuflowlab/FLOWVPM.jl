function _uf_find!(parent::Vector{Int}, i::Int)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

function _uf_union!(parent::Vector{Int}, rank::Vector{Int}, i::Int, j::Int)
    ri = _uf_find!(parent, i)
    rj = _uf_find!(parent, j)
    ri == rj && return false

    if rank[ri] < rank[rj]
        parent[ri] = rj
    elseif rank[ri] > rank[rj]
        parent[rj] = ri
    else
        parent[rj] = ri
        rank[ri] += 1
    end

    return true
end

function _build_cell_list!(
    sorted_indices::Vector{Int},
    offsets::Vector{Int},
    keys::Vector{Int},
    candidate_indices::Vector{Int},
    pfield::ParticleField,
    cell_size::Real,
    origin,
    Nx::Int,
    Ny::Int,
    Nz::Int,
)
    n_cells = Nx * Ny * Nz

    fill!(offsets, 0)

    for i in candidate_indices
        ix = clamp(floor(Int, (pfield.particles[1, i] - origin[1]) / cell_size), 0, Nx - 1)
        iy = clamp(floor(Int, (pfield.particles[2, i] - origin[2]) / cell_size), 0, Ny - 1)
        iz = clamp(floor(Int, (pfield.particles[3, i] - origin[3]) / cell_size), 0, Nz - 1)
        key = ix + iy * Nx + iz * Nx * Ny
        keys[i] = key
        offsets[key + 2] += 1
    end

    for i in 2:(n_cells + 1)
        offsets[i] += offsets[i - 1]
    end

    counts = copy(offsets)
    for i in candidate_indices
        key = keys[i] + 1
        counts[key] += 1
        sorted_indices[counts[key]] = i
    end

    return nothing
end

function _cell_triplet(key::Int, Nx::Int, Ny::Int)
    layer = Nx * Ny
    iz = div(key, layer)
    rem_key = key - iz * layer
    iy = div(rem_key, Nx)
    ix = rem_key - iy * Nx
    return ix, iy, iz
end

function _apply_cluster_removals!(pfield::ParticleField, to_remove::Vector{Int})
    sort!(to_remove; rev=true)

    n_removed = 0
    for idx in to_remove
        remove_particle(pfield, idx)
        n_removed += 1
    end

    return n_removed
end

function _finalize_merged_particle!(
    pfield::ParticleField,
    representative::Int,
    n_members::Int,
    gamma_x,
    gamma_y,
    gamma_z,
    x_weighted_x,
    x_weighted_y,
    x_weighted_z,
    c_weighted_x,
    c_weighted_y,
    c_weighted_z,
    x_unweighted_x,
    x_unweighted_y,
    x_unweighted_z,
    c_unweighted_x,
    c_unweighted_y,
    c_unweighted_z,
    weight_sum,
    vol_sum,
    sigma3_sum,
    circulation_weighted_sum,
    sigma_sum,
)
    R = eltype(pfield.particles)
    zeroR = zero(R)
    weight_threshold = sqrt(eps(R))

    x_x = zeroR
    x_y = zeroR
    x_z = zeroR
    c_x = zeroR
    c_y = zeroR
    c_z = zeroR

    if weight_sum > weight_threshold
        inv_weight = inv(weight_sum)
        x_x = x_weighted_x * inv_weight
        x_y = x_weighted_y * inv_weight
        x_z = x_weighted_z * inv_weight
        c_x = c_weighted_x * inv_weight
        c_y = c_weighted_y * inv_weight
        c_z = c_weighted_z * inv_weight
    else
        inv_members = inv(R(n_members))
        x_x = x_unweighted_x * inv_members
        x_y = x_unweighted_y * inv_members
        x_z = x_unweighted_z * inv_members
        c_x = c_unweighted_x * inv_members
        c_y = c_unweighted_y * inv_members
        c_z = c_unweighted_z * inv_members
    end

    sigma = cbrt(sigma3_sum)
    circulation = circulation_weighted_sum / sigma_sum

    set_X(pfield, representative, (x_x, x_y, x_z))
    set_Gamma(pfield, representative, (gamma_x, gamma_y, gamma_z))
    set_sigma(pfield, representative, sigma)
    set_vol(pfield, representative, vol_sum)
    set_circulation(pfield, representative, circulation)
    set_C(pfield, representative, (c_x, c_y, c_z))
    set_U(pfield, representative, zeroR)
    set_vorticity(pfield, representative, zeroR)
    set_J(pfield, representative, zeroR)
    set_PSE(pfield, representative, zeroR)
    set_M(pfield, representative, zeroR)
    set_SFS(pfield, representative, zeroR)
    set_U_prev(pfield, representative, zeroR)

    return nothing
end

function _merge_clusters_aggressive!(
    pfield::ParticleField,
    parent::Vector{Int},
    candidate_indices::Vector{Int},
)
    np = get_np(pfield)
    R = eltype(pfield.particles)
    zeroR = zero(R)

    root_count = zeros(Int, np)
    representative = zeros(Int, np)
    roots = Int[]
    sizehint!(roots, length(candidate_indices))

    gamma_x = fill(zeroR, np)
    gamma_y = fill(zeroR, np)
    gamma_z = fill(zeroR, np)
    x_weighted_x = fill(zeroR, np)
    x_weighted_y = fill(zeroR, np)
    x_weighted_z = fill(zeroR, np)
    c_weighted_x = fill(zeroR, np)
    c_weighted_y = fill(zeroR, np)
    c_weighted_z = fill(zeroR, np)
    x_unweighted_x = fill(zeroR, np)
    x_unweighted_y = fill(zeroR, np)
    x_unweighted_z = fill(zeroR, np)
    c_unweighted_x = fill(zeroR, np)
    c_unweighted_y = fill(zeroR, np)
    c_unweighted_z = fill(zeroR, np)
    weight_sum = fill(zeroR, np)
    vol_sum = fill(zeroR, np)
    sigma3_sum = fill(zeroR, np)
    circulation_weighted_sum = fill(zeroR, np)
    sigma_sum = fill(zeroR, np)

    for i in candidate_indices
        root = _uf_find!(parent, i)
        if root_count[root] == 0
            push!(roots, root)
            representative[root] = i
        else
            representative[root] = min(representative[root], i)
        end
        root_count[root] += 1

        gamma_i_x = pfield.particles[GAMMA_INDEX.start, i]
        gamma_i_y = pfield.particles[GAMMA_INDEX.start + 1, i]
        gamma_i_z = pfield.particles[GAMMA_INDEX.start + 2, i]
        pos_x = pfield.particles[X_INDEX.start, i]
        pos_y = pfield.particles[X_INDEX.start + 1, i]
        pos_z = pfield.particles[X_INDEX.start + 2, i]
        c_i_x = pfield.particles[C_INDEX.start, i]
        c_i_y = pfield.particles[C_INDEX.start + 1, i]
        c_i_z = pfield.particles[C_INDEX.start + 2, i]
        gamma_mag = sqrt(gamma_i_x * gamma_i_x + gamma_i_y * gamma_i_y + gamma_i_z * gamma_i_z)
        sigma = pfield.particles[SIGMA_INDEX, i]

        gamma_x[root] += gamma_i_x
        gamma_y[root] += gamma_i_y
        gamma_z[root] += gamma_i_z
        vol_sum[root] += pfield.particles[VOL_INDEX, i]
        sigma3_sum[root] += sigma * sigma * sigma
        circulation_weighted_sum[root] += sigma * pfield.particles[CIRCULATION_INDEX, i]
        sigma_sum[root] += sigma

        x_unweighted_x[root] += pos_x
        x_unweighted_y[root] += pos_y
        x_unweighted_z[root] += pos_z
        c_unweighted_x[root] += c_i_x
        c_unweighted_y[root] += c_i_y
        c_unweighted_z[root] += c_i_z

        if gamma_mag > zeroR
            weight_sum[root] += gamma_mag
            x_weighted_x[root] += gamma_mag * pos_x
            x_weighted_y[root] += gamma_mag * pos_y
            x_weighted_z[root] += gamma_mag * pos_z
            c_weighted_x[root] += gamma_mag * c_i_x
            c_weighted_y[root] += gamma_mag * c_i_y
            c_weighted_z[root] += gamma_mag * c_i_z
        end
    end

    to_remove = Int[]
    sizehint!(to_remove, length(candidate_indices) - 1)

    for root in roots
        count = root_count[root]
        count <= 1 && continue

        rep = representative[root]
        _finalize_merged_particle!(
            pfield,
            rep,
            count,
            gamma_x[root],
            gamma_y[root],
            gamma_z[root],
            x_weighted_x[root],
            x_weighted_y[root],
            x_weighted_z[root],
            c_weighted_x[root],
            c_weighted_y[root],
            c_weighted_z[root],
            x_unweighted_x[root],
            x_unweighted_y[root],
            x_unweighted_z[root],
            c_unweighted_x[root],
            c_unweighted_y[root],
            c_unweighted_z[root],
            weight_sum[root],
            vol_sum[root],
            sigma3_sum[root],
            circulation_weighted_sum[root],
            sigma_sum[root],
        )
    end

    for i in candidate_indices
        root = _uf_find!(parent, i)
        root_count[root] <= 1 && continue
        i == representative[root] && continue
        push!(to_remove, i)
    end

    return _apply_cluster_removals!(pfield, to_remove)
end

function merge_particles!(
    pfield::ParticleField;
    r_merge::Real=0.5,
    r_hash::Real=-1.0,
    sigma_relative::Bool=true,
    check_neighboring_cells::Bool=true,
    max_sigma_ratio::Real=2.0,
    skip_static::Bool=true,
    verbose::Bool=false,
)
    np = get_np(pfield)
    np <= 1 && return 0
    r_merge <= 0 && return 0
    max_sigma_ratio < 1 && return 0

    candidate_indices = Int[]
    sizehint!(candidate_indices, np)

    xmin = typemax(eltype(pfield.particles))
    ymin = typemax(eltype(pfield.particles))
    zmin = typemax(eltype(pfield.particles))
    xmax = typemin(eltype(pfield.particles))
    ymax = typemin(eltype(pfield.particles))
    zmax = typemin(eltype(pfield.particles))
    sigma_sum = zero(eltype(pfield.particles))

    for i in 1:np
        if skip_static && get_static(pfield, i)
            continue
        end

        push!(candidate_indices, i)

        x = pfield.particles[1, i]
        y = pfield.particles[2, i]
        z = pfield.particles[3, i]
        sigma = pfield.particles[SIGMA_INDEX, i]

        xmin = min(xmin, x)
        ymin = min(ymin, y)
        zmin = min(zmin, z)
        xmax = max(xmax, x)
        ymax = max(ymax, y)
        zmax = max(zmax, z)
        sigma_sum += sigma
    end

    length(candidate_indices) <= 1 && return 0

    effective_r_hash = r_hash < 0.0 ? r_merge : r_hash
    mean_sigma = sigma_sum / length(candidate_indices)
    cell_size = sigma_relative ? effective_r_hash * mean_sigma : effective_r_hash
    if !(cell_size > 0)
        return 0
    end

    Nx = max(1, floor(Int, (xmax - xmin) / cell_size) + 1)
    Ny = max(1, floor(Int, (ymax - ymin) / cell_size) + 1)
    Nz = max(1, floor(Int, (zmax - zmin) / cell_size) + 1)
    n_cells = Nx * Ny * Nz

    sorted_indices = Vector{Int}(undef, length(candidate_indices))
    offsets = zeros(Int, n_cells + 1)
    keys = zeros(Int, np)
    origin = (xmin, ymin, zmin)

    _build_cell_list!(sorted_indices, offsets, keys, candidate_indices, pfield, cell_size, origin, Nx, Ny, Nz)

    parent = collect(1:np)
    rank = zeros(Int, np)

    for key in 0:(n_cells - 1)
        range_start = offsets[key + 1] + 1
        range_stop = offsets[key + 2]
        range_start > range_stop && continue

        cell_indices = @view sorted_indices[range_start:range_stop]

        for a in range_start:range_stop
            ia = sorted_indices[a]
            xi = pfield.particles[1, ia]
            yi = pfield.particles[2, ia]
            zi = pfield.particles[3, ia]
            sigma_i = pfield.particles[SIGMA_INDEX, ia]

            for b in (a + 1):range_stop
                ib = sorted_indices[b]
                sigma_j = pfield.particles[SIGMA_INDEX, ib]
                sigma_min = min(sigma_i, sigma_j)
                sigma_max = max(sigma_i, sigma_j)
                sigma_min <= 0 && continue
                sigma_max / sigma_min > max_sigma_ratio && continue

                dx = pfield.particles[1, ib] - xi
                dy = pfield.particles[2, ib] - yi
                dz = pfield.particles[3, ib] - zi
                dist2 = dx * dx + dy * dy + dz * dz
                r_pair = sigma_relative ? r_merge * sigma_min : r_merge
                dist2 < r_pair * r_pair && _uf_union!(parent, rank, ia, ib)
            end
        end

        if check_neighboring_cells

            ix, iy, iz = _cell_triplet(key, Nx, Ny)

            for dz in -1:1, dy in -1:1, dx in -1:1
                dx == 0 && dy == 0 && dz == 0 && continue

                nix = ix + dx
                niy = iy + dy
                niz = iz + dz

                (0 <= nix < Nx) || continue
                (0 <= niy < Ny) || continue
                (0 <= niz < Nz) || continue

                neighbor_key = nix + niy * Nx + niz * Nx * Ny
                neighbor_key <= key && continue

                neighbor_start = offsets[neighbor_key + 1] + 1
                neighbor_stop = offsets[neighbor_key + 2]
                neighbor_start > neighbor_stop && continue

                for ia in cell_indices
                    xi = pfield.particles[1, ia]
                    yi = pfield.particles[2, ia]
                    zi = pfield.particles[3, ia]
                    sigma_i = pfield.particles[SIGMA_INDEX, ia]

                    for b in neighbor_start:neighbor_stop
                        ib = sorted_indices[b]
                        sigma_j = pfield.particles[SIGMA_INDEX, ib]
                        sigma_min = min(sigma_i, sigma_j)
                        sigma_max = max(sigma_i, sigma_j)
                        sigma_min <= 0 && continue
                        sigma_max / sigma_min > max_sigma_ratio && continue

                        dx = pfield.particles[1, ib] - xi
                        dy = pfield.particles[2, ib] - yi
                        dz = pfield.particles[3, ib] - zi
                        dist2 = dx * dx + dy * dy + dz * dz
                        r_pair = sigma_relative ? r_merge * sigma_min : r_merge
                        dist2 < r_pair * r_pair && _uf_union!(parent, rank, ia, ib)
                    end
                end
            end
        end
    end

    n_removed = _merge_clusters_aggressive!(pfield, parent, candidate_indices)

    if verbose && n_removed > 0
        println("Merged $(length(candidate_indices)) candidate particles into $(length(candidate_indices) - n_removed) particles")
    end

    return n_removed
end
