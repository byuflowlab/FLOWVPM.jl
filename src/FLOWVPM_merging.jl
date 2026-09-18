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

# Sparse cell binning: candidates are labeled by packing their integer cell
# coordinates into a single Int64 key, then grouped by sorting. Only occupied
# cells are materialized (CSR over the sorted unique keys), so memory is
# O(#candidates) regardless of spatial extent. A dense (extent/cell_size)^3
# grid here has caused OutOfMemoryError whenever a single runaway particle
# stretched the bounding box.
const CELL_COORD_BITS = 21
const CELL_COORD_MAX = (1 << CELL_COORD_BITS) - 1

# Coordinates beyond CELL_COORD_MAX cells from the origin collapse into the
# boundary cell. That only groups extreme outliers together, and every pair is
# still distance-checked before acting, so results are unaffected. Comparing
# in float before converting also absorbs Inf/huge values that would throw in
# floor(Int, ·).
@inline function _cell_coord(delta::Real, inv_cell::Real)
    r = delta * inv_cell
    r >= CELL_COORD_MAX && return CELL_COORD_MAX
    r <= 0 && return 0
    return floor(Int, r)
end

@inline _pack_cell_key(ix::Int, iy::Int, iz::Int) =
    ix | (iy << CELL_COORD_BITS) | (iz << (2 * CELL_COORD_BITS))

"""
Bin `candidate_indices` into cells of size `cell_size` anchored at `origin`.
On return `sorted_indices` holds the candidates grouped by cell (ascending
particle index within a cell), `unique_keys[1:n_cells]` the sorted packed key
of each occupied cell, and `offsets[c]+1 : offsets[c+1]` cell `c`'s range in
`sorted_indices`. `keys[i]` is left holding the packed key of candidate `i`
(`keys` must have length ≥ the largest candidate index). Returns `n_cells`,
the number of occupied cells. All buffers end up O(#candidates)-sized.
"""
function _build_cell_list!(
    sorted_indices::Vector{Int},
    offsets::Vector{Int},
    unique_keys::Vector{Int},
    keys::Vector{Int},
    candidate_indices::Vector{Int},
    pfield::ParticleField,
    cell_size::Real,
    origin,
)
    inv_cell = inv(cell_size)
    @inbounds for i in candidate_indices
        ix = _cell_coord(pfield.particles[1, i] - origin[1], inv_cell)
        iy = _cell_coord(pfield.particles[2, i] - origin[2], inv_cell)
        iz = _cell_coord(pfield.particles[3, i] - origin[3], inv_cell)
        keys[i] = _pack_cell_key(ix, iy, iz)
    end

    n = length(candidate_indices)
    resize!(sorted_indices, n)
    copyto!(sorted_indices, candidate_indices)
    # Tie-breaking on the index keeps within-cell order ascending in particle
    # index, matching the stable counting sort this replaces. QuickSort is
    # in-place: no allocation in the hot loop.
    sort!(sorted_indices; alg=QuickSort, by=i -> (keys[i], i))

    resize!(unique_keys, n)
    resize!(offsets, n + 1)
    n_cells = 0
    prev = typemin(Int)
    @inbounds for t in 1:n
        k = keys[sorted_indices[t]]
        if k != prev
            n_cells += 1
            unique_keys[n_cells] = k
            offsets[n_cells] = t - 1
            prev = k
        end
    end
    offsets[n_cells + 1] = n
    return n_cells
end

# Range of the cell with packed key `key` in `sorted_indices`, or an empty
# range if that cell is unoccupied.
@inline function _cell_range(offsets::Vector{Int}, unique_keys::Vector{Int}, n_cells::Int, key::Int)
    c = searchsortedfirst(unique_keys, key, 1, n_cells, Base.Order.Forward)
    @inbounds if c <= n_cells && unique_keys[c] == key
        return (offsets[c] + 1):offsets[c + 1]
    end
    return 1:0
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
    circulation_weighted_sum,
    sigma_sum,
    candidates_by_root::Vector{Int},
    range_start::Int,
    range_end::Int,
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

    use_weighted = weight_sum > weight_threshold
    if use_weighted
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

    # Second-moment merged σ (026 §22.1, Ryan ruling 2026-09-16):
    #     σ_new² = ⟨σ²⟩_w + (1/3)⟨|xᵢ − x̄|²⟩_w,  w = |Γ|
    # — the moment-optimal single-isotropic replacement (total strength and
    # centroid are already conserved; the one free parameter matches the trace
    # of the pair's vorticity second moment). Coincident limit → σ, killing
    # the +26%/pair σ-pump of the old volume-conserving cbrt(Σσ³) rule.
    # Variance is taken about the placement actually used above (weighted
    # centroid, or the unweighted fallback with w = 1 when Σ|Γ| ≈ 0 — same
    # branch condition, so the means stay mutually consistent). The same loop
    # gathers the |α|-weighted ledger means for the merge lineage (§22.2).
    # Heap-allocated like `acc` below so dual-number eltypes don't blow the
    # stack: [Σw, Σwσ², Σw|Δx|², Σwσ₀², Σw·dvisc, Σw·drvpm,
    # Σw·(±axis_x,y,z) (sign-aligned to the running sum, same convention as
    # `_rsplit_accumulate!`), Σw·weight].
    rs = pfield.resolution_split
    ext = zeros(R, 10)
    for k in range_start:range_end
        i = candidates_by_root[k]
        w = one(R)
        if use_weighted
            g_x = pfield.particles[GAMMA_INDEX.start, i]
            g_y = pfield.particles[GAMMA_INDEX.start + 1, i]
            g_z = pfield.particles[GAMMA_INDEX.start + 2, i]
            w = sqrt(g_x * g_x + g_y * g_y + g_z * g_z)
        end
        d_x = pfield.particles[X_INDEX.start, i] - x_x
        d_y = pfield.particles[X_INDEX.start + 1, i] - x_y
        d_z = pfield.particles[X_INDEX.start + 2, i] - x_z
        sigma_i = pfield.particles[SIGMA_INDEX, i]
        ext[1] += w
        ext[2] += w * sigma_i * sigma_i
        ext[3] += w * (d_x * d_x + d_y * d_y + d_z * d_z)
        if rs !== nothing
            s0 = _rsplit_slot_value(rs.sigma_0, i)
            ext[4] += w * s0 * s0
            ext[5] += w * _rsplit_slot_value(rs.dvisc, i)
            ext[6] += w * _rsplit_slot_value(rs.drvpm, i)
            a_x = _rsplit_slot_value(rs.axis, 1, i)
            a_y = _rsplit_slot_value(rs.axis, 2, i)
            a_z = _rsplit_slot_value(rs.axis, 3, i)
            sgn = ext[7] * a_x + ext[8] * a_y + ext[9] * a_z < zeroR ?
                  -one(R) : one(R)
            ext[7] += sgn * w * a_x
            ext[8] += sgn * w * a_y
            ext[9] += sgn * w * a_z
            ext[10] += w * _rsplit_slot_value(rs.weight, i)
        end
    end
    # ext[1] > 0 always: use_weighted implies Σ|Γ| > threshold; otherwise
    # every member contributes w = 1 and n_members ≥ 2.
    inv_w = inv(ext[1])
    separation_var = ext[3] * inv_w / 3
    sigma = sqrt(ext[2] * inv_w + separation_var)
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

    # Coincident-limit ledger lineage (026 §22.2, Ryan rulings 2026-09-17/18),
    # replacing the old reset (σ₀ := merged σ, accumulators := 0) that made
    # merge-driven growth invisible to the fractional split triggers: every
    # ledger line — σ₀², each Δσ² accumulator, and the stretch-axis direction
    # state — becomes the SAME |α|-weighted mean over members, and the
    # separation term (1/3)⟨|Δx|²⟩_w — in σ² but NOT σ₀² — is credited to
    # drvpm (grow side → compress/tri3 trigger). Merging equals is
    # maturity-neutral; only genuine coarsening advances the split clock.
    rs === nothing || _rsplit_merge_lineage!(rs, representative,
        ext[4] * inv_w, ext[5] * inv_w, ext[6] * inv_w + separation_var,
        ext[7] * inv_w, ext[8] * inv_w, ext[9] * inv_w, ext[10] * inv_w)

    return nothing
end

# Accumulate all per-cluster scalar sums for a single root, then finalize.
# Members of `root` live contiguously in `candidates_by_root[range_start:range_end]`
# (preserving the original `candidate_indices` traversal order, so floating-point
# sums are bit-identical to the historical row-wise accumulation).
# Accumulators are heap-allocated (zeros(R, 20)) rather than stack scalars so that
# dual-number types (e.g. ForwardDiff) don't blow out the stack frame.
function _accumulate_and_finalize_root!(
    pfield::ParticleField,
    candidates_by_root::Vector{Int},
    range_start::Int,
    range_end::Int,
    representative::Int,
)
    R = eltype(pfield.particles)
    zeroR = zero(R)

    if R <: Union{Float32, Float64}
        gamma_x = zeroR; gamma_y = zeroR; gamma_z = zeroR
        x_weighted_x = zeroR; x_weighted_y = zeroR; x_weighted_z = zeroR
        c_weighted_x = zeroR; c_weighted_y = zeroR; c_weighted_z = zeroR
        x_unweighted_x = zeroR; x_unweighted_y = zeroR; x_unweighted_z = zeroR
        c_unweighted_x = zeroR; c_unweighted_y = zeroR; c_unweighted_z = zeroR
        weight_sum = zeroR
        vol_sum = zeroR
        circulation_weighted_sum = zeroR
        sigma_sum = zeroR

        for k in range_start:range_end
            i = candidates_by_root[k]

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

            gamma_x += gamma_i_x
            gamma_y += gamma_i_y
            gamma_z += gamma_i_z
            vol_sum += pfield.particles[VOL_INDEX, i]
            circulation_weighted_sum += sigma * pfield.particles[CIRCULATION_INDEX, i]
            sigma_sum += sigma
            x_unweighted_x += pos_x
            x_unweighted_y += pos_y
            x_unweighted_z += pos_z
            c_unweighted_x += c_i_x
            c_unweighted_y += c_i_y
            c_unweighted_z += c_i_z

            if gamma_mag > zeroR
                weight_sum += gamma_mag
                x_weighted_x += gamma_mag * pos_x
                x_weighted_y += gamma_mag * pos_y
                x_weighted_z += gamma_mag * pos_z
                c_weighted_x += gamma_mag * c_i_x
                c_weighted_y += gamma_mag * c_i_y
                c_weighted_z += gamma_mag * c_i_z
            end
        end

        n_members = range_end - range_start + 1
        _finalize_merged_particle!(
            pfield,
            representative,
            n_members,
            gamma_x, gamma_y, gamma_z,
            x_weighted_x, x_weighted_y, x_weighted_z,
            c_weighted_x, c_weighted_y, c_weighted_z,
            x_unweighted_x, x_unweighted_y, x_unweighted_z,
            c_unweighted_x, c_unweighted_y, c_unweighted_z,
            weight_sum,
            vol_sum,
            circulation_weighted_sum,
            sigma_sum,
            candidates_by_root,
            range_start,
            range_end,
        )
        return nothing
    end

    acc = zeros(R, 20)

    for k in range_start:range_end
        i = candidates_by_root[k]

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

        acc[1]  += gamma_i_x
        acc[2]  += gamma_i_y
        acc[3]  += gamma_i_z
        acc[17] += pfield.particles[VOL_INDEX, i]
        acc[19] += sigma * pfield.particles[CIRCULATION_INDEX, i]
        acc[20] += sigma
        acc[10] += pos_x
        acc[11] += pos_y
        acc[12] += pos_z
        acc[13] += c_i_x
        acc[14] += c_i_y
        acc[15] += c_i_z

        if gamma_mag > zeroR
            acc[16] += gamma_mag
            acc[4]  += gamma_mag * pos_x
            acc[5]  += gamma_mag * pos_y
            acc[6]  += gamma_mag * pos_z
            acc[7]  += gamma_mag * c_i_x
            acc[8]  += gamma_mag * c_i_y
            acc[9]  += gamma_mag * c_i_z
        end
    end

    n_members = range_end - range_start + 1
    _finalize_merged_particle!(
        pfield,
        representative,
        n_members,
        acc[1],  acc[2],  acc[3],   # gamma
        acc[4],  acc[5],  acc[6],   # x_weighted
        acc[7],  acc[8],  acc[9],   # c_weighted
        acc[10], acc[11], acc[12],  # x_unweighted
        acc[13], acc[14], acc[15],  # c_unweighted
        acc[16],                    # weight_sum
        acc[17],                    # vol_sum
        acc[19],                    # circulation_weighted_sum
        acc[20],                    # sigma_sum
        candidates_by_root,
        range_start,
        range_end,
    )

    return nothing
end

function _finalize_merge_clusters!(
    pfield::ParticleField,
    ws::MergingWorkspace;
    on_representative::Union{Nothing,Function}=nothing,
)
    np = get_np(pfield)
    candidate_indices = ws.candidate_indices
    parent = ws.parent
    n_candidates = length(candidate_indices)

    # Workspace buffers indexed by raw particle index (0..np-1 maps to 1..np)
    root_count = ws.root_count
    representative = ws.representative
    resize!(root_count, np); fill!(root_count, 0)
    resize!(representative, np)  # written before read; no fill needed

    # Pass 1: compute root for each candidate and count members per root.
    # Track which roots have at least one member ("seen roots") and assign
    # the representative as the minimum-index member of each root.
    roots = ws.roots
    empty!(roots)
    sizehint!(roots, n_candidates)
    for i in candidate_indices
        root = _uf_find!(parent, i)
        if root_count[root] == 0
            push!(roots, root)
            representative[root] = i
        elseif i < representative[root]
            representative[root] = i
        end
        root_count[root] += 1
    end

    # Pass 2: counting-sort candidates by root into CSR layout.
    # root_offset[root+1] = exclusive prefix start for `root`'s members.
    # Stable: members appear in original candidate_indices order.
    root_offset = ws.root_offset
    resize!(root_offset, np + 1); fill!(root_offset, 0)
    for r in roots
        root_offset[r + 1] = root_count[r]
    end
    for i in 2:(np + 1)
        root_offset[i] += root_offset[i - 1]
    end
    # Write cursor reuses root_count (decremented to base, then incremented per write)
    counts = ws.counts  # reuse cell-list cursor buffer
    resize!(counts, np + 1)
    copyto!(counts, 1, root_offset, 1, np + 1)

    candidates_by_root = ws.candidates_by_root
    resize!(candidates_by_root, n_candidates)
    # Cursor for root r lives at counts[r] and starts at root_offset[r]
    # (exclusive prefix). Increment-then-write places members in root r's
    # slot range root_offset[r]+1 : root_offset[r+1].
    for i in candidate_indices
        root = _uf_find!(parent, i)  # cached path-compressed lookup; cheap
        counts[root] += 1
        candidates_by_root[counts[root]] = i
    end

    # Pass 3: for each paired root, accumulate its two members and finalize
    # the representative.
    to_remove = ws.to_remove
    empty!(to_remove)
    sizehint!(to_remove, n_candidates)

    for r in roots
        count = root_count[r]
        count <= 1 && continue

        range_start = root_offset[r] + 1
        range_end = root_offset[r + 1]
        rep = representative[r]

        _accumulate_and_finalize_root!(
            pfield, candidates_by_root, range_start, range_end, rep,
        )

        on_representative === nothing || on_representative(rep)

        # Queue all members except the representative for removal.
        for k in range_start:range_end
            idx = candidates_by_root[k]
            idx == rep && continue
            push!(to_remove, idx)
        end
    end

    # Remove in descending order to keep indices stable.
    sort!(to_remove; rev=true)
    for idx in to_remove
        remove_particle(pfield, idx)
    end

    return length(to_remove)
end

function merge_particles!(
    pfield::ParticleField;
    r_merge::Real=0.5,
    r_hash::Real=-1.0,
    sigma_relative::Bool=true,
    max_sigma_ratio::Real=2.0,
    skip_static::Bool=true,
    verbose::Bool=false,
    gamma_align_cos::Real=-1.0,
    on_representative::Union{Nothing,Function}=nothing,
    event_io::Union{Nothing,IO}=nothing,
    event_tag::Integer=0,
)
    np = get_np(pfield)
    np <= 1 && return 0
    r_merge <= 0 && return 0
    max_sigma_ratio < 1 && return 0

    ws = pfield.merging_workspace
    candidate_indices = ws.candidate_indices
    empty!(candidate_indices)
    sizehint!(candidate_indices, np)

    xmin = typemax(eltype(pfield.particles))
    ymin = typemax(eltype(pfield.particles))
    zmin = typemax(eltype(pfield.particles))
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
        sigma_sum += sigma
    end

    length(candidate_indices) <= 1 && return 0

    effective_r_hash = r_hash < 0.0 ? r_merge : r_hash
    mean_sigma = sigma_sum / length(candidate_indices)
    cell_size = sigma_relative ? effective_r_hash * mean_sigma : effective_r_hash
    if !(cell_size > 0)
        return 0
    end

    sorted_indices = ws.sorted_indices
    offsets = ws.offsets
    unique_keys = ws.counts  # reused as the sorted occupied-cell keys

    keys = ws.keys
    resize!(keys, np)  # written before read for every candidate; no fill needed

    origin = (xmin, ymin, zmin)

    n_cells = _build_cell_list!(sorted_indices, offsets, unique_keys, keys, candidate_indices, pfield, cell_size, origin)

    parent = ws.parent
    rank = ws.rank
    resize!(parent, np)
    resize!(rank, np); fill!(rank, 0)
    @inbounds for i in 1:np
        parent[i] = i
    end

    # Reuse root_count as an assignment marker while selecting disjoint pairs.
    # _finalize_merge_clusters! clears and rebuilds it before grouping roots.
    paired = ws.root_count
    resize!(paired, np); fill!(paired, 0)

    for c in 1:n_cells
        range_start = offsets[c] + 1
        range_stop = offsets[c + 1]

        for a in range_start:range_stop
            ia = sorted_indices[a]
            paired[ia] != 0 && continue

            xi = pfield.particles[1, ia]
            yi = pfield.particles[2, ia]
            zi = pfield.particles[3, ia]
            sigma_i = pfield.particles[SIGMA_INDEX, ia]

            nearest = 0
            nearest_dist2 = zero(sigma_i)

            for b in (a + 1):range_stop
                ib = sorted_indices[b]
                paired[ib] != 0 && continue

                sigma_j = pfield.particles[SIGMA_INDEX, ib]
                sigma_min = min(sigma_i, sigma_j)
                sigma_max = max(sigma_i, sigma_j)
                sigma_min <= 0 && continue
                sigma_max / sigma_min > max_sigma_ratio && continue

                if gamma_align_cos > -1.0
                    gax = pfield.particles[GAMMA_INDEX.start,     ia]
                    gay = pfield.particles[GAMMA_INDEX.start + 1, ia]
                    gaz = pfield.particles[GAMMA_INDEX.start + 2, ia]
                    gbx = pfield.particles[GAMMA_INDEX.start,     ib]
                    gby = pfield.particles[GAMMA_INDEX.start + 1, ib]
                    gbz = pfield.particles[GAMMA_INDEX.start + 2, ib]
                    ma2 = gax*gax + gay*gay + gaz*gaz
                    mb2 = gbx*gbx + gby*gby + gbz*gbz
                    if ma2 > 0 && mb2 > 0
                        cosθ = (gax*gbx + gay*gby + gaz*gbz) / sqrt(ma2 * mb2)
                        cosθ < gamma_align_cos && continue
                    end
                end

                dx = pfield.particles[1, ib] - xi
                dy = pfield.particles[2, ib] - yi
                dz = pfield.particles[3, ib] - zi
                dist2 = dx * dx + dy * dy + dz * dz
                r_pair = sigma_relative ? r_merge * sigma_min : r_merge
                dist2 < r_pair * r_pair || continue

                if nearest == 0 || dist2 < nearest_dist2 ||
                        (dist2 == nearest_dist2 && ib < nearest)
                    nearest = ib
                    nearest_dist2 = dist2
                end
            end

            if nearest != 0
                _uf_union!(parent, rank, ia, nearest)
                paired[ia] = 1
                paired[nearest] = 1
                if event_io !== nothing
                    # Per-event merge telemetry (026 wave-2): one CSV row per
                    # accepted pair, schema step,np,sigma_i,sigma_j,dist
                    sigma_n = pfield.particles[SIGMA_INDEX, nearest]
                    println(event_io, event_tag, ',', np, ',', sigma_i, ',',
                            sigma_n, ',', sqrt(nearest_dist2))
                end
            end
        end
    end

    event_io === nothing || flush(event_io)

    n_removed = _finalize_merge_clusters!(pfield, ws; on_representative=on_representative)

    if verbose && n_removed > 0
        println("Merged $(length(candidate_indices)) candidate particles into $(length(candidate_indices) - n_removed) particles")
    end

    return n_removed
end
