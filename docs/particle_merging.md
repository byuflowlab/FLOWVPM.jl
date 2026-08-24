# Particle Merging

This note describes the current `merge_particles!` implementation. It is meant
as developer theory and maintenance context, not as a user-facing Documenter
page.

## Operation

`merge_particles!` replaces connected clusters of nearby candidate particles
with one representative particle. A particle is a candidate unless it is static
and `skip_static=true`, which is the default.

The implementation first builds an undirected graph over candidate particles.
An edge is added only for a pair that passes all merge gates:

- The pair is considered by the cell-list search. In the current code this
  means both particles are in the same Cartesian hash cell.
- The pair has compatible smoothing radii:

  $$\frac{\max(\sigma_i, \sigma_j)}{\min(\sigma_i, \sigma_j)} \le \rho_{\max}$$

  Pairs with nonpositive $\min(\sigma_i, \sigma_j)$ are rejected.
- The pair distance is below the merge radius:

  $$\lVert\mathbf{x}_i - \mathbf{x}_j\rVert < r_m$$

  when `sigma_relative=false`, or

  $$\lVert\mathbf{x}_i - \mathbf{x}_j\rVert < r_m \min(\sigma_i, \sigma_j)$$

  when `sigma_relative=true`.

The connected components of this graph are the merge clusters. Single-particle
components are left unchanged. Each multi-particle cluster is accumulated once,
written into the minimum-index member of the cluster, and all other members are
removed.

## Merged Quantities

For a cluster with members indexed by $i$, the merged vortex strength is

$$\boldsymbol{\Gamma}_\star = \sum_{i \in \mathcal{C}} \boldsymbol{\Gamma}_i$$

This conserves the stored vortex strength component by component.

The explicit stored particle volume is also summed:

$$V_\star = \sum_{i \in \mathcal{C}} V_i$$

The smoothing radius is chosen by conserving smoothing support volume up to the
common geometric constant:

$$\sigma_\star = \left(\sum_{i \in \mathcal{C}} \sigma_i^3\right)^{1/3}$$

For equal kernel shape, each support volume is proportional to $\sigma_i^3$, so
this preserves the total support measure represented by the merged cluster.

The representative position is a vortex-strength-weighted centroid:

$$
\mathbf{x}_\star =
\frac{\sum_{i \in \mathcal{C}} \lVert\boldsymbol{\Gamma}_i\rVert \mathbf{x}_i}
     {\sum_{i \in \mathcal{C}} \lVert\boldsymbol{\Gamma}_i\rVert}
$$

where $\lVert\boldsymbol{\Gamma}_i\rVert$ is the Euclidean norm of the particle
strength vector. If the cluster has zero total strength weight, the
implementation falls back to the arithmetic mean:

$$\mathbf{x}_\star = \frac{1}{n_\mathcal{C}} \sum_{i \in \mathcal{C}} \mathbf{x}_i$$

The auxiliary vector field $\mathbf{C}$ is merged with the same weights and the
same zero-strength fallback:

$$
\mathbf{C}_\star =
\frac{\sum_{i \in \mathcal{C}} \lVert\boldsymbol{\Gamma}_i\rVert \mathbf{C}_i}
     {\sum_{i \in \mathcal{C}} \lVert\boldsymbol{\Gamma}_i\rVert}
$$

or, for zero total strength weight,

$$\mathbf{C}_\star = \frac{1}{n_\mathcal{C}} \sum_{i \in \mathcal{C}} \mathbf{C}_i$$

The stored circulation value is not summed as a conserved scalar. The current
implementation stores a representative sigma-weighted value:

$$
\kappa_\star =
\frac{\sum_{i \in \mathcal{C}} \sigma_i \kappa_i}
     {\sum_{i \in \mathcal{C}} \sigma_i}
$$

This should be read as a reconstructed representative property of the merged
particle, not as conservation of total circulation.

After the representative is written, scratch and derived fields are reset to
zero:

```text
U, J, vorticity, PSE, M, SFS, U_prev
```

These fields cache quantities from the previous particle layout or time-step
state. Once several particles have been replaced by one particle, the old values
are no longer consistent with the new position, strength, smoothing radius, or
volume. Resetting them forces later solver stages to recompute or repopulate
the derived state for the new particle.

## Conservation and Moment Properties

For moment statements, interpret each particle as a regularized vortex blob.
The pre-merge vorticity represented by one cluster is

$$
\boldsymbol{\omega}_{\mathcal{C}}(\mathbf{y}) =
\sum_{i \in \mathcal{C}}
\boldsymbol{\Gamma}_i \zeta_{\sigma_i}(\mathbf{y} - \mathbf{x}_i)
$$

and the merged particle replaces it by

$$
\boldsymbol{\omega}_\star(\mathbf{y}) =
\boldsymbol{\Gamma}_\star
\zeta_{\sigma_\star}(\mathbf{y} - \mathbf{x}_\star)
$$

Assume a normalized radial kernel with

$$
\int \zeta_\sigma(\mathbf{r})\,d\mathbf{r} = 1,\qquad
\int \mathbf{r}\zeta_\sigma(\mathbf{r})\,d\mathbf{r} = \mathbf{0},\qquad
\int \mathbf{r}\mathbf{r}^T\zeta_\sigma(\mathbf{r})\,d\mathbf{r}
= m_2\sigma^2\mathbf{I}_3
$$

for a kernel-dependent constant $m_2$ and identity tensor $\mathbf{I}_3$.
These are vorticity or blob moments, not velocity-field moments directly, but
they determine impulse and the far-field induced-velocity expansion.

The zeroth vorticity moment is preserved exactly:

$$
\int \boldsymbol{\omega}_\star(\mathbf{y})\,d\mathbf{y}
= \boldsymbol{\Gamma}_\star
= \sum_{i \in \mathcal{C}}\boldsymbol{\Gamma}_i
= \int \boldsymbol{\omega}_{\mathcal{C}}(\mathbf{y})\,d\mathbf{y}
$$

This is the same component-wise conservation of stored vortex strength described
above. The stored particle volume is also preserved exactly,

$$
V_\star = \sum_{i \in \mathcal{C}} V_i
$$

and the kernel support-volume measure is preserved because

$$
\sigma_\star^3 = \sum_{i \in \mathcal{C}}\sigma_i^3
$$

so any common geometric or kernel constant multiplying $\sigma^3$ is unchanged
by the merge.

The stored circulation $\kappa$ is not conserved as
$\sum_i\kappa_i$. The implementation stores the representative value

$$
\kappa_\star =
\frac{\sum_{i \in \mathcal{C}}\sigma_i\kappa_i}
     {\sum_{i \in \mathcal{C}}\sigma_i}
$$

so $\kappa_\star$ is a sigma-weighted property of the replacement particle, not
a conserved scalar total.

The first vorticity moment before merging is

$$
\mathbf{M}_1 =
\int \mathbf{y}\otimes\boldsymbol{\omega}_{\mathcal{C}}(\mathbf{y})\,d\mathbf{y}
= \sum_{i \in \mathcal{C}}\mathbf{x}_i\otimes\boldsymbol{\Gamma}_i
$$

because the centered first moment of each radial kernel is zero. After merging,

$$
\mathbf{M}_{1,\star} =
\int \mathbf{y}\otimes\boldsymbol{\omega}_\star(\mathbf{y})\,d\mathbf{y}
= \mathbf{x}_\star\otimes\boldsymbol{\Gamma}_\star
$$

These are not generally equal. Equality would require the centroid to satisfy
the component-wise moment relation

$$
\mathbf{x}_\star\otimes\sum_i\boldsymbol{\Gamma}_i
= \sum_i\mathbf{x}_i\otimes\boldsymbol{\Gamma}_i
$$

but the implemented centroid is weighted by
$\lVert\boldsymbol{\Gamma}_i\rVert$, not by each signed component of
$\boldsymbol{\Gamma}_i$.

The corresponding linear impulse is

$$
\mathbf{I} =
\frac{1}{2}\int \mathbf{y}\times\boldsymbol{\omega}_{\mathcal{C}}(\mathbf{y})\,d\mathbf{y}
= \frac{1}{2}\sum_{i \in \mathcal{C}}\mathbf{x}_i\times\boldsymbol{\Gamma}_i
$$

while the merged particle gives

$$
\mathbf{I}_\star =
\frac{1}{2}\mathbf{x}_\star\times\boldsymbol{\Gamma}_\star
$$

so impulse is also not generally conserved. A useful special case occurs when
all $\boldsymbol{\Gamma}_i$ are parallel, same-signed multiples of one common
vector. Then the norm-weighted centroid is the same as the strength-weighted
centroid for that common direction, and the current formula preserves the
linear impulse of the cluster.

The second vorticity moment before merging is

$$
\mathbf{M}_2 =
\int \mathbf{y}\mathbf{y}^T\otimes
\boldsymbol{\omega}_{\mathcal{C}}(\mathbf{y})\,d\mathbf{y}
= \sum_{i \in \mathcal{C}}
\left(\mathbf{x}_i\mathbf{x}_i^T + m_2\sigma_i^2\mathbf{I}_3\right)
\otimes\boldsymbol{\Gamma}_i
$$

and the merged particle has

$$
\mathbf{M}_{2,\star} =
\left(\mathbf{x}_\star\mathbf{x}_\star^T
+ m_2\sigma_\star^2\mathbf{I}_3\right)
\otimes\boldsymbol{\Gamma}_\star
$$

These tensors do not generally match. Conserving $\sigma^3$ preserves a support
volume measure, but it does not preserve the $\sigma^2\boldsymbol{\Gamma}$
contribution that appears in the second vorticity moment, nor does the
single-center term reproduce the distributed
$\sum_i\mathbf{x}_i\mathbf{x}_i^T\otimes\boldsymbol{\Gamma}_i$ contribution.

The angular impulse or angular-momentum proxy

$$
\mathbf{A} =
-\frac{1}{3}\int
\mathbf{y}\times\left(\mathbf{y}\times\boldsymbol{\omega}(\mathbf{y})\right)
\,d\mathbf{y}
$$

depends on these second moments. Therefore the current merge rule does not
generally conserve $\mathbf{A}$; degenerate or symmetric clusters may preserve
it only accidentally.

Other fields follow the same interpretation as in `## Merged Quantities`.
$\mathbf{C}$ is a strength-weighted representative value, not a conserved sum.
`U`, `J`, `vorticity`, `PSE`, `M`, `SFS`, and `U_prev` are derived or scratch
state and are deliberately reset rather than conserved. Static particles are
excluded from candidate clusters by default, so their properties are unaffected.

## Alternative Conserving Formulations

This section describes possible alternatives to the current merge rule. These
are theory and design notes only; the implemented `merge_particles!` rule still
conserves $\boldsymbol{\Gamma}$, stored $V$, and support measure $\sigma^3$,
not the $\sigma^2\boldsymbol{\Gamma}$ quantities below.

### Rigid-sphere angular momentum

One possible meaning of particle angular momentum is intrinsic spin angular
momentum. Interpret particle $i$ as a uniformly rotating spherical fluid
element with radius proportional to $\sigma_i$, volume $V_i$, and rigid-body
angular velocity $\boldsymbol{\Omega}_i$. For rigid rotation,

$$
\boldsymbol{\omega}_i = 2\boldsymbol{\Omega}_i,\qquad
\boldsymbol{\Gamma}_i = V_i\boldsymbol{\omega}_i
$$

and the scalar moment of inertia of the spherical element has the form

$$
J_i = \frac{2}{5}V_i\sigma_i^2
$$

up to any common proportionality constant between physical radius and
$\sigma_i$. The intrinsic spin angular momentum is therefore

$$
\mathbf{L}_i
= J_i\boldsymbol{\Omega}_i
= \frac{2}{5}V_i\sigma_i^2\frac{\boldsymbol{\omega}_i}{2}
= \frac{1}{5}\sigma_i^2\boldsymbol{\Gamma}_i .
$$

For a cluster,

$$
\mathbf{L}_{\mathcal{C}}
= \frac{1}{5}\sum_{i\in\mathcal{C}}\sigma_i^2\boldsymbol{\Gamma}_i .
$$

Conserving both the zeroth vorticity moment and this rigid-sphere spin angular
momentum with one isotropic replacement particle would require

$$
\boldsymbol{\Gamma}_\star = \sum_{i\in\mathcal{C}}\boldsymbol{\Gamma}_i,
\qquad
\mathbf{L}_\star
= \frac{1}{5}\sigma_\star^2\boldsymbol{\Gamma}_\star
= \mathbf{L}_{\mathcal{C}} .
$$

A single scalar $\sigma_\star$ can satisfy this vector equation only when
$\sum_i\sigma_i^2\boldsymbol{\Gamma}_i$ is parallel to
$\boldsymbol{\Gamma}_\star$. In the co-linear signed-strength case,
$\boldsymbol{\Gamma}_i = \gamma_i\hat{\boldsymbol{\Gamma}}$, this reduces to

$$
\sigma_\star^2 =
\frac{\sum_i\gamma_i\sigma_i^2}{\sum_i\gamma_i}
$$

when the denominator is nonzero and the result is positive.

If exact conservation of $\boldsymbol{\Gamma}_\star=\sum_i\boldsymbol{\Gamma}_i$
is relaxed, there are more algebraic choices. Given a desired $\sigma_\star$,
one can set

$$
\boldsymbol{\Gamma}_\star =
\frac{5\mathbf{L}_{\mathcal{C}}}{\sigma_\star^2}.
$$

Alternatively, choose $\boldsymbol{\Gamma}_\star$ parallel to
$\mathbf{L}_{\mathcal{C}}$ and solve

$$
\sigma_\star^2 =
\frac{5\lVert\mathbf{L}_{\mathcal{C}}\rVert}
     {\lVert\boldsymbol{\Gamma}_\star\rVert}.
$$

The consequence is direct: total vorticity strength and the leading far-field
Biot-Savart term change, so induced velocity, linear impulse, and wake loading
may change. Such formulas are intentional model reduction or damping choices,
not conservative vortex coarsening. Compromise formulations can instead
minimize
$\lVert\boldsymbol{\Gamma}_\star-\sum_i\boldsymbol{\Gamma}_i\rVert$ subject to
exact $\mathbf{L}_\star=\mathbf{L}_{\mathcal{C}}$, or minimize the
spin-angular-momentum defect subject to exact $\boldsymbol{\Gamma}_\star$.

### Other conserved-property choices

Different coarsening rules choose different invariants:

- $\boldsymbol{\Gamma}$ only: mandatory minimum for conservative vorticity
  coarsening.
- $\boldsymbol{\Gamma}$ plus stored volume: the current exact stored-volume
  invariant.
- $\boldsymbol{\Gamma}$ plus support volume: the current $\sigma^3$
  support-measure rule.
- $\boldsymbol{\Gamma}$ plus rigid-sphere spin angular momentum: the
  $\sigma^2\boldsymbol{\Gamma}$ rule above, when it is feasible.
- $\boldsymbol{\Gamma}$ plus linear impulse: choose replacement positions to
  satisfy
  $\sum_a\mathbf{x}_a\times\boldsymbol{\Gamma}_a
  = \sum_i\mathbf{x}_i\times\boldsymbol{\Gamma}_i$
  when feasible.
- Full first and second vorticity moments: generally require more degrees of
  freedom than one isotropic particle provides.

### Angular impulse alternative

Angular impulse is not the same quantity as rigid-sphere spin angular momentum.
For isotropic vortex particles,

$$
\mathbf{A}_a =
-\frac{1}{3}\mathbf{x}_a\times
  \left(\mathbf{x}_a\times\boldsymbol{\Gamma}_a\right)
+ \frac{2}{3}m_2\sigma_a^2\boldsymbol{\Gamma}_a .
$$

One scalar-width replacement particle generally cannot match an arbitrary
cluster total $\boldsymbol{\Gamma}_{\mathcal{C}}$ and angular impulse
$\mathbf{A}_{\mathcal{C}}$. Two replacement particles are sufficient to match
$\boldsymbol{\Gamma}_{\mathcal{C}}$ and $\mathbf{A}_{\mathcal{C}}$ if arbitrary
vector strengths and two distinct positive radii are allowed. This is weaker
than full second-vorticity-moment matching; two particles are not sufficient in
general for that stronger tensor target.

A constructive two-particle rule is available when the particles share one
center $\mathbf{x}_c$, usually the existing representative center. Define

$$
\mathbf{D} =
\frac{3}{2m_2}
\left(
\mathbf{A}_{\mathcal{C}}
+ \frac{1}{3}\mathbf{x}_c\times
  \left(\mathbf{x}_c\times\boldsymbol{\Gamma}_{\mathcal{C}}\right)
\right).
$$

Choose distinct positive squared radii
$s_1=\sigma_1^2$ and $s_2=\sigma_2^2$, then set

$$
\boldsymbol{\Gamma}_1 =
\frac{\mathbf{D} - s_2\boldsymbol{\Gamma}_{\mathcal{C}}}{s_1-s_2},
\qquad
\boldsymbol{\Gamma}_2 =
\frac{s_1\boldsymbol{\Gamma}_{\mathcal{C}} - \mathbf{D}}{s_1-s_2}.
$$

Then

$$
\boldsymbol{\Gamma}_1+\boldsymbol{\Gamma}_2
= \boldsymbol{\Gamma}_{\mathcal{C}},\qquad
s_1\boldsymbol{\Gamma}_1+s_2\boldsymbol{\Gamma}_2
= \mathbf{D},
$$

so the two colocated particles match both
$\boldsymbol{\Gamma}_{\mathcal{C}}$ and $\mathbf{A}_{\mathcal{C}}$. The formula
can create large or oppositely signed strengths if $s_1$ and $s_2$ are too
close, or if $\mathbf{D}$ is far from parallel to
$\boldsymbol{\Gamma}_{\mathcal{C}}$.

With more replacement particles, the practical choice depends on which moments
matter. Use two colocated particles only for exact
$\boldsymbol{\Gamma}+\mathbf{A}$. Use separated particles if linear impulse is
also targeted, and solve a small constrained least-squares problem over
$\mathbf{x}_a$, $\sigma_a$, and $\boldsymbol{\Gamma}_a$. By a degrees-of-freedom
count, at least five isotropic particles are needed before attempting to match
$\boldsymbol{\Gamma}$, the full first vorticity moment, and the full symmetric
second vorticity moment.

## Implementation

The implementation is allocation-conscious and uses the `MergingWorkspace`
stored in the `ParticleField` so repeated calls reuse integer buffers.

The high-level flow is:

1. Scan the current particle array once. This builds `candidate_indices`,
   excludes static particles when requested, computes the candidate bounding
   box, and accumulates the mean smoothing radius.
2. Choose the cell size. If `r_hash < 0`, the hash radius defaults to
   `r_merge`. With `sigma_relative=true`, the cell size is
   `effective_r_hash * mean_sigma`; otherwise it is `effective_r_hash`.
3. Build a uniform Cartesian cell list over the candidate bounding box. The
   implementation keys each candidate into a cell and uses counts plus prefix
   sums to fill `sorted_indices`. This is a counting-sort-style layout, not a
   radix sort.
4. Initialize union-find state over raw particle indices.
5. For each populated cell, test all pairs within that cell. Pairs that satisfy
   the sigma-ratio gate and distance gate are joined with union-find. The
   union-find uses path compression and union by rank.
6. Group candidates by final root using another counting-sort-style pass. This
   produces CSR-like contiguous ranges of candidates for each root.
7. For each root with more than one member, accumulate the merged quantities
   once, finalize them into the minimum-index representative, and queue every
   other member for removal.
8. Sort removals in descending index order and remove those particles. Removing
   from largest to smallest keeps queued indices valid while the particle array
   is compacted.

The cell-list search currently checks pairs only within the same populated
cell. The cell size therefore controls both search cost and which candidate
pairs can be discovered by the implementation.

## Cost

Let $N$ be the current number of particles, $C_h = N_x N_y N_z$ be the number of
Cartesian hash cells, $P$ be the number of same-cell pair checks, and $R$ be the
number of particles removed.

The integer workspace is $O(N + C_h)$: candidate lists and union-find arrays scale
with particle count, while offsets and counts scale with the cell grid. There
is no persistent floating workspace for merging beyond the per-cluster
accumulators used while finalizing one cluster.

The time cost is

$$O\!\left(N + C_h + P \alpha(N) + R \log R\right)$$

where $\alpha(N)$ is the inverse Ackermann factor from union-find. In typical
use, bounded cell occupancy makes $P = O(N)$, so the routine is near-linear in
the number of particles plus cells. The worst case is quadratic if many
candidates fall in one cell, because all same-cell pairs are checked.

`MergingWorkspace` owns the hot-loop integer buffers:

```text
candidate_indices, sorted_indices, offsets, counts, keys,
parent, rank, root_count, representative, roots, root_offset,
candidates_by_root, to_remove
```

These buffers are resized as needed and then reused across calls to avoid most
hot-loop allocations during repeated merging.

## Conservation Checks

The tests in `test/runtests_merging.jl` exercise the main invariants:

- component-wise conservation of $\boldsymbol{\Gamma}$;
- conservation of stored $V$;
- support-volume conservation through $\sigma_\star^3 = \sum_i \sigma_i^3$;
- strength-weighted centroids for $\mathbf{x}$ and $\mathbf{C}$;
- sigma-weighted representative circulation;
- exclusion of static particles by default;
- sigma-ratio rejection;
- descending removals after multiple independent clusters.

Those tests are the best executable reference for the intended behavior of the
current formulas.

## Symbol Glossary

| Symbol | Meaning / code name |
| --- | --- |
| $\mathcal{C}$ | One connected merge cluster, represented by a root range in `candidates_by_root` |
| $i, j$ | Raw particle indices such as `ia`, `ib`, or `i` |
| $n_\mathcal{C}$ | `n_members` |
| $\mathbf{x}_i$ | Particle position from `X_INDEX`; local scalars `pos_x`, `pos_y`, `pos_z` |
| $\mathbf{x}_\star$ | Merged position written by `set_X` |
| $\mathbf{x}_c$ | Shared center used in the two-particle angular-impulse construction |
| $\mathbf{y}$ | Continuous spatial coordinate used in moment integrals |
| $\mathbf{r}$ | Kernel-centered coordinate, usually $\mathbf{y} - \mathbf{x}_i$ |
| $a$ | Replacement-particle index in alternative formulations |
| $\boldsymbol{\Gamma}_i$ | Particle vortex strength from `GAMMA_INDEX`; local scalars `gamma_i_x`, `gamma_i_y`, `gamma_i_z` |
| $\boldsymbol{\Gamma}_\star$ | Merged vortex strength written by `set_Gamma`; accumulated in `gamma_x`, `gamma_y`, `gamma_z` |
| $\boldsymbol{\Gamma}_{\mathcal{C}}$ | Cluster total vortex strength used in alternative formulations |
| $\gamma_i$ | Signed scalar strength in the co-linear alternative-formulation case |
| $\hat{\boldsymbol{\Gamma}}$ | Common unit or reference direction in the co-linear case |
| $\boldsymbol{\omega}$ | Vorticity field represented by regularized vortex blobs; conceptual quantity |
| $\boldsymbol{\omega}_{\mathcal{C}}$ | Pre-merge cluster vorticity approximation; conceptual quantity |
| $\boldsymbol{\omega}_\star$ | Post-merge single-particle vorticity approximation; conceptual quantity |
| $\boldsymbol{\Omega}_i$ | Rigid-body angular velocity of particle $i$ in the spherical spin model |
| $\zeta_\sigma$ | Normalized radial smoothing kernel of width $\sigma$; conceptual quantity |
| $m_2$ | Kernel-dependent second-moment constant; conceptual quantity |
| $\mathbf{I}_3$ | Identity tensor in the kernel second moment; conceptual quantity |
| $\mathbf{M}_1$ | First vorticity moment $\int \mathbf{y}\otimes\boldsymbol{\omega}\,d\mathbf{y}$; conceptual quantity |
| $\mathbf{M}_{1,\star}$ | First vorticity moment of the merged particle; conceptual quantity |
| $\mathbf{M}_2$ | Second vorticity moment $\int \mathbf{y}\mathbf{y}^T\otimes\boldsymbol{\omega}\,d\mathbf{y}$; conceptual quantity |
| $\mathbf{M}_{2,\star}$ | Second vorticity moment of the merged particle; conceptual quantity |
| $\mathbf{I}$ | Linear impulse $\frac{1}{2}\int\mathbf{y}\times\boldsymbol{\omega}\,d\mathbf{y}$; conceptual quantity |
| $\mathbf{I}_\star$ | Linear impulse of the merged particle; conceptual quantity |
| $\mathbf{A}$ | Angular impulse or angular-momentum proxy $-\frac{1}{3}\int\mathbf{y}\times(\mathbf{y}\times\boldsymbol{\omega})\,d\mathbf{y}$; conceptual quantity |
| $\mathbf{A}_a$ | Angular impulse contribution of replacement particle $a$ in alternative formulations |
| $\mathbf{A}_{\mathcal{C}}$ | Cluster angular impulse used in alternative formulations |
| $J_i$ | Scalar moment of inertia for the spherical particle spin model |
| $\mathbf{L}_i$ | Rigid-sphere spin angular momentum of particle $i$ |
| $\mathbf{L}_{\mathcal{C}}$ | Cluster rigid-sphere spin angular momentum |
| $\mathbf{L}_\star$ | Rigid-sphere spin angular momentum of one replacement particle |
| $\mathbf{D}$ | Construction vector for matching angular impulse with two colocated particles |
| $\lVert\boldsymbol{\Gamma}_i\rVert$ | `gamma_mag` |
| $\sum_i \lVert\boldsymbol{\Gamma}_i\rVert$ | `weight_sum` |
| $V_i$ | Particle volume from `VOL_INDEX` |
| $V_\star$ | Merged volume written by `set_vol`; accumulated as `vol_sum` |
| $\sigma_i$ | Particle smoothing radius from `SIGMA_INDEX`; local variable `sigma` |
| $\sigma_\star$ | Merged smoothing radius written by `set_sigma` |
| $s_1, s_2$ | Distinct positive squared radii in the two-particle angular-impulse construction |
| $\sum_i \sigma_i^3$ | `sigma3_sum` |
| $\sum_i \sigma_i$ | `sigma_sum` |
| $\mathbf{C}_i$ | Particle auxiliary vector from `C_INDEX`; local scalars `c_i_x`, `c_i_y`, `c_i_z` |
| $\mathbf{C}_\star$ | Merged auxiliary vector written by `set_C` |
| $\kappa_i$ | Stored particle circulation from `CIRCULATION_INDEX` |
| $\kappa_\star$ | Merged representative circulation written by `set_circulation` |
| $\sum_i \sigma_i \kappa_i$ | `circulation_weighted_sum` |
| $r_m$ | `r_merge` |
| $\rho_{\max}$ | `max_sigma_ratio` |
| $N$ | `np`, the current particle count |
| $N_x, N_y, N_z$ | `Nx`, `Ny`, `Nz` |
| $C_h$ | `n_cells` |
| $P$ | Number of same-cell pair checks in the nested cell loop |
| $R$ | `length(to_remove)` or returned `n_removed` |
| $\alpha(N)$ | Inverse Ackermann factor from union-find operations |
