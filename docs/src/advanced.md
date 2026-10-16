# Advanced Usage
For more advanced examples please see the examples folder.

## Accessing the ParticleField
`FLOWVPM` provides getter and setter functions for particles. Getter functions take the form `get_property(P)` where `P` is a particle, or `get_property(pfield::ParticleField, i::Int)` where `i` is the particle index. Getter functions return views. Setter functions take the from `set_property(P, val)` and `set_property(pfield, i, val)` where `val` is the new value to be set and is applied via broadcasting. Getter and setter functions are used by replacing "property" with the desired particle property such as `get_X(P)`. The following are the particle properties: 

- `X`: position
- `Gamma`: vortex strength
- `sigma`: radius
- `vol`: volume
- `circulation`: circulation
- `U`: velocity
- `vorticity`: vorticity 
- `J`: velocity gradient
- `M`: auxiliary computational storage
- `C`: SFS model parameters
- `static`: tag indicating whether this particle's states should evolve in time or not (sometimes used to represent solid bodies)
- `PSE`: Storage needed for `ParticleStrengthExchange`
- `SFS`: Storage of SFS values

## Particle Field Settings
Here we provide details concerning all settings for a [`ParticleField`](@ref). Keyword arguments to the constructor (such as governing equations, fast multipole backend settings, and the turbulence model) are reviewed in the following subsections.

### `formulation::Formulation`
The VPM formulation to be used. Options include:
- [`rVPM`](@ref FLOWVPM.rVPM): The reformulated VPM equations (default).
- [`cVPM`](@ref FLOWVPM.cVPM): The classic VPM equations.
- [`formulation_tube_continuity`](@ref FLOWVPM.formulation_tube_continuity): VPM equations derived from tube conservation of mass.
- [`formulation_tube_momentum`](@ref FLOWVPM.formulation_tube_momentum): VPM equations derived from tube conservation of momentum.

### `viscous::ViscousScheme`
The viscous model used when propagating particles.
- [`Inviscid`](@ref FLOWVPM.Inviscid): Uses no viscous modeling for the particles (default).
- [`CoreSpreading`](@ref FLOWVPM.CoreSpreading): Uses a core spreading viscous model in which particles grow in size at each step. Once the cores have grown by `beta`, they are reset to `sgm0` by a Galerkin (L2) projection of the old vorticity field ([`galerkin_shrink!`](@ref FLOWVPM.galerkin_shrink!), parameters [`GalerkinReset`](@ref FLOWVPM.GalerkinReset)): neighbouring particles take strength increments, edge particles may be added and close ones merged, so the field needs `maxparticles` headroom.
- [`ParticleStrengthExchange`](@ref FLOWVPM.ParticleStrengthExchange): Uses particle strength exchange to simulate viscosity.

### `kernel::Kernel`
Regularization scheme used to de-singularize the vorticity and velocity fields.
- [`singular`](@ref FLOWVPM.singular): Singular vortex kernel (e.g. no regularization).
- [`gaussian`](@ref FLOWVPM.gaussian): Gaussian vortex kernel.
- [`gaussianerf`](@ref FLOWVPM.gaussianerf): Gaussian error function vortex kernel (default).
- [`winckelmans`](@ref FLOWVPM.winckelmans): Winckelman vortex kernel

### `UJ`
Function used to solve the ``N``-body problem to obtain the velocity field, vorticity field, and subfilter scale contributions.
- [`UJ_fmm`](@ref FLOWVPM.UJ_fmm): Uses the fast multipole method to solve the ``N``-body problem (default).
- [`UJ_direct`](@ref FLOWVPM.UJ_direct(pfield::FLOWVPM.ParticleField)): Loops through all particle interactions directly.

The integrators make every U/J evaluation through this one function (each stage, the relaxation, an SFS procedure's test-filter pass), called as `UJ(pfield; reset, reset_sfs, sfs)` plus `relaxation=true` for the relaxation pass, so a custom `UJ` accepts further keywords (`; kwargs...`). A code that couples external sources to the particles (e.g. a lifting-line driver's bound and trailing filaments, or a non-uniform freestream) sets `UJ` to its full-field evaluation: FLOWVPM needs no change for a new source. The integrators' former `custom_UJ` keyword must be `nothing`.

### `Uinf::Function`
The freestream fluid velocity. Defaults to `(t) -> SVector{3,Float64}(0,0,0)`.

### `SFS::SubFilterScale`
The subfilter-scale (SFS) model used in the reformulated VPM equations. The SFS model simulated the turbulent energy cascade. For simulations with little turbulence the model may be unnecessary. Custom modifications can be made, setting the coefficient or the alpha used is possible.
- [`noSFS`](@ref FLOWVPM.noSFS): No SFS model is used (default).
- [`SFS_Cs_nobackscatter`](@ref FLOWVPM.SFS_Cs_nobackscatter): Uses a constant coefficient for the SFS model.
- [`SFS_Cd_twolevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_twolevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.999`. Recommended for high fidelity modeling.
- [`SFS_Cd_threelevel_nobackscatter`](@ref FLOWVPM.SFS_Cd_threelevel_nobackscatter): Uses a dynamic coefficient for the SFS model. Sets `alpha=0.667`, which reduces the effect of the SFS significantly.

### `integration`
The time integration scheme used in propogating particles states forward in time.
- [`euler`](@ref FLOWVPM.euler): Uses an euler time stepping scheme.
- [`rungekutta3`](@ref FLOWVPM.rungekutta3): Uses a low storage RK3 time stepping scheme (default).

### `transposed::Bool`
Determines how the stretching term is calculated. Defaults to `true`, which is recommended for stability.

### `relaxation::Relaxation`
Scheme used to ensure the field is divergence free.
- [`norelaxation`](@ref FLOWVPM.norelaxation): Does nothing (e.g. no relaxation).
- [`pedrizzetti`](@ref FLOWVPM.pedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity. (default).
- [`correctedpedrizzetti`](@ref FLOWVPM.correctedpedrizzetti): Relaxation scheme where the vortex strength is aligned with the local vorticity while preserving the magnitude of the particle strength.

Under [`rungekutta3`](@ref FLOWVPM.rungekutta3) the relaxation evaluates U and J once more after the last stage through the field's `UJ` (see above), with `sfs=false, reset_sfs=false, relaxation=true`. A `UJ` with external sources must include them in that call too: Pedrizzetti aligns Γ with the vorticity it reads, without renormalising, so a particle-only vorticity next to an external body rotates the fresh particles off its trailing direction and weakens them each step.

### `fmm::FMM`
Settings for the fast multipole solver. Defaults autotune the FMM so that the absolute and relative tolerance of the velocity calculations are within 1e-3. See the the [`FMM`](@ref FLOWVPM.FMM) API for detailed settings

### `useGPU::Int`
Determines whether or not to use GPU acceleration with `CUDA.jl` to solve the n-body problem. Still experimental and does not work on SFS calculation, only works on NVIDIA GPUs. To allow GPU usage set to values greater than 0. (default is 0).

## Core Spreading and the Galerkin Reset

[`CoreSpreading`](@ref FLOWVPM.CoreSpreading) diffuses vorticity by growing every
core, ``\sigma^2 \mathrel{+}= 2\nu\,\Delta t``. Once the cores have grown by `beta`
they are shrunk back to `sgm0` by a Galerkin reset, which can also be called on its
own to shrink any set of large cores in an ordinary (inviscid or LES) run.

**How the reset works.** It finds new strengths whose combined field best matches the
old field everywhere (an ``L_2`` projection onto Gaussians, not a point-by-point
interpolation, which is numerically singular for overlapping cores):

- the reset particles go to `sgm0`;
- their neighbours keep their own cores and take strength increments (the *halo*),
  so the old cores' tails are not lost;
- new `sgm0` points are added only where the old tails reach past the last particle
  (*edge points*), and reset particles crowded closer than `thin` are merged first;
- the resulting symmetric positive-definite system is solved by a block-Jacobi
  preconditioned conjugate gradient that stops on the field error. It runs matrix
  free on the GPU; on the CPU it stores the overlap pairs up to a memory cap
  (`gram_mem`, 1 GiB) and evaluates the rest on the fly.

The particle count and order change at every reset (edge points in, thinned particles
out): give the field `maxparticles` headroom, and do not hold particle indices across
a reset.

### Using it with core spreading

```julia
import FLOWVPM as vpm

# defaults: reset when the cores have grown by beta = 1.25
viscous = vpm.CoreSpreading(nu, sgm0)

# custom reset parameters (see the table below)
reset   = vpm.GalerkinReset(; tol = 1e-3, gram_mem = 4 * 2^30)
viscous = vpm.CoreSpreading(nu, sgm0; beta = 1.5, galerkin = reset, verbose = true)

pfield = vpm.ParticleField(maxparticles; viscous, kernel = vpm.gaussianerf)
```

`CoreSpreading` options: `beta` (default `1.25`, growth ``\sigma/\sigma_0`` that
triggers a reset), `growth_beta` (default `0`, off: also reset when the largest core of
the field reaches `growth_beta * sgm0`), `iterror` (default `false`: an unconverged
solve warns; `true` throws), `verbose` (one summary line per reset), and `galerkin`
(the [`GalerkinReset`](@ref FLOWVPM.GalerkinReset) parameters). Gaussian
(`gaussianerf`) kernel only.

### Shrinking large cores in any run

[`galerkin_shrink!`](@ref FLOWVPM.galerkin_shrink!) works on any field, with no
viscous scheme, on the CPU or the GPU:

```julia
reset = vpm.GalerkinReset()          # keep one: its workspace is reused across calls
sig   = Array(view(pfield.particles, vpm.SIGMA_INDEX, 1:vpm.get_np(pfield)))
idx   = findall(>(1.5 * sgm0), sig)  # cores above 1.5 sgm0
info  = vpm.galerkin_shrink!(pfield, idx, sgm0; galerkin = reset)
# info: counts (K reset, H halo, E edge points, D thinned), iterations, field error,
#       memory and timing of each phase
```

In FLOWUnsteadyCore, the `simulate!` options `core_reset` and `core_reset_every`
call it every step or every N steps.

### Reset parameters (`GalerkinReset`)

Lengths are in units of `sgm0`.

| keyword | default | meaning | when to change |
|---|---|---|---|
| `tol` | `5e-3` | stop once 10 iterations lower the squared relative field error by less than `tol^2` | tighter (e.g. `1e-10`) for derivatives |
| `itmax` | `300` | iteration cap | raise with a tight `tol` |
| `halo_tail`, `halo_min` | `2.0`, `1.0` | neighbours within `max(halo_min, halo_tail * s_k)` of reset particle k take increments (``s_k^2 = \sigma_k^2 - \sigma_0^2``) | rarely |
| `halo_tol` | `3e-3` | keep only neighbours where the old tail is at least this fraction of its peak (`0`: all) | `0` for best accuracy at higher cost |
| `edge_offsets`, `edge_spacing`, `edge_tol` | `[1.0, 2.0]`, `0.7`, `1e-2` | edge points at ``x_k \pm a\,e_j`` where no particle lies within `edge_spacing` and the old field is above `edge_tol` of its peak (empty offsets: none) | fewer or none to limit particle growth |
| `prune_tol` | `1e-2` | drop edge points weaker than this fraction of the strongest reset particle | raise to add fewer particles |
| `thin` | `0.5` | merge reset particles closer than this before the solve (`0`: off) | rarely |
| `damping` | `1e-2` | ``\mu`` in ``(M + \mu\,\mathrm{diag}\,M)\,c = f`` | rarely |
| `gram_cut` | `5.0` | overlap cutoff in combined core widths (4 is not positive definite on real wakes) | keep 4.5-5 |
| `block_cell`, `block_cap` | `3.0`, `512` | block-Jacobi blocks (`block_cap = 0`: plain CG) | rarely |
| `gram_mem` | `2^30` (1 GiB) | CPU memory for stored overlap pairs; the rest is evaluated on the fly | raise for speed if memory allows |

### Accuracy and cost

On the NREL 5MW wake (checkpoint step 2880) and Lamb-Oseen diffusion tests:

- off-particle vorticity error 0.25-0.5 % on the wake (the previous collocation reset:
  2.8-4.6 %), 0.05-0.13 % on Lamb-Oseen; circulation changes 0.5-0.8 % per reset;
  strengths do not grow;
- a single, unresolved row of particles loses more (about 7 % of its circulation): its
  lateral tails have no particles to go to;
- cost per reset: 0.2-0.35 s for a handful of cores in isolation, 0.6-5 s for one
  revolution's worth, about 3 s for ten thousand cores (Apple M1 Pro, CPU with 4 threads
  or Metal). Every reset also solves for the neighbours of the reset cores, so many
  small resets cost much more than one large one: in a 6-revolution HVAB hover run,
  resetting once per revolution took 0.8 % of the run time with loads within 0.04 % of
  a run without resets, while resetting every step took about 20 % (run 23 % slower).
  **Reset once per revolution** (FLOWUnsteadyCore: `core_reset_every` = steps per
  revolution).

### Derivatives

Reverse-mode AD (Enzyme) through a reset is supported on CPU fields: the linear solve
carries an implicit rule (one adjoint solve with the same preconditioned CG), so the
iterations are never differentiated. The derivative covers the new strengths as
functions of the positions, cores and strengths involved; which particles are
selected, thinned or added is discrete and carries no derivative. Use a tight `tol`
(e.g. `1e-10`) and a larger `itmax` for differentiable runs.

## Code Integration
For examples of integrating FLOWVPM into other codes please see [FLOWUnsteady](https://github.com/byuflowlab/FLOWUnsteady) and
[VortexLattice](https://github.com/byuflowlab/VortexLattice)