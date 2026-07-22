# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FLOWVPM.jl implements the reformulated Vortex Particle Method (rVPM), a meshless
Lagrangian CFD scheme for solving the LES-filtered incompressible Navier-Stokes
equations in vorticity form. Particles carry vorticity/circulation and are
advected/stretched by a velocity field computed via an N-body solver
(direct or fast multipole). It is a standalone package but is also consumed by
FLOWUnsteady and VortexLattice, so public API changes (exported names, keyword
argument defaults) can break downstream repos.

## Commands

```julia
# From repo root, in the Julia REPL:
] activate .
] instantiate
] test                      # runs test/runtests.jl (CPU only by default)
```

```bash
# From shell:
julia --project=. -e 'using Pkg; Pkg.test()'
```

Single test file (bypassing the Test.jl runner wiring in `runtests.jl`):
```bash
julia --project=. test/runtests_leapfrog.jl
julia --project=. test/runtests_singlevortexring.jl
```

Docs build (mirrors `.github/workflows/CI.yml` `docs` job):
```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl   # if present; otherwise use julia-actions/julia-docdeploy locally
```

CI runs on Julia 1.6 and 1.10 across Linux/macOS/Windows (`.github/workflows/CI.yml`).
GPU code paths (`FLOWVPM_gpu.jl`, `FLOWVPM_gpu_erf.jl`) depend on CUDA and are
currently commented out of module loading and of `test/runtests.jl` — GPU tests
do not run in CI.

## Architecture

### Module loading order matters
`src/FLOWVPM.jl` is the entry point. It defines constants/exports and then
`include`s the other `src/FLOWVPM_*.jl` files in an explicit order (kernel →
viscous → formulation → relaxation → subfilterscale → particlefield → fmm →
gpu_erf → UJ → subfilterscale_models → timeintegration → monitors → utils).
Later files depend on types/functions defined earlier, so if you add a new
file it must be inserted at the right point in that list, not just appended.

### Particle storage
There is no `Particle` struct with named fields. Every particle field
(`ParticleField`) stores particles as columns of a single dense
`Matrix{R}` (`pfield.particles`, `nfields = 46` rows per particle). Properties
(position, strength, sigma, U, J, vorticity, SFS terms, etc.) are accessed via
row-index constants (e.g. `X_INDEX`, `GAMMA_INDEX`, `SIGMA_INDEX`, `U_INDEX`,
`J_INDEX`) defined in `src/FLOWVPM_particlefield.jl`, with `get_*`/`set_*`
accessor functions built on top. This layout exists for FMM/GPU interop —
prefer extending the accessor functions over introducing new struct fields.

### Solver composition via parametric types
`ParticleField{R, F<:Formulation, V<:ViscousScheme, TUinf, S<:SubFilterScale,
Tkernel, TUJ, Tintegration, TRelaxation, TGPU}` is fully parameterized on its
solver components so each piece specializes at compile time:
- **Formulation** (`FLOWVPM_formulation.jl`): `ClassicVPM` vs `ReformulatedVPM`
  (parameterized by conservation coefficients `f`, `g`). Aliased in
  `FLOWVPM.jl` as `cVPM`, `rVPM`, `formulation_tube_continuity`, etc.
- **Kernel** (`FLOWVPM_kernel.jl`): regularization function + its derivatives
  (`singular`, `gaussian`, `gaussianerf` (default), `winckelmans`).
- **ViscousScheme** (`FLOWVPM_viscous.jl`): `Inviscid`, `CoreSpreading`,
  `ParticleStrengthExchange`. `_kernel_compatibility` in `FLOWVPM.jl` enforces
  which kernels are valid with which viscous scheme — check/update this when
  adding either.
- **SubFilterScale (SFS)** (`FLOWVPM_subfilterscale.jl` +
  `FLOWVPM_subfilterscale_models.jl`): `NoSFS`, `ConstantSFS`, `DynamicSFS`
  (with pseudo-3-level dynamic procedures). This is the LES turbulence closure.
- **Relaxation** (`FLOWVPM_relaxation.jl`): schemes that re-align vorticity to
  be divergence-free (`pedrizzetti`, `correctedpedrizzetti`, `norelaxation`).
- **UJ** (`FLOWVPM_UJ.jl`): computes the velocity/Jacobian N-body interaction —
  either `UJ_direct` (O(N²), also used as the ground-truth reference in tests)
  or `UJ_fmm` (fast multipole via `FastMultipole.jl`, aliased as `fmm` inside
  this module — not to be confused with the `FMM` settings struct).
- **Integration**: RK3 low-storage (`FLOWVPM_timeintegration.jl`), default.

Solver aliases assembled from these pieces (`rVPM`, `SFS_Cd_twolevel_nobackscatter`,
etc.) are defined near the bottom of `FLOWVPM.jl` — that's the place to look
for/add named presets rather than constructing components inline.

### FMM coupling
`FastMultipole.jl` (imported as `fmm`, aliased `FastMultipole`) does the heavy
N-body lifting. `FMM` (in `FLOWVPM_particlefield.jl`) holds FMM tuning knobs
(`p`, `ncrit`, `theta`, tolerances, autotuning flags, `min_ncrit`). `min_ncrit`
was recently raised from its old default to 50 (`FLOWVPM.jl` history,
`CHANGELOG.md`) specifically because small auto-tuned `ncrit` values were
producing large coupling errors when combined with the dynamic SFS model —
be cautious about lowering it without re-checking SFS accuracy.
`FLOWVPM_fmm.jl` implements the glue (`direct!`, source/target definitions)
that lets `ParticleField` act as a body type FastMultipole understands.

### GPU path (experimental, currently disabled)
`FLOWVPM_gpu.jl`/`FLOWVPM_gpu_erf.jl` implement a CUDA kernel path for direct
P2P evaluation (custom launch-config tuning via `get_launch_config`/
`check_launch`, shared-memory checks). These are not included in the module's
build (commented out in the `include` loop in `FLOWVPM.jl`) or exercised by
CI. `ParticleField` still carries a `useGPU` type parameter and `basic.jl`
demonstrates constructing GPU-backed fields directly against
`FLOWVPM.fmm.direct!` — treat this path as WIP/manual-testing only, not a
CI-verified feature. SFS calculations are explicitly not GPU-accelerated.

### Differentiability
`FLOWVPM_rrules.jl` defines custom reverse-mode rules (ChainRules-style) but
is currently excluded from the `include` list in `FLOWVPM.jl` — it's dormant
code, not part of the active build.

### Tests
`test/runtests.jl` is the entry point and includes `runtests_singlevortexring.jl`
and `runtests_leapfrog.jl`, each validating solver output against known
analytic/reference behavior (isolated vortex ring, leapfrogging vortex rings).
GPU tests are scaffolded (`test_using_GPU` flag, gated on `CUDA.functional()`)
but currently commented out.
