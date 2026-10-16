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

CI runs on Julia 1.9 and 1.10 across Linux/macOS/Windows (`.github/workflows/CI.yml`;
bumped from the 1.6 floor on the `gpu-full` branch since Julia package
extensions require >=1.9). GPU code lives in `ext/FLOWVPMCUDAExt.jl`, a CUDA
package extension (loaded only when `CUDA` is also `using`d) — CPU-only CI
never loads it and is unaffected. See "GPU path" below.

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
Tkernel, TUJ, Tintegration, TRelaxation, AT<:AbstractMatrix{R}}` is fully
parameterized on its solver components so each piece specializes at compile
time (the last parameter, `AT`, is the array type backing `pfield.particles`
— `Matrix` for CPU, `CUDA.CuArray` for GPU; see "GPU path" below):
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

### GPU path (validated on real hardware for the no-FMM direct sum; FMM+GPU not ready)
`ParticleField.particles`/`.scratch` are parameterized over array type
(`AT<:AbstractMatrix{R}`, see the `arraytype` constructor keyword — pass
`arraytype=CUDA.CuArray` to get a GPU-backed field). Hot-path physics
(time integration, relaxation, viscous, SFS in `FLOWVPM_timeintegration.jl`/
`FLOWVPM_relaxation.jl`/`FLOWVPM_viscous.jl`/`FLOWVPM_subfilterscale*.jl`)
is forked per function: `pfield.particles isa Array` → the original CPU
loop, unchanged; `else` → a broadcast implementation, added specifically
because pure broadcasting was a real 4-10x CPU regression in isolation —
don't try to unify these into one implementation without benchmarking
first. The O(N²) direct-sum kernels (`gpu_direct!`/`gpu_zeta_direct!`/
`gpu_estr_direct!`, used by `UJ_direct`/`zeta_direct`/`Estr_direct!`) live in
`ext/FLOWVPMCUDAExt.jl`, a CUDA package extension (loaded only when `CUDA`
is also `using`d — `CUDA` is a `[weakdeps]` entry, not a hard dependency, so
CPU-only downstream users/installs never pull in the ~40-package CUDA
stack). `pfield.useGPU` is a separate, largely vestigial `Int` field — the
real CPU/GPU switch everywhere in this codebase is `pfield.particles isa
Array`, not `useGPU`.

**Validated on an H200 (2026-07-22):** `add_particle`, `UJ_direct`,
`zeta_direct`, `Estr_direct!`, and the Phase 1 broadcast physics are all
correct (checked against the CPU reference, Float32/Float64, with static
particles) and fast (up to ~800x for `UJ_direct` at 1M particles vs. an
8-thread CPU baseline). A permanent regression test lives in
`test/runtests_gpu.jl`, gated on `CUDA.functional()` (no-op on CPU-only CI).
Tiling `zeta_direct`/`Estr_direct!`'s kernels (like `UJ_direct`'s existing
`gpu_atomic_square!`) was tried and **reverted** — it helps at small/mid N
but regresses 1.8-2.2x at ~1M particles (the common real-world size here),
almost certainly an occupancy tradeoff (bigger shared-memory footprint per
block vs. fewer concurrent blocks) that only pays off for compute-heavy
kernels like `UJ_direct`. Fusing the ~10-15 broadcast kernel launches in the
Phase 1 physics was also measured and found to be within noise (<1%) at
100k-1M particles — the O(N²) direct sum totally dominates at that scale, so
this isn't worth pursuing further without an O(N) algorithm.

**FMM + GPU (Phase 4): implemented and H200-validated 2026-08-06 (task 034,
Done + approved) via FastMultipole's device-resident radix lifecycle.**
H200 evidence: job 13061046 (all device test groups pass; cube 8.83e-4 /
wake 2.13e-4 device velocity RMS; RK3 dynamic parity; 023 counter contract
flat) and job 13061128 (sha256-checksummed 033 sampled-direct references:
Float64 u_rel_rms cube 1.25e-4 at n=1e4 / 5.65e-4 at n=1e5, wake 2.27e-4 /
4.79e-4 — all inside the 1e-3 gate).
- `src/FLOWVPM_fmm_radix.jl` couples `ParticleField` to FastMultipole's
  `RadixFMMCache` (branch `matrix-ops`, task 032 interface): a
  `CuArray`-backed field is a `DeviceResident` system (bulk device
  pack/unpack hooks live in `ext/FLOWVPMCUDAExt.jl`; zero per-step
  host/device body transfer), a `Matrix`-backed field can use the same
  machinery through the transfer-based host path (exercised by the CPU
  tests). One cache is built lazily at first use, sized to
  `pfield.maxparticles`, reused across all RK3 substeps/steps; when the
  field outgrows its derived domain box the coupling calls
  `FastMultipole.recenter!` once and retries (user-fixed bounds error
  instead). Only `gaussianerf` is supported; FMM autotuning flags must be
  off; `rbf`/`sfs` on this path fail loudly. Overrides:
  `FLOWVPM.radix_fmm_settings!` (internal, not exported).
- The old `nearfield_device` hazard (the generic
  `FastMultipole.nearfield_device!` fallback silently DROPPED the nearfield)
  is unreachable: `UJ_fmm` routes GPU-backed fields to `UJ_fmm_gpu!` and the
  legacy octree call is CPU-only with `nearfield_device=false` hardwired.
- The coupling is a no-op (loud error stub) when the installed FastMultipole
  lacks the radix interface (`FLOWVPM._FMM_HAS_RADIX == false`) — registry
  releases still load. NOTE the branch's `UJ_fmm` legacy call itself already
  requires an unreleased FastMultipole (`shrink`/`recenter` kwargs; registry
  max is 2.0.4), so `gpu-full` is effectively pinned to dev FastMultipole.
- FastMultipole `matrix-ops` target buffers are SWITCH-RELATIVE; FLOWVPM's
  `fmm.direct!`/`buffer_to_target_system!` hooks pick fixed vs switch-aware
  accessors at load time (`_fmm_get_gradient` shims in `FLOWVPM_fmm.jl`) —
  using fixed rows against matrix-ops was silently corrupting U/J (row
  shift + out-of-buffer `@inbounds` write).
- Tests: `test/runtests_gpu_fmm.jl` (CPU Part A always; device Part B in
  `runtests_gpu_fmm_device.jl`, gated on functional CUDA, hard-required
  under `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1`). H200 job drafts:
  `scripts/cuda_034_run.sh` / `scripts/cuda_034_submit.sh` (stage 4:
  `scripts/cuda_034_refcheck.jl`, the checksummed 033-reference gate).

### Differentiability
`FLOWVPM_rrules.jl` defines custom reverse-mode rules (ChainRules-style) but
is currently excluded from the `include` list in `FLOWVPM.jl` — it's dormant
code, not part of the active build.

### Tests
`test/runtests.jl` is the entry point and includes `runtests_singlevortexring.jl`
and `runtests_leapfrog.jl`, each validating solver output against known
analytic/reference behavior (isolated vortex ring, leapfrogging vortex rings) —
CPU only, always run. It also conditionally includes `test/runtests_gpu.jl`
(the direct-sum GPU regression test described above) if `import CUDA` and
`CUDA.functional()` both succeed; this requires `CUDA` to be present in
whatever environment `] test`/`include("test/runtests.jl")` runs in (it
isn't a hard test dependency in `test/Project.toml`, to keep CPU-only
`] test` from ever needing to fetch it) and real GPU hardware, so it's a
no-op in ordinary CI.
