## [5.0.0] — 2026-09-22
- **Breaking: particle rows 46 → 39.** Removed the static-particle flag, the
  previous-speed row, two unused scratch rows, and the three PSE rows. Row layout
  is now X 1:3, Γ 4:6, σ 7, vol 8, circulation 9, U 10:12, ω 13:15, J 16:24,
  M 25:33, C 34:36, SFS 37:39. Checkpoints written with 46 rows are rejected.
- **Removed `ParticleStrengthExchange`.** PSE (Degond & Mas-Gallic) is only a
  consistent Laplacian approximation on a regular, periodically remeshed particle
  distribution. FLOWVPM is meshless and never remeshes, so the scheme was never
  usable here; no code in FLOWVPM, FLOWUnsteady, or LiftingLines constructed it.
  `CoreSpreading` (with the RBF reset) is the viscous scheme.
- **Removed static particles.** Every particle evolves; `add_particle` no longer
  accepts `static`, `get_static` always returns `false`. The RBF solver takes an
  explicit `active` mask instead (used by the subset core-reset prototype).
- **Removed `U_prev`.** Nothing read it: the FMM error metadata is filled from
  the current `U` row.

## [4.0.4] — 2026-05-16
- Raise `FMM` default `min_ncrit` to 50 for improved dynamic SFS accuracy.
- This eliminates large coupling errors in dynamic SFS with default FMM settings.
- Override logic unchanged; codebases that explicitly set `min_ncrit` won't be affected.
- No breaking changes, but simulation results may change for dynamic SFS with default options.