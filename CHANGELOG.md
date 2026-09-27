## [4.0.4] — 2026-05-16
- Raise `FMM` default `min_ncrit` to 50 for improved dynamic SFS accuracy.
- This eliminates large coupling errors in dynamic SFS with default FMM settings.
- Override logic unchanged; codebases that explicitly set `min_ncrit` won't be affected.
- No breaking changes, but simulation results may change for dynamic SFS with default options.