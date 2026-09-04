#!/bin/bash
#SBATCH --job-name=vpm048
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 048 H200 validation job (+ folded-in 047 device checks). Pattern:
# cuda_034_run.sh (julia pinned to 1.11.7: 1.12.6 segfaults JIT-compiling the
# device step, job 13058191).
#
#   1. FastMultipole device tests on the unified branch: CUDA interface +
#      lifecycle + device-system-interface (incl. the 048 SFS host testset) +
#      radix settings surface;
#   2. FLOWVPM direct-sum GPU regression (runtests_gpu.jl — now asserting the
#      migrated VORTICITY_INDEX zeta storage);
#   3. radix FMM coupling tests (runtests_gpu_fmm.jl Part A + Part B,
#      including the new 048 SFS device testsets: parity vs GPU Estr_direct!,
#      counter contract, allocation delta, graph-replay parity);
#   4. synchronized, warmed same-state P4/P8 x F32/F64 UJ-vs-UJ+SFS A/B;
#   4b. accuracy/efficiency tuning sweep (P x rho_t x near-shell q on the
#       cube; Pareto-frontier configs gated at the strict 5e-4 F64 delivered
#       gate ON THE p018 PRODUCTION FIELD, cube-vs-p018 discrepancy reported);
#   5. 047 device wiring check: construction-locked setting late-flip errors
#      loudly (scripts/fm048_device_settings_check.jl).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM048_DIR:-$HOME/FLOWVPM-046}"
FMDIR="${VPM048_FMDIR:-$HOME/FastMultipole-046}"
ENVDIR="${VPM048_ENV:-$HOME/fm048env}"
P018_BIN="${FM048_P018_BIN:-$WORKDIR/data/fm048/p018_710_particles.bin}"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

echo "=== stage 1: FastMultipole device tests (unified branch) ==="
cd "$FMDIR"
julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
julia --project="$ENVDIR" test/device_system_interface_test.jl
julia --project="$ENVDIR" test/radix_settings_test.jl

cd "$WORKDIR"
echo "=== stage 2: FLOWVPM direct-sum GPU regression (runtests_gpu.jl) ==="
julia --project="$ENVDIR" -e 'using Test; import FLOWVPM, CUDA
CUDA.functional() || error("CUDA not functional on this node")
include("test/runtests_gpu.jl")'

echo "=== stage 3: radix FMM coupling tests (host + device, incl. 048 SFS) ==="
TEST_LOG="$WORKDIR/fm048_device_tests_${SLURM_JOB_ID}.log"
julia --project="$ENVDIR" test/runtests_gpu_fmm.jl 2>&1 | tee "$TEST_LOG"
TEST_LOG_SHA256="$(shasum -a 256 "$TEST_LOG" | awk '{print $1}')"
echo "stage3_raw_log=$TEST_LOG"
echo "stage3_raw_log_sha256=$TEST_LOG_SHA256"

echo "=== stage 4: corrected 048 synchronized A/B matrix ==="
AB_CSV="$WORKDIR/fm048_ab_${SLURM_JOB_ID}.csv"
AB_LOG="$WORKDIR/fm048_ab_${SLURM_JOB_ID}.log"
AB_PROV="$WORKDIR/fm048_ab_${SLURM_JOB_ID}.provenance"
AB_CMD="FM048_P018_BIN=$P018_BIN FM048_AB_CSV=$AB_CSV julia --project=$ENVDIR scripts/fm048_ab_benchmark.jl"
FM048_P018_BIN="$P018_BIN" FM048_AB_CSV="$AB_CSV" julia --project="$ENVDIR" scripts/fm048_ab_benchmark.jl 2>&1 | tee "$AB_LOG"
{
  echo "exact_command=$AB_CMD"
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "flowvpm_sha=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
  echo "fastmultipole_sha=$(git -C "$FMDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
  echo "flowvpm_tree_sha256=$({ find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/test" "$WORKDIR/scripts" -type f -print0; printf '%s\0' "$WORKDIR/Project.toml"; } | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "fastmultipole_tree_sha256=$({ find "$FMDIR/src" "$FMDIR/test" -type f -print0; printf '%s\0' "$FMDIR/Project.toml"; } | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "stage3_raw_log=$TEST_LOG"
  echo "stage3_raw_log_sha256=$TEST_LOG_SHA256"
  echo "flowvpm_project_sha256=$(shasum -a 256 "$WORKDIR/Project.toml" | awk '{print $1}')"
  echo "fastmultipole_project_sha256=$(shasum -a 256 "$FMDIR/Project.toml" | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "p018_snapshot=$P018_BIN"
  echo "p018_snapshot_sha256=$(shasum -a 256 "$P018_BIN" | awk '{print $1}')"
  echo "pkg_status_begin"
  julia --project="$ENVDIR" -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)'
  echo "pkg_status_end"
  echo "raw_log=$AB_LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$AB_LOG" | awk '{print $1}')"
  echo "csv=$AB_CSV"
  echo "csv_sha256=$(shasum -a 256 "$AB_CSV" | awk '{print $1}')"
} > "$AB_PROV"
cat "$AB_PROV"

echo "=== stage 4b: 048 accuracy/efficiency tuning sweep (user-directed 2026-08-22) ==="
SWEEP_CSV="$WORKDIR/fm048_sweep_${SLURM_JOB_ID}.csv"
SWEEP_LOG="$WORKDIR/fm048_sweep_${SLURM_JOB_ID}.log"
SWEEP_PROV="$WORKDIR/fm048_sweep_${SLURM_JOB_ID}.provenance"
SWEEP_CMD="FM048_P018_BIN=$P018_BIN FM048_SWEEP_CSV=$SWEEP_CSV julia --project=$ENVDIR scripts/fm048_tuning_sweep.jl"
FM048_P018_BIN="$P018_BIN" FM048_SWEEP_CSV="$SWEEP_CSV" julia --project="$ENVDIR" scripts/fm048_tuning_sweep.jl 2>&1 | tee "$SWEEP_LOG"
{
  echo "exact_command=$SWEEP_CMD"
  echo "raw_log=$SWEEP_LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$SWEEP_LOG" | awk '{print $1}')"
  echo "csv=$SWEEP_CSV"
  echo "csv_sha256=$(shasum -a 256 "$SWEEP_CSV" | awk '{print $1}')"
  echo "p018_snapshot=$P018_BIN"
  echo "p018_snapshot_sha256=$(shasum -a 256 "$P018_BIN" | awk '{print $1}')"
  echo "ab_provenance=$AB_PROV"
} > "$SWEEP_PROV"
cat "$SWEEP_PROV"

echo "=== stage 5: 047 device construction-lock check ==="
julia --project="$ENVDIR" scripts/fm048_device_settings_check.jl

echo "vpm048 job complete"
