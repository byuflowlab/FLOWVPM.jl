#!/bin/bash
#SBATCH --job-name=ka_uj_ab
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Acceptance gate for the FastMultipole KA migration: whole-UJ_fmm A/B, both
# arms over one cache-built resident state (scripts/ka_uj_fmm_ab.jl).
#
# --cpus-per-task=8, not 1: at 1 CPU the earlier tree-build sweeps were
# host-starved (Julia GC + CUDA driver threads on one core), which showed up
# as 2-15x per-trial stalls and inflated the KA arm specifically, since KA
# issues more and smaller launches. 8 removed it.
#
# Needs FastMultipole@ka-migration dev'd into the env plus KernelAbstractions;
# registry FastMultipole has neither the radix interface nor the KA extension.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM_KAAB_DIR:-$HOME/FLOWVPM-kabench}"
ENVDIR="${VPM_KAAB_ENV:-$HOME/fm_kabench_env}"

cd "$WORKDIR"

CSV="$WORKDIR/ka_uj_fmm_ab_${SLURM_JOB_ID}.csv"
LOG="$WORKDIR/ka_uj_fmm_ab_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/ka_uj_fmm_ab_${SLURM_JOB_ID}.provenance"

echo "=== whole-UJ KA-vs-native A/B: scripts/ka_uj_fmm_ab.jl ==="
KA_UJ_AB_CSV="$CSV" \
KA_UJ_AB_NS="${KA_UJ_AB_NS:-100000,1000000}" \
KA_UJ_AB_REPS="${KA_UJ_AB_REPS:-20}" \
    julia --project="$ENVDIR" scripts/ka_uj_fmm_ab.jl 2>&1 | tee "$LOG"

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "flowvpm_sha=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
  FMDIR=$(julia --project="$ENVDIR" -e 'import FastMultipole; println(dirname(dirname(pathof(FastMultipole))))' 2>/dev/null || true)
  echo "fastmultipole_dir=${FMDIR:-unknown}"
  echo "fastmultipole_sha=$(git -C "${FMDIR:-/nonexistent}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "fastmultipole_branch=$(git -C "${FMDIR:-/nonexistent}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
  echo "csv=$CSV"
  echo "csv_sha256=$(shasum -a 256 "$CSV" | awk '{print $1}')"
} > "$PROV"
cat "$PROV"

echo "ka_uj_ab job complete"
