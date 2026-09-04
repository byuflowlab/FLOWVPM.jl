#!/bin/bash
#SBATCH --job-name=vpm_kabench
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# KA-vs-CUDA H200 gate for the metal-testing/KernelAbstractions side-track.
# Pattern: cuda_034_run.sh / cuda_048_run.sh, trimmed down -- this is a
# standalone correctness+speed comparison (scripts/ka_cuda_bench.jl), not a
# full coupling-test suite, and (unlike 034/048) needs only registry
# FastMultipole (metal-testing is rebased onto flowpanel, compat
# FastMultipole = "2.2.0" in Project.toml -- no dev branch required).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM_KABENCH_DIR:-$HOME/FLOWVPM-kabench}"
ENVDIR="${VPM_KABENCH_ENV:-$HOME/fm_kabench_env}"

cd "$WORKDIR"

CSV="$WORKDIR/ka_cuda_bench_${SLURM_JOB_ID}.csv"
LOG="$WORKDIR/ka_cuda_bench_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/ka_cuda_bench_${SLURM_JOB_ID}.provenance"

echo "=== KA vs CUDA gate: scripts/ka_cuda_bench.jl ==="
KA_CUDA_BENCH_CSV="$CSV" julia --project="$ENVDIR" scripts/ka_cuda_bench.jl 2>&1 | tee "$LOG"

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "flowvpm_sha=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
  echo "flowvpm_tree_sha256=$({ find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/scripts" -type f -print0; printf '%s\0' "$WORKDIR/Project.toml"; } | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
  echo "csv=$CSV"
  echo "csv_sha256=$(shasum -a 256 "$CSV" | awk '{print $1}')"
} > "$PROV"
cat "$PROV"

echo "vpm_kabench job complete"
