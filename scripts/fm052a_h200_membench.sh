#!/usr/bin/env bash
# 052a Phase C companion: same membench as fm052a_gh200_membench.sh but on an
# x86 H200 (PCIe) node, to replace the theoretical PCIe column in the Phase C
# decision-gate table with measurements. Reuses the arch-agnostic
# fm052a_gh200_membench.jl unchanged. Submit as, e.g.:
#   sbatch --job-name=fp052a-h200-membench --partition=m13h \
#     --gres=gpu:h200:1 --cpus-per-task=16 --mem=192G \
#     --time=01:00:00 \
#     --output=$HOME/FLOWPanel-052-h200/data/fm052_multiarch/h200/slurm/fp052a-h200-membench-%j.out \
#     $HOME/FLOWVPM-052-h200/scripts/fm052a_h200_membench.sh
source /etc/profile
set -euo pipefail
: "${SLURM_JOB_ID:?must run under Slurm}"

module load cuda julia/1.11.7-6bmogfl
ENVDIR="${FP052_ENV:-$HOME/fm052env-h100}"
OUTDIR="${FP052A_MEMBENCH_DIR:-$HOME/FLOWPanel-052-h200/data/fm052_multiarch/h200/membench/job-$SLURM_JOB_ID}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUTDIR"

echo "== 052a h200 membench: job=$SLURM_JOB_ID node=$(hostname) arch=$(uname -m)"
nvidia-smi -L
nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader

export FM052A_MEMBENCH_CSV="$OUTDIR/fm052a_h200_membench_${SLURM_JOB_ID}.csv"
julia --project="$ENVDIR" \
  "$HOME/FLOWVPM-052-gh200/scripts/fm052a_gh200_membench.jl" \
  2>&1 | tee "$OUTDIR/fm052a_h200_membench_${SLURM_JOB_ID}.log"

sha=$(sha256sum "$FM052A_MEMBENCH_CSV" | awk '{print $1}')
{
  echo "job_id = \"$SLURM_JOB_ID\""
  echo "node = \"$(hostname)\""
  echo "csv = \"$FM052A_MEMBENCH_CSV\""
  echo "csv_sha256 = \"$sha\""
  echo "created_utc = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
} > "$OUTDIR/fm052a_h200_membench_${SLURM_JOB_ID}.toml"
echo "== membench complete: $OUTDIR"
