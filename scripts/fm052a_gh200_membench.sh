#!/usr/bin/env bash
# 052a Phase C combined memory benchmark on GH200 (own job, not a harness
# stage). Submit as, e.g.:
#   sbatch --job-name=fp052a-gh200-membench --partition=mgh \
#     --gres=gpu:gh200:1 --constraint=arm --cpus-per-task=16 --mem=192G \
#     --time=01:00:00 \
#     --output=$HOME/FLOWPanel-052-gh200/data/fm052_multiarch/gh200/slurm/fp052a-gh200-membench-%j.out \
#     $HOME/FLOWVPM-052-gh200/scripts/fm052a_gh200_membench.sh
source /etc/profile
set -euo pipefail
: "${SLURM_JOB_ID:?must run under Slurm}"

JULIA_BIN="${FP052_JULIA_BIN:-/home/rander39/julia/julia-1.11.7/bin/julia}"
ENVDIR="${FP052_ENV:-$HOME/fm052env-gh200}"
OUTDIR="${FP052A_MEMBENCH_DIR:-$HOME/FLOWPanel-052-gh200/data/fm052_multiarch/gh200/membench/job-$SLURM_JOB_ID}"
export JULIA_DEPOT_PATH="${FP052_DEPOT:-$HOME/fm052depot-gh200}"
export JULIA_PKG_OFFLINE=1
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUTDIR"

echo "== 052a gh200 membench: job=$SLURM_JOB_ID node=$(hostname) arch=$(uname -m)"
nvidia-smi -L
nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader

export FM052A_MEMBENCH_CSV="$OUTDIR/fm052a_gh200_membench_${SLURM_JOB_ID}.csv"
"$JULIA_BIN" --project="$ENVDIR" \
  "$HOME/FLOWVPM-052-gh200/scripts/fm052a_gh200_membench.jl" \
  2>&1 | tee "$OUTDIR/fm052a_gh200_membench_${SLURM_JOB_ID}.log"

sha=$(sha256sum "$FM052A_MEMBENCH_CSV" | awk '{print $1}')
{
  echo "job_id = \"$SLURM_JOB_ID\""
  echo "node = \"$(hostname)\""
  echo "csv = \"$FM052A_MEMBENCH_CSV\""
  echo "csv_sha256 = \"$sha\""
  echo "created_utc = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
} > "$OUTDIR/fm052a_gh200_membench_${SLURM_JOB_ID}.toml"
echo "== membench complete: $OUTDIR"
