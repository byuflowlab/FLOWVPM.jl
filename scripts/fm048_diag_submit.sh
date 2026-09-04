#!/bin/bash
#SBATCH --job-name=vpm048diag
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Task-048 graph-replay J-defect localization (see fm048_replay_diag2.jl).
# Normal H200 QoS (the H200 partitions do not support --qos=test).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${VPM048_DIR:-$HOME/FLOWVPM-046}"
ENVDIR="${VPM048_ENV:-$HOME/fm048env}"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=4

cd "$WORKDIR"
DIAG_LOG="$WORKDIR/fm048_diag2_${SLURM_JOB_ID}.log"
julia --project="$ENVDIR" scripts/fm048_replay_diag2.jl 2>&1 | tee "$DIAG_LOG"
echo "diag_raw_log=$DIAG_LOG"
echo "diag_raw_log_sha256=$(shasum -a 256 "$DIAG_LOG" | awk '{print $1}')"
echo "vpm048diag job complete"
