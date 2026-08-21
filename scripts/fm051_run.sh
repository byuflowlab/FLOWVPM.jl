#!/bin/bash
#SBATCH --job-name=vpm051
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 051 stage 1 H200 rectangular direct benchmark job. Pattern:
# fm049_run.sh (julia pinned to 1.11.7: 1.12.6 segfaults JIT-compiling the
# device step, job 13058191). Runs the fm051 rectangular driver on the p018
# step-710 snapshot dump (n = 210,056): device rectangular pass 1
# (particles -> 36,752 panel centers) and pass 2 (panels -> 210k particle
# positions), each parity-checked on 2000 sampled targets vs the host
# reference and timed (median of 5).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM051_DIR:-$HOME/FLOWVPM-046}"
ENVDIR="${VPM051_ENV:-$HOME/fm048env}"
BINFILE="${VPM051_BIN:-$HOME/FLOWVPM-046/data/fm049/p018_710_particles.bin}"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=8

cd "$WORKDIR"
echo "=== fm051 rectangular direct benchmark ==="
julia --project="$ENVDIR" scripts/fm051_rect_bench.jl "$BINFILE"

echo "vpm051 job complete"
