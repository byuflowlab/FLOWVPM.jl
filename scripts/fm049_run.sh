#!/bin/bash
#SBATCH --job-name=vpm049
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 049 H200 rotor-field verification job. Pattern: cuda_048_run.sh
# (julia pinned to 1.11.7: 1.12.6 segfaults JIT-compiling the device step,
# job 13058191). Runs the fm049 driver on the p018 step-710 snapshot dump
# (n = 210,056): GPU direct reference, device UJ +/- SFS parity and timing,
# full device-resident RK3 nextstep (U_prev broadcast fix), residency A/B.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM049_DIR:-$HOME/FLOWVPM-046}"
ENVDIR="${VPM049_ENV:-$HOME/fm048env}"
BINFILE="${VPM049_BIN:-$HOME/FLOWVPM-046/data/fm049/p018_710_particles.bin}"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=8

cd "$WORKDIR"
echo "=== fm049 rotor field GPU verification ==="
julia --project="$ENVDIR" scripts/fm049_rotor_verify.jl "$BINFILE"

echo "vpm049 job complete"
