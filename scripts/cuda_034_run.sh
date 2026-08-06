#!/bin/bash
#SBATCH --job-name=vpm034
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 034 H200 validation job (DRAFT — cluster submission owned by the GPU
# queue; do not submit without authorization). Pattern: cuda_032_run.sh.
#
#   1. preflight: FLOWVPM direct-sum GPU regression (test/runtests_gpu.jl) —
#      the validated kernels used as the on-device reference below;
#   2. task 034 radix FMM coupling tests (test/runtests_gpu_fmm.jl):
#      Part A (host-resident transfer coupling, CPU) and Part B (device-
#      resident lifecycle: static U/J vs direct on cube+wake, Float64+Float32,
#      023 counter contract, capacity/varying-np, multi-step RK3 dynamic run)
#      under FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 (hard-fails if CUDA is absent).
#
# Environment: $VPM034_ENV must have FLOWVPM (branch gpu-full) and
# FastMultipole (branch matrix-ops) Pkg.develop'ed plus CUDA added — the
# fm023env local-toolkit pattern (compute nodes have no internet; instantiate
# on the login node, see cuda_034_submit.sh).
# no -u: /etc/profile.d scripts reference unset vars on the cluster
set -eo pipefail
source /etc/profile
# julia pinned to 1.11.7: 1.12.6 segfaults in host LLVM while JIT-compiling
# the device step (job 13058191); 1.11.7 is the toolchain of record.
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM034_DIR:-$HOME/FLOWVPM-034}"
ENVDIR="${VPM034_ENV:-$HOME/fm023env}"
cd "$WORKDIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

echo "=== preflight: FLOWVPM direct-sum GPU regression (runtests_gpu.jl) ==="
julia --project="$ENVDIR" -e 'using Test; import FLOWVPM, CUDA
CUDA.functional() || error("CUDA not functional on this node")
include("test/runtests_gpu.jl")'

echo "=== task 034: radix FMM coupling tests (host + device-resident) ==="
julia --project="$ENVDIR" test/runtests_gpu_fmm.jl

echo "vpm034 job complete"
