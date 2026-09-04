#!/bin/bash
#SBATCH --job-name=vpm034
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 034 H200 validation job. Pattern: cuda_032_run.sh.
#
#   1. preflight: FastMultipole CUDA device-interface tests
#      (test/cuda_radix_interface_test.jl, hard-required via
#      FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1) — the 032 surface the coupling sits on;
#   2. preflight: FLOWVPM direct-sum GPU regression (test/runtests_gpu.jl) —
#      the validated kernels used as the on-device reference below;
#   3. task 034 radix FMM coupling tests (test/runtests_gpu_fmm.jl):
#      Part A (host-resident transfer coupling, CPU) and Part B (device-
#      resident lifecycle: static U/J vs direct on cube+wake, Float64+Float32,
#      023 counter contract, allocation probe, capacity/varying-np, multi-step
#      RK3 dynamic run, coarse solve-time sanity print);
#   4. task 034 closing gate: sha256-verify the synced 033 sampled-direct
#      references, then scripts/cuda_034_refcheck.jl — device-resident U/J on
#      the exact 033 case constructions (cube+wake, n=1e4+1e5) vs the
#      checksummed references (Float64 gated at u_rel_rms<=1e-3, Float32
#      reported). VPM034_REFCHECK_ONLY=1 skips stages 1-3 for iteration.
#
# Environment: $VPM034_ENV (default ~/fm034env, task-034-owned — NOT the
# shared fm023env) has FLOWVPM (~/FLOWVPM-034, branch gpu-full) and
# FastMultipole (~/FastMultipole-034, branch matrix-ops) Pkg.develop'ed plus
# CUDA with local-toolkit JLL preferences (compute nodes have no internet;
# instantiate on the login node, see cuda_034_submit.sh).
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
FMDIR="${VPM034_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${VPM034_ENV:-$HOME/fm034env}"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

if [ "${VPM034_REFCHECK_ONLY:-0}" != "1" ]; then
    echo "=== preflight: FastMultipole CUDA interface tests (032 surface) ==="
    cd "$FMDIR"
    julia --project="$ENVDIR" test/cuda_radix_interface_test.jl

    cd "$WORKDIR"
    echo "=== preflight: FLOWVPM direct-sum GPU regression (runtests_gpu.jl) ==="
    julia --project="$ENVDIR" -e 'using Test; import FLOWVPM, CUDA
CUDA.functional() || error("CUDA not functional on this node")
include("test/runtests_gpu.jl")'

    echo "=== task 034: radix FMM coupling tests (host + device-resident) ==="
    julia --project="$ENVDIR" test/runtests_gpu_fmm.jl
fi

echo "=== task 034: 033-checksummed-reference accuracy comparison ==="
# verify the synced references byte-for-byte against the 033 sha256 manifest
# before trusting them as the accuracy ground truth
cd "$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references"
sha256sum -c direct_reference_checksums.sha256
cd "$WORKDIR"
julia --project="$ENVDIR" scripts/cuda_034_refcheck.jl "$FMDIR" 10000 100000

echo "vpm034 job complete"
