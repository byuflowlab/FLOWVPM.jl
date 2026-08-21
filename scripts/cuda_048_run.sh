#!/bin/bash
#SBATCH --job-name=vpm048
#SBATCH --gpus=h200:1
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
#   4. 047 device wiring check: construction-locked setting late-flip errors
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
julia --project="$ENVDIR" test/runtests_gpu_fmm.jl

echo "=== stage 4: 047 device construction-lock check ==="
julia --project="$ENVDIR" scripts/fm048_device_settings_check.jl

echo "vpm048 job complete"
