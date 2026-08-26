#!/usr/bin/env bash
# 052a Phase-B CUDA testset preflight on GH200 (aarch64, offline depot).
# Runs the standard FastMultipole test/cuda_*_test.jl set, one process per
# file, with FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 so a lifecycle load failure
# is an error, never a silent skip. The CUDARadixTransferCounters contract
# (cuda_radix_lifecycle_test.jl) must hold unchanged on C2C hardware.
# Submit as, e.g.:
#   sbatch --job-name=fp052a-gh200-cudatests --partition=mgh \
#     --gres=gpu:gh200:1 --constraint=arm --cpus-per-task=16 --mem=128G \
#     --time=03:00:00 \
#     --output=$HOME/FLOWPanel-052-gh200/data/fm052_multiarch/gh200/slurm/fp052a-gh200-cudatests-%j.out \
#     $HOME/FLOWVPM-052-gh200/scripts/fm052a_gh200_cuda_tests.sh
source /etc/profile
set -uo pipefail

: "${SLURM_JOB_ID:?must run under Slurm}"

JULIA_BIN="${FP052_JULIA_BIN:-/home/rander39/julia/julia-1.11.7/bin/julia}"
ENVDIR="${FP052_ENV:-$HOME/fm052env-gh200}"
FMDIR="${FP052_FMDIR:-$HOME/FastMultipole-052-gh200}"
export JULIA_DEPOT_PATH="${FP052_DEPOT:-$HOME/fm052depot-gh200}"
export JULIA_PKG_OFFLINE=1
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
# single GPU allocated: the twogpu file self-skips (do NOT set
# FASTMULTIPOLE_REQUIRE_TWOGPU_TESTS)

echo "== 052a gh200 cuda testset preflight: job=$SLURM_JOB_ID node=$(hostname) arch=$(uname -m)"
test -x "$JULIA_BIN" || { echo "missing ARM Julia: $JULIA_BIN" >&2; exit 75; }
nvidia-smi -L
cd "$FMDIR"

pass=0; fail=0; failed_files=""
for f in test/cuda_*_test.jl; do
  echo "=== BEGIN $f ($(date -u +%H:%M:%S)) ==="
  if "$JULIA_BIN" --project="$ENVDIR" "$f"; then
    echo "=== PASS $f ==="
    pass=$((pass+1))
  else
    echo "=== FAIL $f (exit $?) ==="
    fail=$((fail+1)); failed_files="$failed_files $f"
  fi
done

echo "== summary: pass=$pass fail=$fail failed:[$failed_files]"
if test "$fail" -ne 0; then
  echo "== 052a gh200 CUDA testset preflight FAILED"
  exit 1
fi
echo "== 052a gh200 CUDA testset preflight PASSED (job $SLURM_JOB_ID)"
