#!/bin/bash
#SBATCH --job-name=fp052cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Mature CPU-S continuation from the same protected step-719 checkpoint.
source /etc/profile
set -euo pipefail
module load julia/1.11.7-6bmogfl

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
ENVDIR="${FP052_ENV:-$HOME/fm052env_cuda63_geoiofree}"
source "$VPMDIR/scripts/fm052_common.sh"
export JULIA_NUM_THREADS=64
export BLAS_NUM_THREADS=64

cd "$FPDIR"
fm052_preflight_checkpoint
RUN=fm052r_cpu_mature
mkdir -p "data/$RUN"
"$VPMDIR/scripts/fm052_provenance.sh" "data/$RUN/${RUN}_provenance.toml" "$FM052_CHECKPOINT_ROOT"
start_ns=$(date +%s%N)
env "${FM052_PRODUCTION_ENV[@]}" NREVS=19.5 \
  RESTART_STEP="$FM052_RESTART_STEP" RESTART_NAME="$FM052_RESTART_NAME" \
  RESTART_PATH="$FM052_CHECKPOINT_ROOT" FLOWPANEL_STEP_TIMERS=true RUN_NAME="$RUN" \
  julia --project="$ENVDIR" --threads=64 examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${RUN}.log"
end_ns=$(date +%s%N)
elapsed_ns=$((end_ns - start_ns))
printf -v elapsed '%d.%09d' "$((elapsed_ns / 1000000000))" "$((elapsed_ns % 1000000000))"
printf '%s\n' "$elapsed" > "data/$RUN/${RUN}_process_wall_s.txt"
test "$(grep -c source_influence_s_gemv data/${RUN}.log || true)" -eq 36
test "$(grep -c source_influence_s_gpu_gemv data/${RUN}.log || true)" -eq 0
test "$(grep -c source_influence_backend data/${RUN}.log || true)" -eq 0
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify "data/$RUN" 720 755
