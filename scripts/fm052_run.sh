#!/bin/bash
#SBATCH --job-name=fp052gpu
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# GPU stages: a=pass parity, b=reduced CPU-S/GPU-S smoke,
# c=step-719 mature GPU continuation, d=exact 1080-step cold acceptance run.
source /etc/profile
set -euo pipefail
module load cuda julia/1.11.7-6bmogfl

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
ENVDIR="${FP052_ENV:-$HOME/fm052env_cuda63_geoiofree}"
STAGES="${FP052_STAGES:-a b c}"
source "$VPMDIR/scripts/fm052_common.sh"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=64
export BLAS_NUM_THREADS=64

cd "$FPDIR"
mkdir -p data
echo "node=$(hostname) job=${SLURM_JOB_ID:-local} stages=$STAGES"
nvidia-smi -L

run_driver() {
  local run_name=$1
  local log="data/${run_name}.log"
  shift
  mkdir -p "data/$run_name"
  # The driver appends to existing PVD collection files, so a rerun into a
  # stale dir doubles their entries and the verify gate (exactly-one
  # reference per step) rejects them. The .vtm payloads are overwritten in
  # place, so clearing the collections alone makes reruns idempotent.
  find "data/$run_name" -maxdepth 1 -name '*.pvd' -delete
  "$VPMDIR/scripts/fm052_provenance.sh" \
    "data/$run_name/${run_name}_provenance.toml" "$FM052_CHECKPOINT_ROOT"
  local start_ns end_ns elapsed_ns elapsed
  start_ns=$(date +%s%N)
  env "$@" RUN_NAME="$run_name" \
    julia --project="$ENVDIR" --threads=64 examples/rotor_hover_pressure_comparison.jl \
    2>&1 | tee "$log"
  end_ns=$(date +%s%N)
  elapsed_ns=$((end_ns - start_ns))
  printf -v elapsed '%d.%09d' "$((elapsed_ns / 1000000000))" "$((elapsed_ns % 1000000000))"
  printf '%s\n' "$elapsed" > "data/$run_name/${run_name}_process_wall_s.txt"
  echo "process wall: $elapsed s"
}

source_gate() {
  local log=$1 expected=$2 expected_path=$3
  local cpu_s_count gpu_s_count backend_count
  cpu_s_count=$(grep -c source_influence_s_gemv "$log" || true)
  gpu_s_count=$(grep -c source_influence_s_gpu_gemv "$log" || true)
  backend_count=$(grep -c source_influence_backend "$log" || true)
  if [ "$expected_path" = gpu ]; then
    test "$gpu_s_count" -eq "$expected" && test "$cpu_s_count" -eq 0 || {
      echo "GPU-S gate failed: expected gpu=$expected cpu=0; found gpu=$gpu_s_count cpu=$cpu_s_count"; exit 1; }
    test "$(grep -c 'GPU-S cleanup verified' "$log" || true)" -eq 1 || {
      echo "GPU-S cleanup gate failed"; exit 1; }
  elif [ "$expected_path" = cpu ]; then
    test "$cpu_s_count" -eq "$expected" && test "$gpu_s_count" -eq 0 || {
      echo "CPU-S gate failed: expected cpu=$expected gpu=0; found cpu=$cpu_s_count gpu=$gpu_s_count"; exit 1; }
  else
    echo "unknown source path $expected_path"; exit 1
  fi
  test "$backend_count" -eq 0 || {
    echo "source-path gate failed: backend occurred $backend_count times"; exit 1; }
}

for stage in $STAGES; do
case "$stage" in
a)
  julia --project="$ENVDIR" --threads=4 examples/fm051_pass_parity.jl
  FM051_MODE=cuda julia --project="$ENVDIR" --threads=4 examples/fm051_pass_parity.jl
  ;;
b)
  SMOKE_ENV=(
    RHPC_MESH=40_40 NT=36 NREVS=0.08333333333333333
    SPINUP_REVS=0 FREESTREAM_RAMP_REVS=0 FREESTREAM_HOLD_REVS=0
    FREESTREAM_WITHDRAW_REVS=0 SETTLE_REVS=0
    P_PER_STEP=6 OVERLAP=3.0 BERNOULLI_ONLY=true SAVE_VTK=true
    RHPC_SOLVER_S=true BLAS_NUM_THREADS=64 BLAS_NUM_THREADS_MARCH=8
    FLOWPANEL_STEP_TIMERS=true
  )
  run_driver fm052b_cpu_s "${SMOKE_ENV[@]}"
  run_driver fm052b_gpu_s "${SMOKE_ENV[@]}" "${FM052_GPU_ENV[@]}" \
    RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 FLOWPANEL_GPU_TIMERS=true
  source_gate data/fm052b_cpu_s.log 3 cpu
  source_gate data/fm052b_gpu_s.log 3 gpu
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" compare \
    data/fm052b_cpu_s data/fm052b_gpu_s data/fm052b_gpu_s
  ;;
c)
  fm052_preflight_checkpoint
  run_driver fm052c_gpu_mature "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
    NREVS=19.5 RESTART_STEP="$FM052_RESTART_STEP" RESTART_NAME="$FM052_RESTART_NAME" \
    RESTART_PATH="$FM052_CHECKPOINT_ROOT" RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 \
    FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true
  source_gate data/fm052c_gpu_mature.log 36 gpu
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
    data/fm052c_gpu_mature data/fm052c_gpu_mature.log
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
    data/fm052c_gpu_mature 720 755
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" memory-gate \
    data/fm052c_gpu_mature data/fm052c_gpu_mature 16
  ;;
d)
  run_driver fm052d_gpu_1080 "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
    NREVS=28.5 FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true
  source_gate data/fm052d_gpu_1080.log 1080 gpu
  grep -q '^n_steps = 1080$' data/fm052d_gpu_1080/fm052d_gpu_1080_case_metadata.toml
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
    data/fm052d_gpu_1080 data/fm052d_gpu_1080.log
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
    data/fm052d_gpu_1080 0 1079
  ;;
*) echo "unknown stage: $stage"; exit 1 ;;
esac
done
