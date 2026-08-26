#!/usr/bin/env bash
#SBATCH --job-name=fp052-arch-invalid
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=fp052-arch-invalid-%j.out
# Isolated task-052 multi-architecture probe/smoke/mature driver.
# Partition, GRES, architecture-qualified job name, and output path MUST be
# supplied by the explicit sbatch commands in the task worklog.

source /etc/profile
set -euo pipefail

: "${FP052_ARCH:?}"
: "${FP052_STAGE:?FP052_STAGE must be probe, smoke, or mature}"
: "${FP052_GPU_GRES:?}"
: "${FP052_PARTITION:?}"
case "$FP052_ARCH" in h200|h100|gh200|b200|l40s) ;; *) echo "invalid canonical architecture slug: $FP052_ARCH" >&2; exit 64;; esac
case "$FP052_STAGE" in probe|smoke|mature) ;; *) echo "invalid architecture stage: $FP052_STAGE" >&2; exit 64;; esac

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052-$FP052_ARCH}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-052-$FP052_ARCH}"
FMDIR="${FP052_FMDIR:-$HOME/FastMultipole-052-$FP052_ARCH}"
ENVDIR="${FP052_ENV:-$HOME/fm052env-$FP052_ARCH}"
ARCH_ROOT="$FPDIR/data/fm052_multiarch/$FP052_ARCH"
MANIFEST_DIR="$ARCH_ROOT/manifests"
REPORT_ROOT="$ARCH_ROOT/comparisons"
: "${SLURM_JOB_ID:?fm052_arch_run.sh must run under Slurm}"
JOB_TAG=$SLURM_JOB_ID
RESULT_MANIFEST="$MANIFEST_DIR/fm052_${FP052_ARCH}_${FP052_STAGE}_${JOB_TAG}_result.toml"
SUBMISSION_MANIFEST="$MANIFEST_DIR/fm052_${FP052_ARCH}_${FP052_STAGE}_${JOB_TAG}_submission.toml"

source "$VPMDIR/scripts/fm052_common.sh"
source "$VPMDIR/scripts/fm052_arch_common.sh"

mkdir -p "$MANIFEST_DIR" "$REPORT_ROOT" "$ARCH_ROOT/slurm"
expected_job_name="fp052-$FP052_ARCH-$FP052_STAGE"
# Under the combined single-allocation chain (fm052_arch_chain.sh) all stages
# share one Slurm job whose name is the chain's, not the stage's.
test "${FP052_CHAIN:-0}" != 1 || expected_job_name="fp052-$FP052_ARCH-chain"
export FP052_SLURM_OUTPUT_PATTERN="$ARCH_ROOT/slurm/${expected_job_name}-%j.out"
test "${SLURM_JOB_NAME:-}" = "$expected_job_name" || {
  echo "architecture-qualified Slurm job-name mismatch: expected=$expected_job_name observed=${SLURM_JOB_NAME:-missing}" >&2
  exit 74
}

finalized=false
fm052_arch_finalize_on_exit() {
  local rc=$?
  if test "$finalized" != true; then
    fm052_arch_write_stage_manifest "$RESULT_MANIFEST" fail "exit_${rc}_line_${BASH_LINENO[0]}" \
      "$FP052_STAGE" "${CURRENT_RUN_DIR:-}" "${CURRENT_REPORT_DIR:-}"
  fi
  exit "$rc"
}
trap fm052_arch_finalize_on_exit EXIT

fm052_arch_validate_identity
fm052_arch_write_stage_manifest "$SUBMISSION_MANIFEST" submitted "allocation_identity_validated" \
  "$FP052_STAGE" "" ""

if test "$FP052_ARCH" = gh200; then
  JULIA_BIN="${FP052_JULIA_BIN:-/home/rander39/julia/julia-1.11.7/bin/julia}"
  export JULIA_DEPOT_PATH="${FP052_DEPOT:-$HOME/fm052depot-gh200}"
  test -x "$JULIA_BIN" || { echo "missing ARM-native Julia executable: $JULIA_BIN" >&2; exit 75; }
else
  module load cuda julia/1.11.7-6bmogfl
  JULIA_BIN="${FP052_JULIA_BIN:-julia}"
fi

THREADS=${SLURM_CPUS_PER_TASK:-64}
export JULIA_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export BLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FP052_DIR="$FPDIR" FP052_VPMDIR="$VPMDIR" FP052_FMDIR="$FMDIR" FP052_ENV="$ENVDIR"
export FP052_JULIA_BIN="$JULIA_BIN"

cd "$FPDIR"
echo "task052 architecture=$FP052_ARCH stage=$FP052_STAGE job=${SLURM_JOB_ID:-missing} node=$FM052_OBS_NODE"
echo "requested gres=$FP052_GPU_GRES partition=$FP052_PARTITION; observed gpu=$FM052_OBS_GPU_NAME vram_mib=$FM052_OBS_GPU_VRAM_MIB uuid=$FM052_OBS_GPU_UUID cc=$FM052_OBS_GPU_CC"
echo "cpu arch=$FM052_OBS_CPU_ARCH model=$FM052_OBS_CPU_MODEL threads=$THREADS"
nvidia-smi -L

run_driver() {
  local run_name=$1
  local log="data/${run_name}.log"
  shift
  CURRENT_RUN_DIR="$FPDIR/data/$run_name"
  mkdir -p "$CURRENT_RUN_DIR"
  "$VPMDIR/scripts/fm052_provenance.sh" \
    "$CURRENT_RUN_DIR/${run_name}_provenance.toml" "$FM052_CHECKPOINT_ROOT"
  local start_ns end_ns elapsed_ns elapsed
  start_ns=$(date +%s%N)
  env "$@" RUN_NAME="$run_name" \
    "$JULIA_BIN" --project="$ENVDIR" --threads="$THREADS" \
      examples/rotor_hover_pressure_comparison.jl 2>&1 | tee "$log"
  end_ns=$(date +%s%N)
  elapsed_ns=$((end_ns - start_ns))
  printf -v elapsed '%d.%09d' "$((elapsed_ns / 1000000000))" "$((elapsed_ns % 1000000000))"
  printf '%s\n' "$elapsed" > "$CURRENT_RUN_DIR/${run_name}_process_wall_s.txt"
}

source_gate() {
  local log=$1 expected=$2 expected_path=$3
  local cpu_s_count gpu_s_count backend_count
  cpu_s_count=$(grep -c source_influence_s_gemv "$log" || true)
  gpu_s_count=$(grep -c source_influence_s_gpu_gemv "$log" || true)
  backend_count=$(grep -c source_influence_backend "$log" || true)
  if test "$expected_path" = gpu; then
    test "$gpu_s_count" -eq "$expected" && test "$cpu_s_count" -eq 0
    test "$(grep -c 'GPU-S cleanup verified' "$log" || true)" -eq 1
    test "$(grep -c source_s_gpu_upload "$log" || true)" -eq 1
  else
    test "$cpu_s_count" -eq "$expected" && test "$gpu_s_count" -eq 0
  fi
  test "$backend_count" -eq 0
}

require_stage_pass() {
  local stage=$1 job_id=$2
  local path="$MANIFEST_DIR/fm052_${FP052_ARCH}_${stage}_${job_id}_result.toml"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid prior-stage job ID for $stage: $job_id" >&2; exit 82; }
  test -s "$path" || { echo "required prior-stage result missing: $path" >&2; exit 80; }
  test "$(sed -n 's/^status = "\(.*\)"/\1/p' "$path")" = pass || {
    echo "required prior stage did not pass: $path" >&2; exit 81; }
}

case "$FP052_STAGE" in
  probe)
    probe_dir="$ARCH_ROOT/compatibility_probe/job-$JOB_TAG"
    CURRENT_REPORT_DIR="$probe_dir"
    mkdir -p "$probe_dir"
    "$VPMDIR/scripts/fm052_provenance.sh" \
      "$probe_dir/fm052_${FP052_ARCH}_probe_provenance.toml" "$FM052_CHECKPOINT_ROOT"
    "$JULIA_BIN" --project="$ENVDIR" --threads="$THREADS" \
      "$VPMDIR/scripts/fm052_arch_probe.jl" "$FP052_ARCH" \
      "$probe_dir/fm052_${FP052_ARCH}_compatibility.toml" \
      2>&1 | tee "$probe_dir/fm052_${FP052_ARCH}_compatibility.log"
    grep -Fq "$FPDIR/" "$probe_dir/fm052_${FP052_ARCH}_compatibility.toml"
    grep -Fq "$VPMDIR/" "$probe_dir/fm052_${FP052_ARCH}_compatibility.toml"
    grep -Fq "$FMDIR/" "$probe_dir/fm052_${FP052_ARCH}_compatibility.toml"
    if fm052_arch_memory_eligibility; then
      :
    else
      rc=$?
      test "$rc" -eq 2 || exit "$rc"
      fm052_arch_write_stage_manifest "$RESULT_MANIFEST" ineligible "$FM052_INELIGIBLE_REASON" \
        probe "" "$probe_dir"
      finalized=true
      exit 0
    fi
    fm052_arch_write_stage_manifest "$RESULT_MANIFEST" pass compatibility_and_official_memory_preflight_passed \
      probe "" "$probe_dir"
    ;;
  smoke)
    : "${FP052_PROBE_JOB:?smoke requires the inspected passing probe job ID}"
    require_stage_pass probe "$FP052_PROBE_JOB"
    fm052_arch_memory_eligibility
    report_dir="$REPORT_ROOT/smoke/job-$JOB_TAG"
    CURRENT_REPORT_DIR="$report_dir"
    mkdir -p "$report_dir"
    smoke_env=(
      RHPC_MESH=40_40 NT=36 NREVS=0.08333333333333333
      SPINUP_REVS=0 FREESTREAM_RAMP_REVS=0 FREESTREAM_HOLD_REVS=0
      FREESTREAM_WITHDRAW_REVS=0 SETTLE_REVS=0
      P_PER_STEP=6 OVERLAP=3.0 BERNOULLI_ONLY=true SAVE_VTK=true
      RHPC_SOLVER_S=true BLAS_NUM_THREADS="$THREADS" BLAS_NUM_THREADS_MARCH=8
      FLOWPANEL_STEP_TIMERS=true
    )
    cpu_run="fm052_${FP052_ARCH}_smoke_cpu_s_${JOB_TAG}"
    gpu_run="fm052_${FP052_ARCH}_smoke_gpu_s_${JOB_TAG}"
    run_driver "$cpu_run" "${smoke_env[@]}"
    run_driver "$gpu_run" "${smoke_env[@]}" "${FM052_GPU_ENV[@]}" \
      RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 FLOWPANEL_GPU_TIMERS=true
    source_gate "data/${cpu_run}.log" 3 cpu
    source_gate "data/${gpu_run}.log" 3 gpu
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" compare \
      "data/$cpu_run" "data/$gpu_run" "$report_dir"
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
      "data/$gpu_run" "data/${gpu_run}.log" "$report_dir"
    CURRENT_RUN_DIR="$FPDIR/data/$gpu_run"
    fm052_arch_write_stage_manifest "$RESULT_MANIFEST" pass smoke_parity_upload_reuse_cleanup_passed \
      smoke "$CURRENT_RUN_DIR" "$report_dir"
    ;;
  mature)
    : "${FP052_PROBE_JOB:?mature requires the inspected passing probe job ID}"
    : "${FP052_SMOKE_JOB:?mature requires the inspected passing smoke job ID}"
    require_stage_pass probe "$FP052_PROBE_JOB"
    require_stage_pass smoke "$FP052_SMOKE_JOB"
    fm052_arch_memory_eligibility
    fm052_preflight_checkpoint
    cpu_run="${FP052_CPU_RUN:-$HOME/FLOWPanel-052/data/fm052r_cpu_mature_pinned}"
    tolerance="${FP052_TOLERANCE:-$HOME/FLOWPanel-052/data/fm052_campaign_lock/fm052_locked_tolerances.toml}"
    test -d "$cpu_run" || { echo "canonical CPU mature reference missing: $cpu_run" >&2; exit 76; }
    test -s "$tolerance" || { echo "locked tolerance missing: $tolerance" >&2; exit 77; }
    gpu_run="fm052_${FP052_ARCH}_mature_gpu_s_${JOB_TAG}"
    report_dir="$REPORT_ROOT/mature/job-$JOB_TAG"
    CURRENT_REPORT_DIR="$report_dir"
    mkdir -p "$report_dir" "data/$gpu_run"
    candidate_provenance="$FPDIR/data/$gpu_run/${gpu_run}_provenance.toml"
    "$VPMDIR/scripts/fm052_provenance.sh" "$candidate_provenance" "$FM052_CHECKPOINT_ROOT"
    cpu_provenance=$(find "$cpu_run" -maxdepth 1 -name '*_provenance.toml' -print)
    test "$(printf '%s\n' "$cpu_provenance" | grep -c .)" -eq 1 || {
      echo "expected exactly one canonical CPU provenance file in $cpu_run" >&2; exit 78; }
    fm052_arch_reference_gate "$cpu_provenance" "$candidate_provenance" \
      "$report_dir/fm052_${FP052_ARCH}_cpu_reference_gate.md"
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" manifest-gate \
      "${FP052_CPU_ENV:-$HOME/fm052env_cuda63_geoiofree}/Manifest.toml" \
      "$ENVDIR/Manifest.toml" "$report_dir/fm052_${FP052_ARCH}_package_manifest_gate.md"
    run_driver "$gpu_run" "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
      NREVS=19.5 RESTART_STEP="$FM052_RESTART_STEP" RESTART_NAME="$FM052_RESTART_NAME" \
      RESTART_PATH="$FM052_CHECKPOINT_ROOT" RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 \
      FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true
    source_gate "data/${gpu_run}.log" 36 gpu
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" gate \
      "$cpu_run" "data/$gpu_run" "$tolerance" "$report_dir"
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
      "data/$gpu_run" "data/${gpu_run}.log" "$report_dir"
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
      "data/$gpu_run" 720 755
    "$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" memory-gate \
      "data/$gpu_run" "$report_dir" 16
    CURRENT_RUN_DIR="$FPDIR/data/$gpu_run"
    fm052_arch_write_stage_manifest "$RESULT_MANIFEST" pass correctness_artifact_source_memory_gates_passed \
      mature "$CURRENT_RUN_DIR" "$report_dir"
    ;;
  *) echo "invalid FP052_STAGE: $FP052_STAGE" >&2; exit 79 ;;
esac

finalized=true
trap - EXIT
echo "task 052 $FP052_ARCH $FP052_STAGE result: $RESULT_MANIFEST"
