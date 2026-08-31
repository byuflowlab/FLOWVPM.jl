#!/usr/bin/env bash
# Run on the cluster only after the protected canonical H200 mature/gate jobs
# complete. It creates an architecture-qualified reference manifest; it does
# not alter, move, or copy the canonical run.
set -euo pipefail

GPU_JOB=${1:?usage: fm052_arch_adopt_h200.sh GPU_JOB GATE_JOB}
GATE_JOB=${2:?usage: fm052_arch_adopt_h200.sh GPU_JOB GATE_JOB}
[[ "$GPU_JOB" =~ ^[0-9]+$ && "$GATE_JOB" =~ ^[0-9]+$ ]] || { echo "job IDs must be numeric" >&2; exit 64; }

FPDIR=${FP052_DIR:-$HOME/FLOWPanel-052-h200}
CANONICAL_FP=${FP052_CANONICAL_DIR:-$HOME/FLOWPanel-052}
CANONICAL_RUN="$CANONICAL_FP/data/fm052c_gpu_mature"
CANONICAL_REPORT="$CANONICAL_FP/data/fm052_mature_gate"
RUN_DIR="$FPDIR/data/fm052_h200_mature_gpu_s"
REPORT_DIR="$FPDIR/data/fm052_multiarch/h200/comparisons/mature"
OUT="$FPDIR/data/fm052_multiarch/h200/manifests/fm052_h200_mature_${GPU_JOB}_result.toml"
PROBE_JOB=${FP052_PROBE_JOB:-}
PROBE="$FPDIR/data/fm052_multiarch/h200/manifests/fm052_h200_probe_${PROBE_JOB}_result.toml"

gpu_state=$(sacct -j "$GPU_JOB" --format=State -n -X | awk 'NF {print $1; exit}')
gate_state=$(sacct -j "$GATE_JOB" --format=State -n -X | awk 'NF {print $1; exit}')
test "$gpu_state" = COMPLETED && test "$gate_state" = COMPLETED || {
  echo "canonical H200 mature/gate not complete: gpu=$gpu_state gate=$gate_state" >&2; exit 65; }
test -d "$CANONICAL_RUN" && test -s "$CANONICAL_REPORT/fm052_gate.md" && test -s "$CANONICAL_REPORT/fm052_memory_gate.md" || {
  echo "canonical H200 acceptance artifacts are incomplete" >&2; exit 66; }

mkdir -p "$FPDIR/data" "$(dirname "$REPORT_DIR")"
if test -e "$RUN_DIR" || test -L "$RUN_DIR"; then
  test "$(readlink "$RUN_DIR")" = "$CANONICAL_RUN" || { echo "qualified H200 run path already exists with another target" >&2; exit 69; }
else
  ln -s "$CANONICAL_RUN" "$RUN_DIR"
fi
if test -e "$REPORT_DIR" || test -L "$REPORT_DIR"; then
  test "$(readlink "$REPORT_DIR")" = "$CANONICAL_REPORT" || { echo "qualified H200 report path already exists with another target" >&2; exit 70; }
else
  ln -s "$CANONICAL_REPORT" "$REPORT_DIR"
fi
canonical_log="$CANONICAL_FP/data/fm052c_gpu_mature.log"
qualified_log="${RUN_DIR}.log"
if test -f "$canonical_log" && test ! -e "$qualified_log" && test ! -L "$qualified_log"; then
  ln -s "$canonical_log" "$qualified_log"
fi

provenance=$(find -L "$RUN_DIR" -maxdepth 1 -name '*_provenance.toml' -print)
test "$(printf '%s\n' "$provenance" | grep -c .)" -eq 1 || { echo "canonical provenance is not unique" >&2; exit 67; }
metadata=$(find -L "$RUN_DIR" -maxdepth 1 -name '*_case_metadata.toml' -print)
test "$(printf '%s\n' "$metadata" | grep -c .)" -eq 1 || { echo "canonical metadata is not unique" >&2; exit 68; }

value() { sed -n "s/^$1 = \"\(.*\)\"/\1/p" "$provenance"; }
gpu_model=$(value cuda_gpu)
cuda_driver=$(value cuda_driver)
julia_version=$(value julia_version)
node=$(sacct -j "$GPU_JOB" --format=NodeList -n -X | awk 'NF {print $1; exit}')
partition=$(sacct -j "$GPU_JOB" --format=Partition -n -X | awk 'NF {print $1; exit}')
vram_bytes=$(sed -n 's/^solver_S_gpu_total_before_bytes = \([0-9]*\)$/\1/p' "$metadata")
vram_mib=$((vram_bytes / 1024 / 1024))
gpu_uuid=not-recorded-legacy-chain
log=$(find "$HOME/FLOWVPM-046" "$CANONICAL_FP" -maxdepth 1 -name "*${GPU_JOB}*.out" -print 2>/dev/null | head -1)
if test -n "$log"; then
  parsed_uuid=$(sed -n 's/.*(UUID: \([^)]*\)).*/\1/p' "$log" | head -1)
  test -z "$parsed_uuid" || gpu_uuid=$parsed_uuid
fi
compute_capability=not-recorded-legacy-chain
cpu_model=not-recorded-legacy-chain
identity_supplement=none
if test -n "$PROBE_JOB" && test -s "$PROBE" && test "$(sed -n 's/^status = "\(.*\)"/\1/p' "$PROBE")" = pass; then
  compute_capability=$(sed -n 's/^observed_compute_capability = "\(.*\)"/\1/p' "$PROBE")
  cpu_model=$(sed -n 's/^cpu_model = "\(.*\)"/\1/p' "$PROBE")
  identity_supplement=$PROBE
fi

mkdir -p "$(dirname "$OUT")"
cat >"$OUT" <<EOF
architecture = "h200"
stage = "mature"
status = "pass"
reason = "adopted_protected_canonical_h200_reference; legacy identity gaps explicitly marked"
legacy_canonical_chain = true
requested_gres = "h200"
requested_partition = "$partition"
observed_gpu_model = "$gpu_model"
observed_gpu_vram_mib = $vram_mib
observed_gpu_uuid = "$gpu_uuid"
observed_compute_capability = "$compute_capability"
cuda_driver = "$cuda_driver"
cuda_runtime = "not-recorded-legacy-chain"
julia_version = "$julia_version"
node = "$node"
partition = "$partition"
cpu_architecture = "x86_64"
cpu_model = "$cpu_model"
job_id = "$GPU_JOB"
job_name = "fp052gpu (legacy pre-extension name)"
gate_job_id = "$GATE_JOB"
run_dir = "$RUN_DIR"
report_dir = "$REPORT_DIR"
provenance_path = "$provenance"
identity_supplement = "$identity_supplement"
created_utc = "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
echo "$OUT"
