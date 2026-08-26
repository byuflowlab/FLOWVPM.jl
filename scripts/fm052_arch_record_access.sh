#!/usr/bin/env bash
# Record a no-job scheduler/account result for the cross-architecture summary.
set -euo pipefail

ARCH=${1:?usage: fm052_arch_record_access.sh ARCH REASON}
REASON=${2:?usage: fm052_arch_record_access.sh ARCH REASON}
case "$ARCH" in
  h100) GRES=h100; PARTITION=cs2; CPU_ARCH=x86_64 ;;
  b200) GRES=b200; PARTITION=cs3; CPU_ARCH=x86_64 ;;
  h200) GRES=h200; PARTITION=m13h; CPU_ARCH=x86_64 ;;
  gh200) GRES=gh200; PARTITION=mgh; CPU_ARCH=aarch64 ;;
  l40s) GRES=l40s; PARTITION=m13l; CPU_ARCH=x86_64 ;;
  *) echo "invalid architecture slug: $ARCH" >&2; exit 64 ;;
esac
FPDIR=${FP052_DIR:-$HOME/FLOWPanel-052-$ARCH}
RESULT_ROOT=${FP052_RESULT_ROOT:-$FPDIR/data/fm052_multiarch}
DATE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$RESULT_ROOT/$ARCH/manifests/fm052_${ARCH}_access_${DATE_TAG}_result.toml"
test ! -e "$OUT" || { echo "refusing to overwrite access result: $OUT" >&2; exit 65; }
mkdir -p "$(dirname "$OUT")"
cat >"$OUT" <<EOF
architecture = "$ARCH"
stage = "access"
status = "ineligible"
reason = "$REASON"
requested_gres = "$GRES"
requested_partition = "$PARTITION"
observed_gpu_model = "not_observed_no_allocation"
observed_gpu_vram_mib = -1
observed_gpu_uuid = "not_observed_no_allocation"
observed_compute_capability = "not_observed_no_allocation"
cuda_driver = "not_observed_no_allocation"
cuda_runtime = "not_observed_no_allocation"
julia_version = "not_observed_no_allocation"
node = "not_allocated"
partition = "$PARTITION"
cpu_architecture = "$CPU_ARCH"
cpu_model = "not_observed_no_allocation"
job_id = "none"
job_name = "none"
run_dir = ""
report_dir = ""
created_utc = "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
echo "$OUT"
