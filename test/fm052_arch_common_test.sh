#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FIXTURE=$(mktemp -d "${TMPDIR:-/tmp}/fm052-arch-test.XXXXXX")
trap 'rm -rf "$FIXTURE"' EXIT
REAL_PATH=$PATH

printf '%s\n' '#!/usr/bin/env bash' \
  'if test "${1:-}" = -L; then echo "GPU 0: ${FM052_FAKE_GPU_NAME:-NVIDIA H200} (UUID: GPU-test)"; else printf "%s\n" "${FM052_FAKE_SMI_ROW:-NVIDIA H200, 143771, GPU-test, 9.0, 580.1}"; fi' \
  > "$FIXTURE/nvidia-smi"
printf '%s\n' '#!/usr/bin/env bash' 'test "${1:-}" = -m && echo x86_64' > "$FIXTURE/uname"
printf '%s\n' '#!/usr/bin/env bash' 'echo test-node' > "$FIXTURE/hostname"
printf '%s\n' '#!/usr/bin/env bash' 'echo "Model name: Test CPU"' > "$FIXTURE/lscpu"
chmod +x "$FIXTURE/nvidia-smi" "$FIXTURE/uname" "$FIXTURE/hostname" "$FIXTURE/lscpu"
export PATH="$FIXTURE:$REAL_PATH"
export FP052_NVIDIA_SMI_BIN="$FIXTURE/nvidia-smi"

source "$ROOT/scripts/fm052_arch_common.sh"

FP052_ARCH=h200 FP052_GPU_GRES=h200 FP052_PARTITION=m13h fm052_arch_configure

set +e
(FP052_ARCH=hopper FP052_GPU_GRES=h200 FP052_PARTITION=m13h; fm052_arch_configure) >/dev/null 2>&1
test $? -eq 64
(FP052_ARCH=h200 FP052_GPU_GRES=h100 FP052_PARTITION=m13h; fm052_arch_configure) >/dev/null 2>&1
test $? -eq 65
set -e

export FP052_ARCH=h200 FP052_GPU_GRES=h200 FP052_PARTITION=m13h
export SLURM_JOB_ID=123 SLURM_JOB_PARTITION=m13h SLURM_JOB_NAME=fp052-h200-probe
export FM052_FAKE_SMI_ROW='NVIDIA H200, 143771, GPU-test, 9.0, 580.1'
fm052_arch_validate_identity

set +e
FM052_FAKE_SMI_ROW='NVIDIA H100 80GB HBM3, 81559, GPU-wrong, 9.0, 580.1' fm052_arch_validate_identity >/dev/null 2>&1
test $? -eq 69
set -e

export FM052_FAKE_SMI_ROW='NVIDIA L40S, 46000, GPU-low, 8.9, 580.1'
fm052_arch_observe
set +e
fm052_arch_memory_eligibility >/dev/null 2>&1
rc=$?
set -e
test "$rc" -eq 2
test -n "$FM052_INELIGIBLE_REASON"

export FP052_JULIA_BIN=/usr/bin/false FP052_ENV="$FIXTURE/no-env"
manifest="$FIXTURE/fm052_l40s_probe_result.toml"
FP052_ARCH=l40s FP052_GPU_GRES=l40s FP052_PARTITION=m13l \
  fm052_arch_write_stage_manifest "$manifest" ineligible "$FM052_INELIGIBLE_REASON" probe "" "$FIXTURE"
grep -q '^architecture = "l40s"$' "$manifest"
grep -q '^status = "ineligible"$' "$manifest"
grep -q '^reason = "startup_vram_' "$manifest"

# Pkg.dependencies() returns UUID => PackageInfo; PackageInfo has no uuid field.
grep -Fq 'for (uuid, package) in sort!(collect(Pkg.dependencies())' \
  "$ROOT/scripts/fm052_arch_probe.jl"
grep -Fq 'Base.require(Base.PkgId(uuid, name))' \
  "$ROOT/scripts/fm052_arch_probe.jl"
if grep -Fq 'package.uuid' "$ROOT/scripts/fm052_arch_probe.jl"; then
  echo "fm052 architecture probe must use the Pkg.dependencies() UUID key" >&2
  exit 1
fi

echo "fm052 architecture shell tests passed"
