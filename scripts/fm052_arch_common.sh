#!/usr/bin/env bash
# Shared architecture identity, eligibility, and provenance gates for task 052.
# This file is used only by the isolated multi-architecture campaign.

fm052_arch_configure() {
  : "${FP052_ARCH:?FP052_ARCH is required (h200, h100, gh200, b200, or l40s)}"
  case "$FP052_ARCH" in
    # eng also hosts H200 nodes and takes the shorter qos=eng line (2026-08-25)
    h200) FM052_EXPECTED_GRES=h200; FM052_EXPECTED_PARTITION="m13h eng"; FM052_EXPECTED_CPU_ARCH=x86_64; FM052_GPU_NAME_RX='H200' ;;
    h100) FM052_EXPECTED_GRES=h100; FM052_EXPECTED_PARTITION=cs2;  FM052_EXPECTED_CPU_ARCH=x86_64; FM052_GPU_NAME_RX='H100' ;;
    gh200) FM052_EXPECTED_GRES=gh200; FM052_EXPECTED_PARTITION=mgh; FM052_EXPECTED_CPU_ARCH=aarch64; FM052_GPU_NAME_RX='GH200' ;;
    b200) FM052_EXPECTED_GRES=b200; FM052_EXPECTED_PARTITION=cs3; FM052_EXPECTED_CPU_ARCH=x86_64; FM052_GPU_NAME_RX='B200' ;;
    l40s) FM052_EXPECTED_GRES=l40s; FM052_EXPECTED_PARTITION=m13l; FM052_EXPECTED_CPU_ARCH=x86_64; FM052_GPU_NAME_RX='L40S' ;;
    *) echo "invalid canonical FP052_ARCH slug: $FP052_ARCH" >&2; return 64 ;;
  esac
  : "${FP052_GPU_GRES:?FP052_GPU_GRES is required}"
  : "${FP052_PARTITION:?FP052_PARTITION is required}"
  test "$FP052_GPU_GRES" = "$FM052_EXPECTED_GRES" || {
    echo "architecture/GRES mismatch: slug=$FP052_ARCH requires gres=$FM052_EXPECTED_GRES, requested=$FP052_GPU_GRES" >&2
    return 65
  }
  case " $FM052_EXPECTED_PARTITION " in
    *" $FP052_PARTITION "*) ;;
    *)
      echo "architecture/partition mismatch: slug=$FP052_ARCH requires partition in {$FM052_EXPECTED_PARTITION}, requested=$FP052_PARTITION" >&2
      return 66 ;;
  esac
  export FM052_EXPECTED_GRES FM052_EXPECTED_PARTITION FM052_EXPECTED_CPU_ARCH FM052_GPU_NAME_RX
}

fm052_arch_observe() {
  local smi=${FP052_NVIDIA_SMI_BIN:-nvidia-smi}
  local gpu_rows gpu_count
  gpu_rows=$("$smi" --query-gpu=name,memory.total,uuid,compute_cap,driver_version --format=csv,noheader,nounits)
  gpu_count=$(printf '%s\n' "$gpu_rows" | awk 'NF {n++} END {print n+0}')
  test "$gpu_count" -eq 1 || {
    echo "task 052 requires exactly one visible GPU; observed $gpu_count" >&2
    return 67
  }
  IFS=',' read -r FM052_OBS_GPU_NAME FM052_OBS_GPU_VRAM_MIB FM052_OBS_GPU_UUID FM052_OBS_GPU_CC FM052_OBS_CUDA_DRIVER <<<"$gpu_rows"
  FM052_OBS_GPU_NAME=${FM052_OBS_GPU_NAME# }; FM052_OBS_GPU_NAME=${FM052_OBS_GPU_NAME% }
  FM052_OBS_GPU_VRAM_MIB=${FM052_OBS_GPU_VRAM_MIB//[[:space:]]/}
  FM052_OBS_GPU_UUID=${FM052_OBS_GPU_UUID//[[:space:]]/}
  FM052_OBS_GPU_CC=${FM052_OBS_GPU_CC//[[:space:]]/}
  FM052_OBS_CUDA_DRIVER=${FM052_OBS_CUDA_DRIVER//[[:space:]]/}
  FM052_OBS_CPU_ARCH=$(uname -m)
  FM052_OBS_CPU_MODEL=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
  FM052_OBS_NODE=$(hostname)
  FM052_OBS_PARTITION=${SLURM_JOB_PARTITION:-not-slurm}
  export FM052_OBS_GPU_NAME FM052_OBS_GPU_VRAM_MIB FM052_OBS_GPU_UUID FM052_OBS_GPU_CC
  export FM052_OBS_CUDA_DRIVER FM052_OBS_CPU_ARCH FM052_OBS_CPU_MODEL FM052_OBS_NODE FM052_OBS_PARTITION
}

fm052_arch_validate_identity() {
  fm052_arch_configure
  : "${SLURM_JOB_ID:?architecture validation must run inside a Slurm allocation}"
  test "${SLURM_JOB_PARTITION:-}" = "$FP052_PARTITION" || {
    echo "requested/allocated partition mismatch: requested=$FP052_PARTITION allocated=${SLURM_JOB_PARTITION:-missing}" >&2
    return 68
  }
  fm052_arch_observe
  [[ "$FM052_OBS_GPU_NAME" =~ $FM052_GPU_NAME_RX ]] || {
    echo "requested/observed GPU mismatch: slug=$FP052_ARCH gres=$FP052_GPU_GRES observed=$FM052_OBS_GPU_NAME" >&2
    return 69
  }
  test "$FM052_OBS_CPU_ARCH" = "$FM052_EXPECTED_CPU_ARCH" || {
    echo "CPU architecture mismatch: slug=$FP052_ARCH expected=$FM052_EXPECTED_CPU_ARCH observed=$FM052_OBS_CPU_ARCH" >&2
    return 70
  }
}

# Exact production Float64 S plus its two vectors, rounded conservatively up.
FM052_PRODUCTION_S_ALLOCATION_BYTES=10806265000
FM052_POST_UPLOAD_RESERVE_BYTES=$((32 * 1024 * 1024 * 1024))
FM052_EMERGENCY_MARGIN_BYTES=$((4 * 1024 * 1024 * 1024))

fm052_arch_memory_eligibility() {
  local total_bytes required_bytes
  total_bytes=$((FM052_OBS_GPU_VRAM_MIB * 1024 * 1024))
  required_bytes=$((FM052_PRODUCTION_S_ALLOCATION_BYTES + FM052_POST_UPLOAD_RESERVE_BYTES + FM052_EMERGENCY_MARGIN_BYTES))
  if (( total_bytes < required_bytes )); then
    FM052_INELIGIBLE_REASON="startup_vram_${total_bytes}_below_official_requirement_${required_bytes}"
    export FM052_INELIGIBLE_REASON
    echo "official GPU-S preflight ineligible: $FM052_INELIGIBLE_REASON" >&2
    return 2
  fi
}

fm052_toml_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '%s' "$value"
}

fm052_arch_write_stage_manifest() {
  local path=$1 status=$2 reason=$3 stage=$4 run_dir=${5:-} report_dir=${6:-}
  local julia_cmd=${FP052_JULIA_BIN:-julia}
  local julia_version=unavailable cuda_runtime=unavailable manifest_sha=missing provenance_path= source_sha=missing
  if command -v "$julia_cmd" >/dev/null 2>&1 || test -x "$julia_cmd"; then
    julia_version=$("$julia_cmd" --version 2>/dev/null || echo unavailable)
    cuda_runtime=$("$julia_cmd" --project="${FP052_ENV:-.}" -e \
      'try; using CUDA; print(CUDA.runtime_version()); catch; print("unavailable"); end' 2>/dev/null || echo unavailable)
  fi
  test ! -f "${FP052_ENV:-}/Manifest.toml" || manifest_sha=$(sha256sum "${FP052_ENV}/Manifest.toml" | awk '{print $1}')
  if test -n "$run_dir" && test -d "$run_dir"; then
    provenance_path=$(find "$run_dir" -maxdepth 1 -name '*_provenance.toml' -print | head -1)
  elif test -n "$report_dir" && test -d "$report_dir"; then
    provenance_path=$(find "$report_dir" -maxdepth 1 -name '*_provenance.toml' -print | head -1)
  fi
  if test -n "$provenance_path"; then
    source_sha=$(sed -n 's/^source_checksums_sha256 = "\(.*\)"/\1/p' "$provenance_path")
  fi
  mkdir -p "$(dirname "$path")"
  cat >"$path" <<EOF
architecture = "$(fm052_toml_escape "$FP052_ARCH")"
stage = "$(fm052_toml_escape "$stage")"
status = "$(fm052_toml_escape "$status")"
reason = "$(fm052_toml_escape "$reason")"
requested_gres = "$(fm052_toml_escape "$FP052_GPU_GRES")"
requested_partition = "$(fm052_toml_escape "$FP052_PARTITION")"
observed_gpu_model = "$(fm052_toml_escape "${FM052_OBS_GPU_NAME:-unavailable}")"
observed_gpu_vram_mib = ${FM052_OBS_GPU_VRAM_MIB:--1}
observed_gpu_uuid = "$(fm052_toml_escape "${FM052_OBS_GPU_UUID:-unavailable}")"
observed_compute_capability = "$(fm052_toml_escape "${FM052_OBS_GPU_CC:-unavailable}")"
cuda_driver = "$(fm052_toml_escape "${FM052_OBS_CUDA_DRIVER:-unavailable}")"
cuda_runtime = "$(fm052_toml_escape "$cuda_runtime")"
julia_version = "$(fm052_toml_escape "$julia_version")"
node = "$(fm052_toml_escape "${FM052_OBS_NODE:-${SLURM_JOB_NODELIST:-unavailable}}")"
partition = "$(fm052_toml_escape "${FM052_OBS_PARTITION:-${SLURM_JOB_PARTITION:-unavailable}}")"
cpu_architecture = "$(fm052_toml_escape "${FM052_OBS_CPU_ARCH:-unavailable}")"
cpu_model = "$(fm052_toml_escape "${FM052_OBS_CPU_MODEL:-unavailable}")"
job_id = "$(fm052_toml_escape "${SLURM_JOB_ID:-local}")"
job_name = "$(fm052_toml_escape "${SLURM_JOB_NAME:-local}")"
slurm_output_pattern = "$(fm052_toml_escape "${FP052_SLURM_OUTPUT_PATTERN:-unavailable}")"
run_dir = "$(fm052_toml_escape "$run_dir")"
report_dir = "$(fm052_toml_escape "$report_dir")"
provenance_path = "$(fm052_toml_escape "$provenance_path")"
source_checksums_sha256 = "$(fm052_toml_escape "$source_sha")"
package_manifest_sha256 = "$(fm052_toml_escape "$manifest_sha")"
created_utc = "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
}

fm052_science_fingerprint() {
  local checksum_file=$1
  awk '
    {
      hash=$1; path=$2
      if (path ~ /\/FLOWPanel-[^/]*\//) sub(/^.*\/FLOWPanel-[^/]*\//, "FLOWPanel/", path)
      else if (path ~ /\/FLOWVPM-[^/]*\//) sub(/^.*\/FLOWVPM-[^/]*\//, "FLOWVPM/", path)
      else if (path ~ /\/FastMultipole-[^/]*\//) sub(/^.*\/FastMultipole-[^/]*\//, "FastMultipole/", path)
      else next
      if (path ~ /^(FLOWPanel|FLOWVPM|FastMultipole)\/src\// ||
          path ~ /^(FLOWPanel|FLOWVPM|FastMultipole)\/Project.toml$/ ||
          path == "FLOWPanel/examples/rotor_hover_pressure_comparison.jl") print hash, path
    }' "$checksum_file" | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

fm052_arch_reference_gate() {
  local cpu_provenance=$1 candidate_provenance=$2 out=$3
  local cpu_sources candidate_sources cpu_science candidate_science
  cpu_sources="$(dirname "$cpu_provenance")/$(sed -n 's/^source_checksums_file = "\(.*\)"/\1/p' "$cpu_provenance")"
  candidate_sources="$(dirname "$candidate_provenance")/$(sed -n 's/^source_checksums_file = "\(.*\)"/\1/p' "$candidate_provenance")"
  test -s "$cpu_sources" && test -s "$candidate_sources" || { echo "missing source checksum list" >&2; return 71; }
  cpu_science=$(fm052_science_fingerprint "$cpu_sources")
  candidate_science=$(fm052_science_fingerprint "$candidate_sources")
  test "$cpu_science" = "$candidate_science" || {
    echo "canonical CPU science-source fingerprint mismatch: cpu=$cpu_science candidate=$candidate_science" >&2
    return 72
  }
  local key cpu_value candidate_value
  for key in checkpoint_checksums_sha256; do
    cpu_value=$(sed -n "s/^$key = \"\(.*\)\"/\1/p" "$cpu_provenance")
    candidate_value=$(sed -n "s/^$key = \"\(.*\)\"/\1/p" "$candidate_provenance")
    test -n "$cpu_value" && test "$cpu_value" = "$candidate_value" || {
      echo "canonical CPU provenance mismatch for $key: cpu=$cpu_value candidate=$candidate_value" >&2
      return 73
    }
  done
  mkdir -p "$(dirname "$out")"
  cat >"$out" <<EOF
# fm052 canonical CPU reference provenance gate

- CPU provenance: \`$cpu_provenance\`
- Candidate provenance: \`$candidate_provenance\`
- Science-source fingerprint: \`$cpu_science\`
- Checkpoint checksum: matched
- Package manifest equivalence: checked separately with platform-specific paths ignored
- Verdict: **PASS**
EOF
}
