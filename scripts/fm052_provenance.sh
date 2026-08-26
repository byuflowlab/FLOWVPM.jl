#!/bin/bash
set -euo pipefail

OUT=${1:?usage: fm052_provenance.sh OUTPUT_TOML CHECKPOINT_ROOT}
CHECKPOINT=${2:?usage: fm052_provenance.sh OUTPUT_TOML CHECKPOINT_ROOT}
FPDIR=${FP052_DIR:-$HOME/FLOWPanel-052}
VPMDIR=${FP052_VPMDIR:-$HOME/FLOWVPM-046}
FMDIR=${FP052_FMDIR:-$HOME/FastMultipole-046}
ENVDIR=${FP052_ENV:-$HOME/fm052env_cuda63_geoiofree}
JULIA_CMD=${FP052_JULIA_BIN:-julia}
SOURCE_LIST="${OUT%.toml}_sources.sha256"
CHECKPOINT_LIST="${OUT%.toml}_checkpoint.sha256"

mkdir -p "$(dirname "$OUT")"
find "$FPDIR/src" "$FPDIR/examples" "$FPDIR/test" "$FPDIR/Project.toml" \
  "$VPMDIR/src" "$VPMDIR/scripts" "$VPMDIR/test" "$VPMDIR/Project.toml" \
  "$FMDIR/src" "$FMDIR/test" "$FMDIR/Project.toml" \
  -type f \( -name '*.jl' -o -name '*.sh' -o -name 'Project.toml' \) -print0 \
  | sort -z | xargs -0 sha256sum > "$SOURCE_LIST"
find "$CHECKPOINT" -type f \( -name "*${FM052_RESTART_STEP:-719}*" \
  -o -name '*.pvd' -o -name '*metadata.toml' \) -print0 \
  | sort -z | xargs -0 sha256sum > "$CHECKPOINT_LIST"

hash_or_missing() {
  if test -f "$1"; then sha256sum "$1" | awk '{print $1}'; else echo missing; fi
}

project_hashes=$(for project in "$FPDIR/Project.toml" "$VPMDIR/Project.toml" "$FMDIR/Project.toml"; do
  hash_or_missing "$project"
done | paste -sd, -)

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)
gpu_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo -1)
gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)
cpu_model=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1 || echo unavailable)
cuda_runtime=$("$JULIA_CMD" --project="$ENVDIR" -e 'try; using CUDA; print(CUDA.runtime_version()); catch; print("unavailable"); end' 2>/dev/null || echo unavailable)

MESH="$FPDIR/examples/data/dji9443_20260725_45_185_capped_captess4.msh"
cat > "$OUT" <<EOF
job_id = "${SLURM_JOB_ID:-local}"
job_name = "${SLURM_JOB_NAME:-local}"
node_list = "${SLURM_JOB_NODELIST:-local}"
cpus_per_task = "${SLURM_CPUS_PER_TASK:-unknown}"
memory_per_node = "${SLURM_MEM_PER_NODE:-unknown}"
gpu_resources = "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-none}}"
requested_architecture = "${FP052_ARCH:-canonical-h200}"
requested_gres = "${FP052_GPU_GRES:-h200}"
requested_partition = "${FP052_PARTITION:-${SLURM_JOB_PARTITION:-unknown}}"
partition = "${SLURM_JOB_PARTITION:-unknown}"
slurm_time_limit = "${SLURM_TIMELIMIT:-unknown}"
julia_version = "$("$JULIA_CMD" --version 2>&1)"
cuda_driver = "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)"
cuda_gpu = "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)"
cuda_gpu_uuid = "$gpu_uuid"
cuda_gpu_vram_mib = "$gpu_vram"
cuda_compute_capability = "$gpu_cc"
cuda_runtime = "$cuda_runtime"
cpu_architecture = "$(uname -m)"
cpu_model = "$cpu_model"
source_checksums_file = "$(basename "$SOURCE_LIST")"
source_checksums_sha256 = "$(hash_or_missing "$SOURCE_LIST")"
checkpoint_checksums_file = "$(basename "$CHECKPOINT_LIST")"
checkpoint_checksums_sha256 = "$(hash_or_missing "$CHECKPOINT_LIST")"
manifest_sha256 = "$(hash_or_missing "$ENVDIR/Manifest.toml")"
preferences_sha256 = "$(hash_or_missing "$ENVDIR/LocalPreferences.toml")"
project_sha256 = "$project_hashes"
mesh_path = "$MESH"
mesh_sha256 = "$(hash_or_missing "$MESH")"
checkpoint_path = "$CHECKPOINT"
restart_step = ${FM052_RESTART_STEP:-719}
EOF

echo "wrote provenance manifest: $OUT"
