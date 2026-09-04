#!/bin/bash
#SBATCH --job-name=vpm049
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 049 H200 rotor-field verification job. Pattern: cuda_048_run.sh
# (julia pinned to 1.11.7: 1.12.6 segfaults JIT-compiling the device step,
# job 13058191). Runs the corrected driver on variable-count p018 snapshots
# 710:719: references, full matrix, RK3, and true same-job residency A/B.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
export FM049_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${VPM049_DIR:-$HOME/FLOWVPM-046}"
ENVDIR="${VPM049_ENV:-$HOME/fm048env}"
BINDIR="${VPM049_BINDIR:-$HOME/FLOWVPM-046/data/fm049}"
OUTDIR="${VPM049_OUTDIR:-$HOME/FLOWVPM-046/data/fm049/results-$SLURM_JOB_ID}"
export FM049_MANIFEST="$BINDIR/manifest.csv"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=8

cd "$WORKDIR"
echo "=== fm049 rotor field GPU verification ==="
mkdir -p "$OUTDIR"
cd "$OUTDIR"
test -f "$ENVDIR/Manifest.toml"
test -f "$FM049_MANIFEST"
find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/scripts" -type f -print0 | \
  sort -z | xargs -0 sha256sum > synced_flowvpm_source_sha256.txt
find "$HOME/FastMultipole-046/src" -type f -print0 | \
  sort -z | xargs -0 sha256sum > synced_fastmultipole_source_sha256.txt
cp "$WORKDIR/data/fm049/submission_provenance.txt" .
hash_artifacts() {
  files=()
  for f in fm049_results.csv fm049_budget.csv fm049_report.md fm049_raw.log \
      synced_flowvpm_source_sha256.txt synced_fastmultipole_source_sha256.txt \
      submission_provenance.txt; do
    [ -f "$f" ] && files+=("$f")
  done
  [ "${#files[@]}" -eq 0 ] || sha256sum "${files[@]}" > artifact_sha256.txt
}
trap hash_artifacts EXIT
julia --project="$ENVDIR" "$WORKDIR/scripts/fm049_rotor_verify.jl" \
  "$BINDIR"/p018_{710..719}_particles.bin 2>&1 | tee fm049_raw.log

echo "vpm049 job complete"
