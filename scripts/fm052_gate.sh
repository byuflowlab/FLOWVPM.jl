#!/bin/bash
# Apply a pre-locked same-configuration campaign-scatter ceiling to mature arms.
set -euo pipefail
TOLERANCE=${1:?usage: fm052_gate.sh TOLERANCE CPU_RUN GPU_RUN OUTDIR}
CPU_RUN=${2:?}
GPU_RUN=${3:?}
OUTDIR=${4:?}
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
ENVDIR="${FP052_ENV:-$HOME/fm052env_cuda63_geoiofree}"

mkdir -p "$OUTDIR"
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" gate \
  "$CPU_RUN" "$GPU_RUN" "$TOLERANCE" "$OUTDIR"
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
  "$GPU_RUN" "${GPU_RUN}.log" "$OUTDIR"
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" memory-gate \
  "$GPU_RUN" "$OUTDIR" 16

find "$OUTDIR" -maxdepth 1 -type f ! -name sha256_outputs.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTDIR/sha256_outputs.txt"
echo "fm052 mature-continuation gates passed against $TOLERANCE"
