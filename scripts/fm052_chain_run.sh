#!/bin/bash
#SBATCH --job-name=fp052chain
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Combined protected H200 arm: stages a b c, then the mature gate inline, then
# the exact 1080-step stage d — one allocation so the H200 queue wait is paid
# once instead of three times. Stage d only runs if the inline gate passes
# (set -e); no alternative-architecture run is authorized by this script.
source /etc/profile
set -euo pipefail
module load cuda julia/1.11.7-6bmogfl

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
TOLERANCE="${FP052_TOLERANCE:-$FPDIR/data/fm052_campaign_lock/fm052_locked_tolerances.toml}"
CPU_RUN="${FP052_CPU_RUN:-$FPDIR/data/fm052r_cpu_mature_pinned}"
GPU_RUN="${FP052_GPU_RUN:-$FPDIR/data/fm052c_gpu_mature}"
OUTDIR="${FP052_GATE_OUT:-$FPDIR/data/fm052_mature_gate}"

test -n "${SLURM_JOB_ID:-}" || { echo "fm052_chain_run.sh must run under Slurm"; exit 1; }
test -s "$TOLERANCE" || { echo "locked tolerance missing: $TOLERANCE"; exit 1; }
test -d "$CPU_RUN" || { echo "canonical CPU mature reference missing: $CPU_RUN"; exit 1; }

echo "fm052 chain: stages a b c -> inline mature gate -> stage d (job $SLURM_JOB_ID)"
FP052_STAGES="a b c" bash "$VPMDIR/scripts/fm052_run.sh"

echo "fm052 chain: stages a b c passed; applying mature gate inline"
bash "$VPMDIR/scripts/fm052_gate.sh" "$TOLERANCE" "$CPU_RUN" "$GPU_RUN" "$OUTDIR"

echo "fm052 chain: mature gate passed; running exact 1080-step acceptance in-allocation"
FP052_STAGES="d" bash "$VPMDIR/scripts/fm052_run.sh"
echo "fm052 chain: complete (a b c + gate + d)"
