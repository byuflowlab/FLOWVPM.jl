#!/bin/bash
#SBATCH --job-name=fp052gate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Dependency-gated mature comparison.  The 1080-step acceptance arm is only
# released after this job exits successfully.
source /etc/profile
set -euo pipefail
module load julia/1.11.7-6bmogfl

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
TOLERANCE="${FP052_TOLERANCE:-$FPDIR/data/fm052_campaign_lock/fm052_locked_tolerances.toml}"
CPU_RUN="${FP052_CPU_RUN:-$FPDIR/data/fm052r_cpu_mature}"
GPU_RUN="${FP052_GPU_RUN:-$FPDIR/data/fm052c_gpu_mature}"
OUTDIR="${FP052_GATE_OUT:-$FPDIR/data/fm052_mature_gate}"

test -n "${SLURM_JOB_ID:-}" || { echo "fm052_gate_run.sh must run under Slurm"; exit 1; }
bash "$VPMDIR/scripts/fm052_gate.sh" "$TOLERANCE" "$CPU_RUN" "$GPU_RUN" "$OUTDIR"

long_job=$(cd "$VPMDIR" && sbatch --parsable \
  --dependency="afterok:${SLURM_JOB_ID}" \
  --export=ALL,FP052_STAGES=d scripts/fm052_run.sh)
printf '%s\n' "$long_job" > "$OUTDIR/fm052_long_job_id.txt"
echo "mature gates passed; exact 1080-step acceptance job queued as $long_job"
