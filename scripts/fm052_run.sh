#!/bin/bash
#SBATCH --job-name=fp052
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Task 052 stage A: GPU arm of the FLOWPanel 018 rotor driver on one H200.
# Pattern: fm049_run.sh (julia pinned to 1.11.7 — 1.12.6 segfaults JIT-
# compiling the device step, job 13058191; FASTMULTIPOLE_FORCE_CUDA_LOAD=1
# because compute nodes hide the device nodes from the preflight).
#
# Stages (select with FP052_STAGES, default "a b c"):
#   a  seam unit smoke: examples/fm051_pass_parity.jl in host mode (datum)
#      then FM051_MODE=cuda (device direct_rectangular! parity gates)
#   b  reduced driver comparison on the coarse 40_40 mesh: CPU arm, then
#      GPU arm (VPM_ARRAYTYPE=cuarray FLOWPANEL_GPU_INFLUENCE=cuda), then
#      CT / Gamma(r/R) / per-step-wall comparison (scripts/fm052_compare.jl)
#   c  production-shape GPU measurement: p018_L1_ov3 knobs (45_185_ct4,
#      OVERLAP=3 P_PER_STEP=14 MERGE_R_FACTOR=0.0052 NT=36), 1 rev, GPU arm
#      only, per-pass timers on; per-step trajectory + 30-rev extrapolation
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-046}"
ENVDIR="${FP052_ENV:-$HOME/fm052env}"
STAGES="${FP052_STAGES:-a b c}"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=32
export BLAS_NUM_THREADS=32

cd "$FPDIR"
mkdir -p data

# knobs shared by both stage-b arms (coarse mesh, compressed freestream
# schedule => 36 steps total; BERNOULLI_ONLY skips the CG pressure solves so
# the CPU arm lands in the ~10-20 min bracket on 32 cores; SAVE_VTK=true is
# REQUIRED — monitor CSVs are only written when a save path exists)
STAGEB_COMMON="RHPC_MESH=40_40 NT=36 NREVS=1 \
  FREESTREAM_RAMP_REVS=0.3 FREESTREAM_HOLD_REVS=0.2 \
  FREESTREAM_WITHDRAW_REVS=0.3 SETTLE_REVS=0.2 \
  P_PER_STEP=6 OVERLAP=3.0 BERNOULLI_ONLY=true SAVE_VTK=true"

# production p018_L1_ov3 shape, compressed to 1 rev (36 steps) for the cost
# measurement (the pulse schedule is shrunk so required_revs == NREVS; the
# wake is younger than production maturity — the report extrapolates from
# the per-step trajectory vs n_particles)
STAGEC_COMMON="RHPC_MESH=45_185_ct4 NT=36 NREVS=${FP052C_NREVS:-1} \
  FREESTREAM_RAMP_REVS=0.5 FREESTREAM_HOLD_REVS=0.0 \
  FREESTREAM_WITHDRAW_REVS=0.5 SETTLE_REVS=0.0 \
  P_PER_STEP=14 OVERLAP=3.0 MERGE_R_FACTOR=0.0052 SAVE_VTK=true"

GPU_ARM="VPM_ARRAYTYPE=cuarray FLOWPANEL_GPU_INFLUENCE=cuda FLOWPANEL_GPU_TIMERS=1"

for stage in $STAGES; do
case "$stage" in

a)
  echo "=== STAGE A: fm051 pass parity, host datum ==="
  julia --project="$ENVDIR" --threads=4 examples/fm051_pass_parity.jl
  echo "=== STAGE A: fm051 pass parity, CUDA seam ==="
  FM051_MODE=cuda julia --project="$ENVDIR" --threads=4 examples/fm051_pass_parity.jl
  ;;

b)
  echo "=== STAGE B: reduced driver, CPU arm ==="
  env $STAGEB_COMMON RUN_NAME=fm052b_cpu \
    julia --project="$ENVDIR" --threads=32 examples/rotor_hover_pressure_comparison.jl \
    2>&1 | tee data/fm052b_cpu.log
  echo "=== STAGE B: reduced driver, GPU arm ==="
  env $STAGEB_COMMON $GPU_ARM RUN_NAME=fm052b_gpu \
    julia --project="$ENVDIR" --threads=32 examples/rotor_hover_pressure_comparison.jl \
    2>&1 | tee data/fm052b_gpu.log
  echo "=== STAGE B: comparison ==="
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" compare \
    data/fm052b_cpu data/fm052b_gpu
  ;;

c)
  echo "=== STAGE C: production-shape GPU arm ==="
  env $STAGEC_COMMON $GPU_ARM RUN_NAME=fm052c_gpu \
    julia --project="$ENVDIR" --threads=32 examples/rotor_hover_pressure_comparison.jl \
    2>&1 | tee data/fm052c_gpu.log
  echo "=== STAGE C: report ==="
  julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
    data/fm052c_gpu data/fm052c_gpu.log
  ;;

*) echo "unknown stage '$stage'"; exit 1 ;;
esac
done

echo "fp052 job complete"
