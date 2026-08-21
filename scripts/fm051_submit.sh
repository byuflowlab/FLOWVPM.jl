#!/bin/bash
# Task 051 stage 1 local driver: sync the unified FLOWVPM (branch flowpanel)
# and FastMultipole (branch flowpanel-20260817) working trees plus the p018
# step-710 snapshot dump to the BYU cluster, then submit the H200 benchmark
# job (scripts/fm051_run.sh). Pattern: fm049_submit.sh (fm023env
# local-toolkit CUDA recipe: compute nodes have no internet, the three CUDA
# JLLs get local=true preferences, precompilation deferred to the GPU node).
# Reuses the existing fm048env (FLOWVPM + FastMultipole dev'd).
#   bash scripts/fm051_submit.sh          # from the FLOWVPM.jl repo root
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
ENVDIR='$HOME/fm048env'
FMLOCAL=../FastMultipole
BINLOCAL=../FastMultipole/MATRIX_OPERATOR_REFACTOR/data/rotor_field_gpu_verification/p018_710_particles.bin

[ -f "$BINLOCAL" ] || { echo "snapshot dump missing: $BINLOCAL (run fm049_extract_snapshot.jl first)"; exit 1; }

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR $VPMDIR/data/fm049"

rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/test" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"

rsync -az "$BINLOCAL" \
    "$(dirname "$BINLOCAL")/manifest.csv" \
    "$REMOTE:$VPMDIR/data/fm049/"

# local=true preferences for the three CUDA JLLs (fm023env recipe)
ssh "$REMOTE" 'mkdir -p fm048env && cat > fm048env/LocalPreferences.toml <<EOF
[CUDA_Compiler_jll]
local = "true"

[CUDA_Driver_jll]
local = "true"

[CUDA_Runtime_jll]
local = "true"
EOF'

# login-node env setup (compute nodes have no internet); CUDA precompiles on
# the GPU node against the system toolkit. Env is the standing fm048env; the
# develop/add calls are idempotent.
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.add([\\\"CUDA\\\", \\\"Test\\\", \\\"Random\\\", \\\"SHA\\\", \\\"Statistics\\\", \\\"StaticArrays\\\"]); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/fm051_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
