#!/bin/bash
# Task 048 local driver: sync the unified FLOWVPM (branch flowpanel) and
# FastMultipole (branch flowpanel-20260817) working trees to the BYU cluster
# into the task-046/048 trees + a task-048 env, then submit the H200
# validation job. Pattern: cuda_034_submit.sh + the fm023env local-toolkit
# CUDA recipe (compute nodes have no internet; artifact downloads fail, so the
# three CUDA JLLs get local=true preferences and precompilation is deferred to
# the GPU node).
#   bash scripts/cuda_048_submit.sh          # from the FLOWVPM.jl repo root
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
ENVDIR='$HOME/fm048env'
FMLOCAL=../FastMultipole

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR"

rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/test" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"

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
# the GPU node against the system toolkit.
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.add([\\\"CUDA\\\", \\\"Test\\\", \\\"Random\\\", \\\"SHA\\\", \\\"Statistics\\\", \\\"StaticArrays\\\"]); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/cuda_048_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
