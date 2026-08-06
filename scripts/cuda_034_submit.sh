#!/bin/bash
# Task 034 local driver: sync the FLOWVPM working tree (branch gpu-full) and
# the FastMultipole working tree (branch matrix-ops) to the BYU cluster into a
# SEPARATE task-034 tree + env (another agent owns ~/FastMultipole-023 and
# ~/fm023env and refreshes them for its own jobs), then submit the H200
# validation job. Pattern: cuda_032_submit.sh + the fm023env local-toolkit
# CUDA recipe (compute nodes have no internet; artifact downloads fail, so the
# three CUDA JLLs get local=true preferences and precompilation is deferred to
# the GPU node).
#   bash scripts/cuda_034_submit.sh          # from the FLOWVPM.jl repo root
# Requires a live ssh master session to `orc`.
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-034
FMDIR=FastMultipole-034          # task-034-owned staged copy of matrix-ops
ENVDIR='$HOME/fm034env'
FMLOCAL=../FastMultipole         # local matrix-ops working tree

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR/MATRIX_OPERATOR_REFACTOR"

# FLOWVPM tree (gpu-full working tree)
rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

# FastMultipole tree (matrix-ops working tree): src + test (the CUDA interface
# preflight) + the MATRIX_OPERATOR_REFACTOR scripts the test includes
rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/test" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"
rsync -az --delete \
    "$FMLOCAL/MATRIX_OPERATOR_REFACTOR/scripts" \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"

# login-node env setup (compute nodes have no internet). Auto-precompile is
# disabled: CUDA precompiles on the GPU node against the system toolkit
# (login-node pkgimages are CPU-target specific and artifact fetches fail).
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.add([\\\"CUDA\\\", \\\"Preferences\\\", \\\"Test\\\", \\\"Random\\\"]); import Preferences; for p in (\\\"CUDA_Runtime_jll\\\", \\\"CUDA_Compiler_jll\\\", \\\"CUDA_Driver_jll\\\"); Preferences.set_preferences!(p, \\\"local\\\" => \\\"true\\\"; force=true); end; Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/cuda_034_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
