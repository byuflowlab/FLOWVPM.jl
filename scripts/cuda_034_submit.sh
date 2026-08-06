#!/bin/bash
# Task 034 local driver (DRAFT — do not run without GPU-queue authorization):
# sync the FLOWVPM working tree (branch gpu-full) and the FastMultipole
# working tree (branch matrix-ops) to the BYU cluster, refresh the shared env,
# and submit the H200 validation job. Pattern: cuda_032_submit.sh.
#   bash scripts/cuda_034_submit.sh          # from the FLOWVPM.jl repo root
# Requires a live ssh master session to `orc`.
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-034
FMDIR=FastMultipole-023          # matrix-ops working tree of record on orc
ENVDIR='$HOME/fm023env'

ssh "$REMOTE" "mkdir -p $VPMDIR"

# FLOWVPM tree (gpu-full working tree)
rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

# FastMultipole matrix-ops src/Project must already be current in $FMDIR
# (the 032/033 sync flow owns that tree); this only asserts it exists.
ssh "$REMOTE" "test -f $FMDIR/src/FastMultipole.jl"

# login-node instantiate (compute nodes have no internet), then submit
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.add(\\\"CUDA\\\"); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/cuda_034_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
