#!/bin/bash
# Local driver for the KA-vs-CUDA H200 gate: sync the metal-testing working
# tree AND FastMultipole (this branch's Project.toml requires FastMultipole
# 2.2.0+, which is NOT on the General registry -- registry tops out at
# 2.0.4, same unreleased-version situation task 034/048 hit on gpu-full; here
# it's `../FastMultipole` on branch flowpanel-20260817, confirmed via
# test/metal_env/Manifest.toml) to the BYU cluster into their own trees + env
# (separate from the task-034/048 trees, owned by other in-flight work), then
# submit scripts/ka_cuda_bench_run.sh. Pattern: cuda_034_submit.sh.
#   bash scripts/ka_cuda_bench_submit.sh    # from the FLOWVPM.jl repo root, metal-testing branch
# Requires a live ssh master session to `orc`.
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-kabench
FMDIR=FastMultipole-kabench
ENVDIR='$HOME/fm_kabench_env'
FMLOCAL=../FastMultipole

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR"

rsync -az --delete --exclude .git \
    src ext scripts Project.toml \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"

# src/error.jl does `include("../test/evaluate_multipole.jl")` at package
# load time -- it's a real dependency, not test-only code, so it must be
# synced too (self-contained, no further includes/deps of its own).
ssh "$REMOTE" "mkdir -p $FMDIR/test"
rsync -az "$FMLOCAL/test/evaluate_multipole.jl" "$REMOTE:$FMDIR/test/"

# local=true preferences for the three CUDA JLLs (compute nodes have no
# internet; artifact downloads fail -- same recipe as fm034env/cuda_034_submit.sh)
ssh "$REMOTE" "mkdir -p fm_kabench_env && cat > fm_kabench_env/LocalPreferences.toml <<EOF
[CUDA_Compiler_jll]
local = \"true\"

[CUDA_Driver_jll]
local = \"true\"

[CUDA_Runtime_jll]
local = \"true\"
EOF"

# login-node env setup (compute nodes have no internet). FLOWVPM + FastMultipole
# dev'd from the synced trees; CUDA + KernelAbstractions from the registry.
# Auto-precompile disabled: CUDA precompiles on the GPU node against the
# system toolkit (login-node pkgimages are CPU-target specific and artifact
# fetches fail there), same as cuda_034_submit.sh.
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.add([\\\"CUDA\\\", \\\"KernelAbstractions\\\"]); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/ka_cuda_bench_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
echo "Results land in ~/$VPMDIR/ka_cuda_bench_<jobid>.{csv,log,provenance} on $REMOTE."
