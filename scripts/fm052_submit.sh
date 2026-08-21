#!/bin/bash
# Task 052 local driver: sync the FLOWPanel (branch fastmultipole), FLOWVPM
# (branch flowpanel), and FastMultipole (branch flowpanel-20260817) working
# trees to the BYU cluster, build a NEW fm052env (FLOWPanel + its registered
# deps; FLOWVPM/FastMultipole path-dev'd), then submit the H200 job
# (scripts/fm052_run.sh). Pattern: fm049_submit.sh (fm023env local-toolkit
# CUDA recipe: compute nodes have no internet, the three CUDA JLLs get
# local=true preferences, precompilation deferred to the GPU node; heavy
# FLOWPanel deps (GeoIO/VSPGeom/Xfoil artifacts) download during the
# login-node instantiate).
#   bash scripts/fm052_submit.sh          # from the FLOWVPM.jl repo root
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
FPDIR=FLOWPanel-052
ENVDIR='$HOME/fm052env'
FMLOCAL=../FastMultipole
FPLOCAL=../FLOWPanel.jl

# mesh files the 018 driver needs on the cluster (stage b: 40_40 hard-coded
# TE seeds; stage c: 45_185_ct4, auto-detected seeds re-load the same file)
MESHES=(dji9443_new_40_40.msh dji9443_20260725_45_185_capped_captess4.msh)

for m in "${MESHES[@]}"; do
  [ -f "$FPLOCAL/examples/data/$m" ] || { echo "mesh missing: $FPLOCAL/examples/data/$m"; exit 1; }
done

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR $FPDIR/examples/data"

rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/test" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"

rsync -az --delete --exclude .git \
    "$FPLOCAL/src" "$FPLOCAL/test" "$FPLOCAL/Project.toml" \
    "$REMOTE:$FPDIR/"

# examples: scripts only (examples/data is 178 MB locally; sync just the
# needed meshes). No --delete so re-runs never clobber cluster-side outputs.
rsync -az "$FPLOCAL"/examples/*.jl "$REMOTE:$FPDIR/examples/"
for m in "${MESHES[@]}"; do
  rsync -az "$FPLOCAL/examples/data/$m" "$REMOTE:$FPDIR/examples/data/"
done

# CUDA deliberately NOT in fm052env: CUDA >=6.2 pulls CUDATools->PrettyTables 3,
# unsatisfiable against FLOWPanel's geo stack (PrettyTables 2.x). CUDA is
# provided at RUN time by environment stacking: JULIA_LOAD_PATH appends the
# validated fm048env (CUDA 6.3 + local-toolkit JLL prefs) after fm052env —
# login-node load test confirmed PrettyTables 2.4 + CUDA/CUDATools coexist.
# local=true preferences for the three CUDA JLLs (fm023env recipe)
ssh "$REMOTE" 'mkdir -p fm052env && cat > fm052env/LocalPreferences.toml <<EOF
[CUDA_Compiler_jll]
local = "true"

[CUDA_Driver_jll]
local = "true"

[CUDA_Runtime_jll]
local = "true"
EOF'

# login-node env setup (compute nodes have no internet): develop the two
# unregistered path deps FIRST so FLOWPanel's resolve can see them, then
# develop FLOWPanel (its registered deps — GeoIO, VSPGeom, CCBlade, Xfoil,
# PythonPlot, ... — resolve+download here), then the extras the drivers use.
# CUDA precompiles on the GPU node against the system toolkit.
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$FPDIR\\\"); Pkg.add([\\\"Test\\\", \\\"Random\\\", \\\"SHA\\\", \\\"Statistics\\\", \\\"StaticArrays\\\"]); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/fm052_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
