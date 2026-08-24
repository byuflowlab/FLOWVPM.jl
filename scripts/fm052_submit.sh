#!/bin/bash
# Task 052 local driver: sync the FLOWPanel (branch fastmultipole), FLOWVPM
# (branch flowpanel), and FastMultipole (branch flowpanel-20260817) working
# trees to the BYU cluster, build a NEW fm052env (FLOWPanel + its registered
# deps; FLOWVPM/FastMultipole path-dev'd), then submit the H200 job
# (scripts/fm052_run.sh). Pattern: fm049_submit.sh (fm023env local-toolkit
# CUDA recipe: compute nodes have no internet, the three CUDA JLLs get
# local=true preferences, with CUDA precompilation deferred to the GPU node).
#   bash scripts/fm052_submit.sh          # from the FLOWVPM.jl repo root
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
FPDIR=FLOWPanel-052
ENVDIR='$HOME/fm052env_cuda63_geoiofree'
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

# GeoIO is no longer a FLOWPanel dependency, so CUDA 6.3 and PrettyTables 3
# resolve in this single environment. Keep Julia 1.11.7 pinned until the known
# Julia 1.12 device-step LLVM crash has passed its separate H200 gate.
# local=true preferences for the three CUDA JLLs (fm023env recipe)
ssh "$REMOTE" 'mkdir -p fm052env_cuda63_geoiofree && cat > fm052env_cuda63_geoiofree/LocalPreferences.toml <<EOF
[CUDA_Compiler_jll]
local = "true"

[CUDA_Driver_jll]
local = "true"

[CUDA_Runtime_jll]
local = "true"
EOF'

# login-node env setup (compute nodes have no internet): develop the two
# unregistered path deps FIRST so FLOWPanel's resolve can see them, then
# develop FLOWPanel, then add CUDA 6.3 and the direct driver dependencies.
# CUDA precompiles on the GPU node against the system toolkit.
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; haskey(Pkg.project().dependencies, \\\"GeoIO\\\") && Pkg.rm(\\\"GeoIO\\\"); Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); Pkg.develop(path=\\\"\$HOME/$FPDIR\\\"); Pkg.add(Pkg.PackageSpec(name=\\\"CUDA\\\", version=\\\"6.3\\\")); Pkg.add([\\\"Test\\\", \\\"Random\\\", \\\"SHA\\\", \\\"Statistics\\\", \\\"StaticArrays\\\", \\\"VSPGeom\\\"]); Pkg.resolve(); Pkg.instantiate()\" \
  && cd $VPMDIR \
  && sbatch scripts/fm052_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
