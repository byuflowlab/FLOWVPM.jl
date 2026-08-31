#!/bin/bash
# Task 052 local driver: sync the FLOWPanel (branch fastmultipole), FLOWVPM
# (branch flowpanel), and FastMultipole (branch flowpanel-20260817) working
# trees to the BYU cluster, build a NEW fm052env (FLOWPanel + its registered
# deps; FLOWVPM/FastMultipole path-dev'd), lock the fixed campaign scatter,
# and submit the complete dependency-gated CPU/GPU campaign. Pattern:
# fm049_submit.sh (fm023env local-toolkit
# CUDA recipe: compute nodes have no internet, the three CUDA JLLs get
# local=true preferences, with CUDA precompilation deferred to the GPU node).
#   bash scripts/fm052_submit.sh          # from the FLOWVPM.jl repo root
# This is the only entry point: it computes and records the locked tolerance,
# submits Stage A/B/C and the independent CPU arm, schedules their correctness
# gate, and lets that gate schedule Stage D only when everything passes.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VPMLOCAL=$(cd -- "$SCRIPT_DIR/.." && pwd)
FMLOCAL=$(cd -- "$VPMLOCAL/../FastMultipole" && pwd)
FPLOCAL=$(cd -- "$VPMLOCAL/../FLOWPanel.jl" && pwd)
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
FPDIR=FLOWPanel-052
ENVDIR='$HOME/fm052env_cuda63_geoiofree'

# mesh files the 018 driver needs on the cluster (stage b: 40_40 hard-coded
# TE seeds; stage c: 45_185_ct4, auto-detected seeds re-load the same file)
MESHES=(dji9443_new_40_40.msh dji9443_20260725_45_185_capped_captess4.msh)

for m in "${MESHES[@]}"; do
  [ -f "$FPLOCAL/examples/data/$m" ] || { echo "mesh missing: $FPLOCAL/examples/data/$m"; exit 1; }
done

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR $FPDIR/examples/data"

rsync -az --delete "$VPMLOCAL/src/" "$REMOTE:$VPMDIR/src/"
rsync -az --delete "$VPMLOCAL/ext/" "$REMOTE:$VPMDIR/ext/"
rsync -az --delete "$VPMLOCAL/test/" "$REMOTE:$VPMDIR/test/"
rsync -az --delete "$VPMLOCAL/scripts/" "$REMOTE:$VPMDIR/scripts/"
rsync -az "$VPMLOCAL/Project.toml" "$REMOTE:$VPMDIR/Project.toml"

rsync -az --delete "$FMLOCAL/src/" "$REMOTE:$FMDIR/src/"
rsync -az --delete "$FMLOCAL/test/" "$REMOTE:$FMDIR/test/"
rsync -az "$FMLOCAL/Project.toml" "$REMOTE:$FMDIR/Project.toml"

rsync -az --delete "$FPLOCAL/src/" "$REMOTE:$FPDIR/src/"
rsync -az --delete "$FPLOCAL/test/" "$REMOTE:$FPDIR/test/"
rsync -az "$FPLOCAL/Project.toml" "$REMOTE:$FPDIR/Project.toml"

# Synchronize the example code without transferring the 178 MB data directory;
# excluded data is preserved, and the two exact meshes are copied below.
rsync -az --delete --exclude data "$FPLOCAL/examples/" "$REMOTE:$FPDIR/examples/"
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
  '"

# Fixed campaign M2 scatter: same-configuration p018_L1 cold continuation and
# warm continuation, revolutions 22--31 = raw simulation steps 792--1151.
# The protected inputs live in the production tree and are never modified.
result=$(ssh "$REMOTE" "bash -lc '
  source /etc/profile
  set -euo pipefail
  module load julia/1.11.7-6bmogfl
  vpm=\$HOME/$VPMDIR
  fp=\$HOME/$FPDIR
  envdir=$ENVDIR
  cold=/home/rander39/projects/FLOWPanel.jl/data/p018_L1_s2
  warm=/home/rander39/projects/FLOWPanel.jl/data/p018_L1_warm
  lockdir=\$fp/data/fm052_campaign_lock
  test -f \$cold/p018_L1_s2_CT_vs_rev.csv
  test -f \$cold/monitors/p018_L1_s2_monitor03_bound_circulation_system1.csv
  test -f \$warm/p018_L1_warm_CT_vs_rev.csv
  test -f \$warm/monitors/p018_L1_warm_monitor03_bound_circulation_system1.csv
  mkdir -p \$lockdir
  julia --project=\$envdir \$vpm/scripts/fm052_compare.jl lock \
    \$cold \$warm 792 1151 \$lockdir > \$lockdir/fm052_lock.log
  tolerance=\$lockdir/fm052_locked_tolerances.toml
  test -s \$tolerance
  cd \$vpm
  gpu_job=\$(sbatch --parsable --export=ALL,FP052_STAGES=\"a b c\" scripts/fm052_run.sh)
  cpu_job=\$(sbatch --parsable scripts/fm052_cpu_run.sh)
  gate_job=\$(sbatch --parsable --dependency=afterok:\$gpu_job:\$cpu_job \
    --export=ALL,FP052_TOLERANCE=\$tolerance scripts/fm052_gate_run.sh)
  manifest=\$lockdir/fm052_submission.toml
  {
    printf \"gpu_job = \\\"%s\\\"\\n\" \$gpu_job
    printf \"cpu_job = \\\"%s\\\"\\n\" \$cpu_job
    printf \"gate_job = \\\"%s\\\"\\n\" \$gate_job
    printf \"long_job_file = \\\"%s\\\"\\n\" \$fp/data/fm052_mature_gate/fm052_long_job_id.txt
    printf \"tolerance = \\\"%s\\\"\\n\" \$tolerance
    printf \"lock_window = \\\"792:1151\\\"\\n\"
  } > \$manifest
  printf \"GPU=%s CPU=%s GATE=%s MANIFEST=%s\" \$gpu_job \$cpu_job \$gate_job \$manifest
'")

echo "$result"
echo "The gate job will submit the exact 1080-step H200 run only after all mature gates pass."
