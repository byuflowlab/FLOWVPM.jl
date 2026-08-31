#!/bin/bash
# Task 049 local driver: sync the unified FLOWVPM (branch flowpanel) and
# FastMultipole (branch flowpanel-20260817) working trees plus the p018
# step-710:719 snapshot dumps to the BYU cluster, then submit the H200
# verification job (scripts/fm049_run.sh). Pattern: cuda_048_submit.sh
# (fm023env local-toolkit CUDA recipe: compute nodes have no internet, the
# three CUDA JLLs get local=true preferences, precompilation deferred to the
# GPU node). Reuses the existing fm048env (FLOWVPM + FastMultipole dev'd).
#   bash scripts/fm049_submit.sh          # from the FLOWVPM.jl repo root
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-046
FMDIR=FastMultipole-046
ENVDIR='$HOME/fm048env'
FMLOCAL=../FastMultipole
BINDIRLOCAL=../FastMultipole/MATRIX_OPERATOR_REFACTOR/data/rotor_field_gpu_verification
MANIFEST="$BINDIRLOCAL/manifest.csv"

[ -f "$MANIFEST" ] || { echo "snapshot manifest missing: $MANIFEST"; exit 1; }
[ "$(sed -n '1p' "$MANIFEST")" = "file,step,np,sha256" ] || { echo "bad manifest header"; exit 1; }
[ "$(wc -l < "$MANIFEST" | tr -d ' ')" = 11 ] || { echo "manifest must contain exactly ten snapshots"; exit 1; }
awk -F, '
    NR == 1 { next }
    NF != 4 { exit 1 }
    $1 !~ /^p018_[0-9]+_particles\.bin$/ { exit 1 }
    seen_file[$1]++ { exit 1 }
    seen_step[$2]++ { exit 1 }
    END { if (NR != 11) exit 1 }
' "$MANIFEST" || { echo "manifest rows must have four fields and unique filenames/steps"; exit 1; }

for step in {710..719}; do
    file="$BINDIRLOCAL/p018_${step}_particles.bin"
    [ -f "$file" ] || { echo "snapshot dump missing: $file (run fm049_extract_snapshot.jl first)"; exit 1; }
    row="$(awk -F, -v f="$(basename "$file")" '$1==f {print $2 "," $3 "," $4}' "$MANIFEST")"
    [ -n "$row" ] || { echo "manifest row missing for $file"; exit 1; }
    IFS=, read -r manifest_step manifest_np manifest_hash <<< "$row"
    [ "$manifest_step" = "$step" ] && [ "$manifest_np" -gt 0 ] || { echo "bad manifest row: $row"; exit 1; }
    read -r binary_rows binary_np <<< "$(od -An -N16 -tu8 "$file")"
    [ "$binary_rows" = 46 ] && [ "$binary_np" = "$manifest_np" ] || {
        echo "binary header/manifest mismatch: $file rows=$binary_rows np=$binary_np manifest_np=$manifest_np"; exit 1;
    }
    expected_size=$((16 + 46 * 8 * binary_np))
    actual_size="$(stat -f '%z' "$file")"
    [ "$actual_size" = "$expected_size" ] || { echo "binary size mismatch: $file"; exit 1; }
    actual_hash="$(shasum -a 256 "$file" | awk '{print $1}')"
    [ "$actual_hash" = "$manifest_hash" ] || { echo "snapshot hash mismatch: $file"; exit 1; }
done

VPM_HEAD="$(git rev-parse HEAD)"
VPM_DIFF_SHA="$(git diff --binary | shasum -a 256 | awk '{print $1}')"
FM_HEAD="$(git -C "$FMLOCAL" rev-parse HEAD)"
FM_DIFF_SHA="$(git -C "$FMLOCAL" diff --binary | shasum -a 256 | awk '{print $1}')"
{
    printf 'flowvpm_head=%s\nflowvpm_diff_sha256=%s\n' "$VPM_HEAD" "$VPM_DIFF_SHA"
    printf 'fastmultipole_head=%s\nfastmultipole_diff_sha256=%s\n' "$FM_HEAD" "$FM_DIFF_SHA"
    printf 'input_manifest_sha256=%s\n' "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
} > "$BINDIRLOCAL/submission_provenance.txt"

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR $VPMDIR/data/fm049"

rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git \
    "$FMLOCAL/src" "$FMLOCAL/test" "$FMLOCAL/Project.toml" \
    "$REMOTE:$FMDIR/"

rsync -az "$BINDIRLOCAL"/p018_{710..719}_particles.bin \
    "$MANIFEST" "$BINDIRLOCAL/submission_provenance.txt" \
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
  && sbatch scripts/fm049_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
