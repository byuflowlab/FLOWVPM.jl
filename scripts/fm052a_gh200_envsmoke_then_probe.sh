#!/usr/bin/env bash
# 052a Phase A/B combined GH200 job: run the offline-environment smoke test
# (the 052a Phase-A exit criterion) and, only if it passes, exec the standard
# fm052 probe stage under the same allocation. Submit as, e.g.:
#   sbatch --job-name=fp052-gh200-probe --partition=mgh \
#     --gres=gpu:gh200:1 --constraint=arm --cpus-per-task=72 --mem=192G \
#     --time=02:00:00 \
#     --output=$HOME/FLOWPanel-052-gh200/data/fm052_multiarch/gh200/slurm/fp052-gh200-probe-%j.out \
#     --export=ALL,FP052_ARCH=gh200,FP052_STAGE=probe,FP052_GPU_GRES=gh200,FP052_PARTITION=mgh \
#     $HOME/FLOWVPM-052-gh200/scripts/fm052a_gh200_envsmoke_then_probe.sh
# The probe's own gates, manifests, and provenance are untouched: this wrapper
# only fronts them with the Phase-A exit criterion.
source /etc/profile
set -euo pipefail

: "${SLURM_JOB_ID:?must run under Slurm}"
test "${FP052_ARCH:-}" = gh200 || { echo "this wrapper is gh200-only" >&2; exit 64; }
test "${FP052_STAGE:-}" = probe || { echo "this wrapper only fronts the probe stage" >&2; exit 64; }

JULIA_BIN="${FP052_JULIA_BIN:-/home/rander39/julia/julia-1.11.7/bin/julia}"
ENVDIR="${FP052_ENV:-$HOME/fm052env-gh200}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-052-gh200}"
export JULIA_DEPOT_PATH="${FP052_DEPOT:-$HOME/fm052depot-gh200}"
export JULIA_PKG_OFFLINE=1
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"

echo "== 052a env smoke: job=$SLURM_JOB_ID node=$(hostname) arch=$(uname -m)"
test -x "$JULIA_BIN" || { echo "missing ARM-native Julia executable: $JULIA_BIN" >&2; exit 75; }
# evidence for the A1 fallback decision; non-fatal here (the probe enforces identity)
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
module avail cuda 2>&1 | head -20 || true

start=$(date +%s)
"$JULIA_BIN" --project="$ENVDIR" -e 'using Pkg; Pkg.offline(true); Pkg.precompile()'
echo "== offline precompile completed in $(( $(date +%s) - start )) s"
# warm every manifest JLL once, tolerantly: Pkg.precompile() skips JLLs outside
# the load graph (windows-only, stdlib-shipped, weakdep-only), and their first
# require on a cold depot can fail one-shot in the precompile-on-load hook
# (job 13476578); the probe then re-checks the same loads strictly
"$JULIA_BIN" --project="$ENVDIR" -e 'using Pkg
for (uuid, pkg) in Pkg.dependencies()
    name = pkg.name
    (name === nothing || !endswith(name, "_jll")) && continue
    try
        Base.require(Base.PkgId(uuid, name))
    catch err
        println("jll warm-load failed (probe will re-check): ", name, ": ",
                sprint(showerror, err))
    end
end'
echo "== jll warm pass completed in $(( $(date +%s) - start )) s total"
"$JULIA_BIN" --project="$ENVDIR" -e 'using FastMultipole, CUDA; CUDA.versioninfo()'
echo "== 052a Phase-A exit criterion PASSED (job $SLURM_JOB_ID)"

exec bash "$VPMDIR/scripts/fm052_arch_run.sh"
