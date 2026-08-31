#!/usr/bin/env bash
# Prepare an isolated task-052 source tree and per-architecture environment.
# This script never calls sbatch and never writes the live *-046/*-052 trees.
set -euo pipefail

ARCH=${1:?usage: bash scripts/fm052_arch_prepare.sh ARCH}
case "$ARCH" in h200|h100|gh200|b200|l40s) ;; *) echo "invalid architecture slug: $ARCH" >&2; exit 64;; esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VPMLOCAL=$(cd -- "$SCRIPT_DIR/.." && pwd)
FMLOCAL=$(cd -- "$VPMLOCAL/../FastMultipole" && pwd)
FPLOCAL=$(cd -- "$VPMLOCAL/../FLOWPanel.jl" && pwd)
REMOTE=${FP052_REMOTE:-orc}
VPMDIR="FLOWVPM-052-$ARCH"
FMDIR="FastMultipole-052-$ARCH"
FPDIR="FLOWPanel-052-$ARCH"
ENVDIR="fm052env-$ARCH"
CANONICAL_ENV=fm052env_cuda63_geoiofree
MESHES=(dji9443_new_40_40.msh dji9443_20260725_45_185_capped_captess4.msh)

for mesh in "${MESHES[@]}"; do
  test -f "$FPLOCAL/examples/data/$mesh" || { echo "mesh missing: $mesh" >&2; exit 65; }
done

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR $FPDIR/examples/data $FPDIR/data/fm052_multiarch/$ARCH/slurm"
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
rsync -az --delete --exclude data "$FPLOCAL/examples/" "$REMOTE:$FPDIR/examples/"
for mesh in "${MESHES[@]}"; do
  rsync -az "$FPLOCAL/examples/data/$mesh" "$REMOTE:$FPDIR/examples/data/"
done

ssh "$REMOTE" "bash -lc '
  source /etc/profile
  set -euo pipefail
  module load julia/1.11.7-6bmogfl
  test -s \"\$HOME/$CANONICAL_ENV/Project.toml\"
  test -s \"\$HOME/$CANONICAL_ENV/Manifest.toml\"
  mkdir -p \"\$HOME/$ENVDIR\"
  cp \"\$HOME/$CANONICAL_ENV/Project.toml\" \"\$HOME/$ENVDIR/Project.toml\"
  cp \"\$HOME/$CANONICAL_ENV/Manifest.toml\" \"\$HOME/$ENVDIR/Manifest.toml\"
  test ! -f \"\$HOME/$CANONICAL_ENV/LocalPreferences.toml\" || cp \
    \"\$HOME/$CANONICAL_ENV/LocalPreferences.toml\" \"\$HOME/$ENVDIR/LocalPreferences.toml\"
  export JULIA_PKG_PRECOMPILE_AUTO=0
  julia --project=\"\$HOME/$ENVDIR\" -e \"using Pkg; \
    Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); \
    Pkg.develop(path=\\\"\$HOME/$VPMDIR\\\"); \
    Pkg.develop(path=\\\"\$HOME/$FPDIR\\\"); Pkg.resolve()\"
'"

if test "$ARCH" = gh200; then
  ssh "$REMOTE" "bash -lc '
    source /etc/profile
    set -euo pipefail
    module load julia/1.11.7-6bmogfl
    arm_julia=\"\${FP052_JULIA_BIN:-/home/rander39/julia/julia-1.11.7/bin/julia}\"
    depot=\"\$HOME/fm052depot-gh200\"
    test -x \"\$arm_julia\" || { echo \"missing supplied ARM Julia: \$arm_julia\" >&2; exit 66; }
    file -L \"\$arm_julia\" | grep -Eiq \"aarch64|ARM aarch64\" || {
      echo \"supplied GH200 Julia is not an aarch64 executable: \$(file -L \"\$arm_julia\")\" >&2; exit 67; }
    mkdir -p \"\$depot\"
    export JULIA_DEPOT_PATH=\"\$depot\"
    export JULIA_PKG_PRECOMPILE_AUTO=0
    julia --project=\"\$HOME/$ENVDIR\" -e '\''using Pkg, Base.BinaryPlatforms;
      platform=Platform(\"aarch64\", \"linux\"; libc=\"glibc\");
      Pkg.instantiate(; platform=platform, allow_autoprecomp=false)'\''
  '"
else
  ssh "$REMOTE" "bash -lc '
    source /etc/profile
    set -euo pipefail
    module load julia/1.11.7-6bmogfl
    export JULIA_PKG_PRECOMPILE_AUTO=0
    julia --project=\"\$HOME/$ENVDIR\" -e \"using Pkg; Pkg.instantiate(; allow_autoprecomp=false)\"
  '"
fi

echo "prepared isolated $ARCH sources: $FPDIR, $VPMDIR, $FMDIR"
echo "prepared isolated environment: $ENVDIR"
echo "no Slurm job was submitted"
