#!/usr/bin/env bash
#SBATCH --job-name=fp052-arch-chain-invalid
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=fp052-arch-chain-invalid-%j.out
# Combined single-allocation multi-architecture chain: probe -> smoke -> mature
# in one Slurm job, so the GPU queue wait is paid once per architecture. Each
# stage still runs through fm052_arch_run.sh with its full manifest/gating
# machinery; a later stage starts only if the prior stage's result manifest for
# THIS job says pass. An "ineligible" probe (official low-memory rejection)
# ends the chain cleanly with exit 0 — that outcome is a recorded result.
# Stage list can be restricted via FP052_CHAIN_STAGES (e.g. "probe").
source /etc/profile
set -euo pipefail

: "${FP052_ARCH:?}"
: "${FP052_GPU_GRES:?}"
: "${FP052_PARTITION:?}"
: "${SLURM_JOB_ID:?fm052_arch_chain.sh must run under Slurm}"
VPMDIR="${FP052_VPMDIR:-$HOME/FLOWVPM-052-$FP052_ARCH}"
FPDIR="${FP052_DIR:-$HOME/FLOWPanel-052-$FP052_ARCH}"
MANIFEST_DIR="$FPDIR/data/fm052_multiarch/$FP052_ARCH/manifests"
STAGES="${FP052_CHAIN_STAGES:-probe smoke mature}"

export FP052_CHAIN=1

stage_status() {
  sed -n 's/^status = "\(.*\)"/\1/p' \
    "$MANIFEST_DIR/fm052_${FP052_ARCH}_${1}_${SLURM_JOB_ID}_result.toml"
}

for stage in $STAGES; do
  echo "fm052 arch chain [$FP052_ARCH]: entering stage $stage (job $SLURM_JOB_ID)"
  case "$stage" in
    probe)
      FP052_STAGE=probe bash "$VPMDIR/scripts/fm052_arch_run.sh"
      status=$(stage_status probe)
      if test "$status" = ineligible; then
        echo "fm052 arch chain [$FP052_ARCH]: probe recorded official ineligibility; chain ends here"
        exit 0
      fi
      test "$status" = pass || { echo "probe status=$status; aborting chain"; exit 81; }
      ;;
    smoke)
      FP052_STAGE=smoke FP052_PROBE_JOB="$SLURM_JOB_ID" \
        bash "$VPMDIR/scripts/fm052_arch_run.sh"
      test "$(stage_status smoke)" = pass || { echo "smoke did not pass; aborting chain"; exit 81; }
      ;;
    mature)
      FP052_STAGE=mature FP052_PROBE_JOB="$SLURM_JOB_ID" FP052_SMOKE_JOB="$SLURM_JOB_ID" \
        bash "$VPMDIR/scripts/fm052_arch_run.sh"
      test "$(stage_status mature)" = pass || { echo "mature did not pass; aborting chain"; exit 81; }
      ;;
    *) echo "invalid chain stage: $stage"; exit 64 ;;
  esac
done
echo "fm052 arch chain [$FP052_ARCH]: all requested stages passed ($STAGES)"
