#!/usr/bin/env bash
# Fetch one architecture-qualified result tree without overwriting prior fetches.
set -euo pipefail

ARCH=${1:?usage: bash scripts/fm052_arch_fetch.sh ARCH JOB_ID}
JOB_ID=${2:?usage: bash scripts/fm052_arch_fetch.sh ARCH JOB_ID}
case "$ARCH" in h200|h100|gh200|b200|l40s) ;; *) echo "invalid architecture slug: $ARCH" >&2; exit 64;; esac
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "JOB_ID must be numeric" >&2; exit 65; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VPMLOCAL=$(cd -- "$SCRIPT_DIR/.." && pwd)
FMLOCAL=$(cd -- "$VPMLOCAL/../FastMultipole" && pwd)
REMOTE=${FP052_REMOTE:-orc}
REMOTE_ROOT="FLOWPanel-052-$ARCH/data/fm052_multiarch/$ARCH"
DEST="$FMLOCAL/MATRIX_OPERATOR_REFACTOR/data/fm052_multiarch/$ARCH/results-$JOB_ID"
test ! -e "$DEST" || { echo "refusing to overwrite fetched artifacts: $DEST" >&2; exit 66; }
mkdir -p "$DEST"
rsync -az "$REMOTE:$REMOTE_ROOT/" "$DEST/"
ssh "$REMOTE" "bash -lc 'compgen -G \"FLOWPanel-052-$ARCH/data/fm052_multiarch/$ARCH/manifests/fm052_${ARCH}_*_${JOB_ID}_result.toml\" >/dev/null'"
for manifest in "$DEST"/manifests/fm052_${ARCH}_*_${JOB_ID}_result.toml; do
  test -f "$manifest" || continue
  stage=$(sed -n 's/^stage = "\(.*\)"/\1/p' "$manifest")
  run_dir=$(sed -n 's/^run_dir = "\(.*\)"/\1/p' "$manifest")
  if test -n "$run_dir"; then
    mkdir -p "$DEST/runs/$(basename "$run_dir")"
    rsync -az "$REMOTE:$run_dir/" "$DEST/runs/$(basename "$run_dir")/"
    run_log="${run_dir}.log"
    rsync -az "$REMOTE:$run_log" "$DEST/runs/" 2>/dev/null || true
  fi
  report_dir=$(sed -n 's/^report_dir = "\(.*\)"/\1/p' "$manifest")
  if test -n "$report_dir"; then
    mkdir -p "$DEST/reports/$stage"
    rsync -az "$REMOTE:$report_dir/" "$DEST/reports/$stage/"
  fi
done
find "$DEST" -type f -print0 | sort -z | xargs -0 shasum -a 256 > "$DEST/sha256_fetched.txt"
echo "$DEST"
