#!/bin/bash
# Pin G-ids-3 (ids-carried-2026-10-15.md, "Tier 2") by day, before queueing: build serve at REV from a detached worktree and copy
# the matrix harness and this gate's plan beside it, so the night job runs what was pinned whatever the tree does meanwhile.
#
#   REV=<commit> B=$HOME/goinfer-bench/ids3-2026-10 bash pin-ids3.sh
#
# Writes $B/bin/serve-cuda-ids3 (+ .rev, .sha256), $B/run-ids3.sh, $B/matrix-pinned/ (the matrix at REV, plan-ids3.json with serve and commit
# filled in, gids3_check.py) and $B/pinned.txt. Needs no GPU.
set -euo pipefail
REV=${REV:?the commit to pin}
B=${B:?the pinned directory}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
HERE=docs/measurements/integration-harness-2026-10
mkdir -p "$B/bin"
WT="$B/wt"
if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add --detach "$WT" "$REV" >/dev/null
fi
git -C "$WT" checkout -q --detach "$REV"
FULL=$(git -C "$WT" rev-parse HEAD)
cat > "$WT/go.work" <<EOF
go 1.27.0

toolchain go1.27.2

use (
	.
	./backend/cuda
	./backend/metal
	./backend/vulkan
)
EOF
WT_ABS=$(cd "$WT" && pwd -P)
GW="$WT/go.work"
root=$(cd "$WT" && GOWORK="$GW" go list -m -f '{{.Dir}}' github.com/pkieltyka/goinfer) || { echo "FATAL: cannot resolve the workspace root module" >&2; exit 2; }
[ "$(cd "$root" && pwd -P)" = "$WT_ABS" ] || { echo "FATAL: the workspace root resolved to $root, not $WT" >&2; exit 2; }
cuda=$(cd "$WT" && GOWORK="$GW" go list -m -f '{{.Dir}}' github.com/pkieltyka/goinfer/backend/cuda) || { echo "FATAL: cannot resolve backend/cuda in the workspace" >&2; exit 2; }
[ "$(cd "$cuda" && pwd -P)" = "$WT_ABS/backend/cuda" ] || { echo "FATAL: backend/cuda resolved to $cuda, not $WT/backend/cuda" >&2; exit 2; }

(cd "$WT" && GOWORK="$GW" CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-ids3" github.com/pkieltyka/goinfer/backend/cuda/cmd/serve)
echo "$FULL" > "$B/bin/serve-cuda-ids3.rev"
sha256sum "$B/bin/serve-cuda-ids3" | cut -d' ' -f1 > "$B/bin/serve-cuda-ids3.sha256"
rm -rf "$B/matrix-pinned"
cp -r "$WT/$HERE/matrix" "$B/matrix-pinned"
cp "$WT/$HERE/ids-carried-tier2/gids3_check.py" "$B/matrix-pinned/"
cp "$WT/$HERE/ids-carried-tier2/run-ids3.sh" "$B/run-ids3.sh"
python3 - "$WT/$HERE/ids-carried-tier2/plan-ids3.json" "$B/matrix-pinned/plan-ids3.json" "$B/bin/serve-cuda-ids3" "$FULL" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
p["serve"], p["commit"] = sys.argv[3], sys.argv[4][:8]
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
{
  echo "Pinned $(date '+%F %T %Z') for G-ids-3 (ids-carried-2026-10-15.md, Tier 2)."
  echo "bin/serve-cuda-ids3 built at $FULL, CGO_ENABLED=0 -tags cuda, from a detached worktree with its own go.work (the workspace modules from this checkout)."
  echo "sha256 $(cat "$B/bin/serve-cuda-ids3.sha256")"
  echo "matrix-pinned/ is $HERE/matrix at $FULL; plan-ids3.json, gids3_check.py and ../run-ids3.sh from $HERE/ids-carried-tier2 at $FULL."
} > "$B/pinned.txt"
cat "$B/pinned.txt"
