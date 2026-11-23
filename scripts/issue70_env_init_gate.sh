#!/usr/bin/env bash
# issue70_env_init_gate.sh — goinfer#70: importing backend/cuda for text embeddings must perform no
# environment reads before main.
#
# Builds a minimal consumer that imports only backend/cuda and embeddinggemma2, for each release target
# (GOOS/GOARCH explicit, not just two tag sets on one host arch), and asserts that decoder.init contains
# no os.Getenv/os.LookupEnv call. No GPU, no model, no network: the consumer resolves this checkout with
# replace directives and GOWORK=off, and cgo is off because the consumer is pure Go.
#
#   scripts/issue70_env_init_gate.sh            # both targets
#   scripts/issue70_env_init_gate.sh amd64      # one
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/issue70-gate.XXXXXX")
trap 'rm -rf "$work"' EXIT
only=${1:-both}

consumer() {
	mkdir -p "$1"
	cat >"$1/go.mod" <<EOF
module issue70gate

go 1.27.2

require (
	github.com/pkieltyka/goinfer v0.0.0
	github.com/pkieltyka/goinfer/backend/cuda v0.0.0
)

replace github.com/pkieltyka/goinfer => $repo

replace github.com/pkieltyka/goinfer/backend/cuda => $repo/backend/cuda
EOF
	cat >"$1/main.go" <<'EOF'
// A consumer that only wants text embeddings: importing backend/cuda must not read the environment
// before main.
package main

import (
	_ "github.com/pkieltyka/goinfer/backend/cuda"
	_ "github.com/pkieltyka/goinfer/embeddinggemma2"
)

func main() {}
EOF
}

check() {
	local arch=$1 tags=$2 dir=$work/probe-$1
	consumer "$dir"
	(
		cd "$dir"
		GOWORK=off CGO_ENABLED=0 GOFLAGS=-mod=mod GOOS=linux GOARCH="$arch" \
			go build -tags "$tags" -o probe .
	)
	# Disassemble once into a file: piping objdump into `grep -q` closes the pipe early, and objdump then
	# reports a broken pipe, which would read as a failure.
	if ! go tool objdump "$dir/probe" >"$dir/all.dis" 2>"$dir/objdump.err"; then
		echo "FAIL: objdump failed on linux/$arch: $(head -1 "$dir/objdump.err")" >&2
		return 1
	fi
	# The package must be linked and its symbols visible, or a zero count means the disassembly looked at
	# nothing. readSettings is the snapshot the fix introduced, so it is the anchor.
	if ! grep -q 'TEXT github.com/pkieltyka/goinfer/decoder\.Load(SB)' "$dir/all.dis"; then
		echo "FAIL: decoder.Load is not in the linux/$arch binary; the scan would prove nothing" >&2
		return 1
	fi
	# An absent decoder.init is the best outcome: the package has nothing to run at init. An existing one
	# must contain no environment call.
	if grep -q 'TEXT github.com/pkieltyka/goinfer/decoder\.init(SB)' "$dir/all.dis"; then
		local n
		n=$(sed -n '/TEXT github.com\/pkieltyka\/goinfer\/decoder\.init(SB)/,/^$/p' "$dir/all.dis" |
			grep -cE 'CALL.*(os\.Getenv|os\.LookupEnv)' || true)
		printf 'linux/%s %s: decoder.init environment calls = %s\n' "$arch" "$tags" "$n"
		if [ "$n" != "0" ]; then
			echo "FAIL: importing backend/cuda reads the environment at init on linux/$arch" >&2
			return 1
		fi
	else
		printf 'linux/%s %s: decoder.init absent (nothing runs at init)\n' "$arch" "$tags"
	fi
}

rc=0
case "$only" in
amd64) check amd64 'cuda vulkan eg2_combined eg2_fp16' || rc=1 ;;
arm64) check arm64 'cuda eg2_combined eg2_fp16' || rc=1 ;;
*)
	check amd64 'cuda vulkan eg2_combined eg2_fp16' || rc=1
	check arm64 'cuda eg2_combined eg2_fp16' || rc=1
	;;
esac
if [ "$rc" -ne 0 ]; then
	echo "VERDICT: FAIL — goinfer#70 env-init gate"
	exit 1
fi
echo "VERDICT: PASS — no environment reads in decoder.init for the release targets"
