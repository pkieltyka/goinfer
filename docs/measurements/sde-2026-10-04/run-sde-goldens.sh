#!/usr/bin/env bash
# H1.2 (docs/completed/task-hardware-coverage-2026-10.md): the CPU parity suite and the tiny-fixture goldens under Intel SDE,
# so the AVX-512 VNNI paths nobody here owns (Ice Lake -icx, Sapphire Rapids -spr) execute, with Haswell -hsw as the
# control (AVX2 only, so it must match the native run on the 3700X).
#
# Compile both suites from this checkout and keep the output in a new directory.
# Emulation is a correctness check; the run does not measure throughput.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd) || exit 2
SDE=${SDE:-$HOME/tools/sde/sde64}
[[ -x "$SDE" ]] || { echo "missing executable $SDE" >&2; exit 2; }
SDE=$(realpath "$SDE") || exit 2
if [[ -n "${OUT:-}" ]]; then
  mkdir -- "$OUT" || exit 2
else
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/goinfer-sde-goldens.XXXXXX") || exit 2
fi
OUT=$(cd "$OUT" && pwd) || exit 2
printf 'SDE logs and binaries: %s\n' "$OUT"
DBIN="$OUT/decoder.test"
LBIN="$OUT/linalg.test"
(
  cd "$REPO" || exit 2
  export GOWORK=off CGO_ENABLED=0
  go test -c -o "$DBIN" ./decoder &&
    go test -c -o "$LBIN" ./internal/modelcore/linalg
) > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 2; }
# Everything cheap under emulation. The LongPrompt tests take 4+ minutes EACH under SDE and are the same arithmetic the
# 15-fixture goldens already cover, so they are skipped and named as skipped. The two ParityManifest closure tests type-check
# the source by running `go list` as a child process, which SDE's Pin can crash (exit 127, a DisableSigStack assertion; seen
# under -hsw on 2026-11-20 and passing natively); they read no CPU feature, so they are skipped here and run natively in CI.
RUN=${RUN:-'Golden|Parity|Int4|Int8|W4A8|Quant'}
SKIP=${SKIP:-'LongPrompt|^TestParityManifest_(coreClosed|loadersClosed)$'}
CPUS=${CPUS:-"hsw icx spr"}
# linalg's FMA-peak test infers the clock from measured throughput (it read 0.35 GHz under SDE): a timing test, meaningless under emulation.
LSKIP=${LSKIP:-'FMAPeak'}
PER_RUN_MIN=${PER_RUN_MIN:-25}

summary="$OUT/summary.txt"
: > "$summary"
{
  printf 'SDE goldens %s\n' "$(date -Is)"
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short
  go version -m "$DBIN" "$LBIN"
  printf 'Decoder run=%s skip=%s, linalg skip=%s\n' "$RUN" "$SKIP" "$LSKIP"
} >> "$summary"
bad=0

for cpu in $CPUS; do
  # Run the model core's kernel tests from their package directory.
  ( cd "$REPO/internal/modelcore/linalg" && "$SDE" -"$cpu" -- "$LBIN" -test.skip "$LSKIP" -test.v -test.timeout "${PER_RUN_MIN}m" ) > "$OUT/linalg-$cpu.log" 2>&1
  lrc=$?
  # decoder tests read ../testdata, so they run from decoder/.
  ( cd "$REPO/decoder" && "$SDE" -"$cpu" -- "$DBIN" -test.run "$RUN" -test.skip "$SKIP" -test.v -test.timeout "${PER_RUN_MIN}m" > "$OUT/decoder-$cpu.log" 2>&1 )
  drc=$?
  for kind in linalg decoder; do
    log="$OUT/$kind-$cpu.log"
    if [ ! -s "$log" ]; then echo "$cpu $kind: NO LOG (did not run)" >> "$summary"; bad=1; continue; fi
    p=$(grep -c -- '^--- PASS' "$log"); s=$(grep -c -- '^--- SKIP' "$log"); f=$(grep -c -- '^--- FAIL' "$log")
    to=$(grep -c 'panic: test timed out' "$log")
    echo "$cpu $kind: pass=$p skip=$s fail=$f timeout=$to" >> "$summary"
    grep -- '^--- FAIL' "$log" | sed "s/^/    /" >> "$summary"
  done
  echo "$cpu exit codes: linalg=$lrc decoder=$drc" >> "$summary"
  # Any failure is unexpected. (TestDecodeParityInt4 has an "amd64-vnni" golden, captured under this same SDE, since 2026-10-04:
  # docs/measurements/sde-2026-10-04/README.md. A failure of it on icx/spr now means the VNNI path moved.)
  if [[ "$lrc" != 0 || "$drc" != 0 ]]; then bad=1; fi
  nfail=$(grep -h -- '^--- FAIL' "$OUT/decoder-$cpu.log" "$OUT/linalg-$cpu.log" 2>/dev/null | wc -l)
  [ "$nfail" -gt 0 ] && bad=1
  grep -q 'panic: test timed out' "$OUT/decoder-$cpu.log" && bad=1
done
echo "overall: $([ $bad = 0 ] && echo 'no unexpected failure' || echo 'UNEXPECTED FAILURE, TIMEOUT, OR MISSING LOG: read the per-CPU logs')" >> "$summary"
exit $bad
