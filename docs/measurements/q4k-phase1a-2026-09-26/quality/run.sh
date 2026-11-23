#!/bin/bash
# Retired historical quality gate (goinfer#71): it ran a prebuilt binary from a log directory against a
# personal checkout, and arm R's AVX2 kernel came from an external module's working tree.
# Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/q4k-phase1a-2026-09-26/quality/run.sh
# The dated record in this directory (sweep.jsonl, run.log, progress.log) explains what the run measured; the
# recipe is archived in Git and its inputs were not tracked. This stub builds nothing, writes nothing and
# reaches no device.
echo "Retired historical q4k quality gate. Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/q4k-phase1a-2026-09-26/quality/run.sh. See the dated record in this directory." >&2
exit 78
