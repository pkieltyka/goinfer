#!/bin/bash
# RETIRED by goinfer#71: this launcher built and ran benchmark arms that lived in the aikit
# dependency (a `-coverpkg=github.com/townsendmerino/aikit/...` and an `aikit_noavx512` build tag), so it
# cannot produce its measurement from this tree any more. It refuses to run rather than build against a
# stale module, append a partial result, or reach a device.
#
# The unchanged launcher is goinfer at ab2721995, path
# docs/measurements/vk1-2026-10/sde-2026-10-26/run-sde-v2.sh. The measurement it produced is recorded in
# this directory; the vendored replacement of the aikit packages is `internal/modelcore` (provenance in
# its LICENSE/NOTICE).
echo "run-sde-v2.sh is retired (goinfer#71): its arms came from the aikit dependency." >&2
echo "The unchanged launcher is goinfer at ab2721995, path docs/measurements/vk1-2026-10/sde-2026-10-26/run-sde-v2.sh." >&2
echo "The record in this directory explains what it measured. Nothing was built, written or run." >&2
exit 78
