#!/bin/bash
# Retired historical release revalidation run (goinfer#71): its launch condition required an exact
# external module version in the workspace's go.mod, and the dependency is gone.
# Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/release-v0.20.0/run-scoped-revalidation.sh
# The dated record in this directory explains what the run measured; the recipe is archived in Git
# and its inputs were not tracked. This stub builds nothing, writes nothing and reaches no device.
echo "Retired historical release revalidation run. Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/release-v0.20.0/run-scoped-revalidation.sh. See the dated record in this directory." >&2
exit 78
