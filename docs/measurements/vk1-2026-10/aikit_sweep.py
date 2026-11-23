"""Retired historical module-version sweep (goinfer#71).

Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/vk1-2026-10/aikit_sweep.py. It rewrote the
external module's version in a temporary go.mod and ran `go test -mod=mod` once per version, which
resolves that module from the proxy. The dated record in this directory explains what it measured;
the recipe is archived in Git and its inputs were not tracked.
"""

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "Retired historical module-version sweep. Original recipe: goinfer ab2721995e027c8ce6630396c922916cb2357a1d at "
        "docs/measurements/vk1-2026-10/aikit_sweep.py. See the dated record in this directory.\n"
    )
    sys.exit(78)
