"""Retired historical module-version sweep (goinfer#71).

The original program is Git blob 2d30fdbde36a816ad78131647934aea55bce7081 in
ab2721995e027c8ce6630396c922916cb2357a1d; retrieve it with

    git cat-file blob 2d30fdbde36a816ad78131647934aea55bce7081

and it is also at that commit's docs/measurements/vk1-2026-10 path. It rewrote the external
module's version in a temporary go.mod and ran `go test -mod=mod` once per version, resolving that
module from the proxy. The dated record beside this file explains what it measured; the recipe is
archived in Git and its inputs were not tracked.
"""

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "Retired historical module-version sweep. Original recipe: goinfer "
        "ab2721995e027c8ce6630396c922916cb2357a1d at docs/measurements/vk1-2026-10, blob "
        "2d30fdbde36a816ad78131647934aea55bce7081 (git cat-file blob 2d30fdbd...).\n"
    )
    sys.exit(78)
