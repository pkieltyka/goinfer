"""RETIRED by goinfer#71: AX1 copied the aikit towers and GPU backends into goinfer and checked that the
copy still matched. That module is now vendored source in `internal/modelcore` (provenance in its
LICENSE/NOTICE), so the copy and check operations have nothing to operate on and this tool refuses to run
rather than reach for a checkout that no longer exists.

The unchanged tool is goinfer at ab2721995, path docs/measurements/ax1-2026-10/ax1_copy.py. The AX1 record
in this directory is the evidence for the copy it performed; see also docs/upstream-archive.md.
"""

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "ax1_copy.py is retired (goinfer#71): the aikit module it copied is now vendored under internal/modelcore.\n"
        "The unchanged tool is goinfer at ab2721995, path docs/measurements/ax1-2026-10/ax1_copy.py.\n"
    )
    sys.exit(78)
