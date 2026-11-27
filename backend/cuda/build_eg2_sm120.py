#!/usr/bin/env python3
"""Reproduce the sealed EG2 sm_120 images directly from committed PTX."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "eg2/testdata/eg2-sm120"

def verify(path, expected):
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise SystemExit(f"{path.name}: SHA-256 {got}, expected {expected}")

def assemble(wheel, manifest, staging):
    exe = staging / "ptxas"
    with zipfile.ZipFile(wheel) as archive:
        exe.write_bytes(archive.read(manifest["ptxas_member"]))
    verify(exe, manifest["ptxas_sha256"])
    exe.chmod(0o700)
    for module in manifest["modules"]:
        ptx = ROOT / module["ptx"]
        image = staging / module["cubin"]
        verify(ptx, module["ptx_sha256"])
        result = subprocess.run([str(exe), *manifest["ptxas_flags"], "-o", str(image), str(ptx)],
                                cwd=staging, capture_output=True, check=False)
        if result.returncode:
            raise SystemExit(result.stderr.decode(errors="replace"))
        verify(image, module["cubin_sha256"])
        if image.stat().st_size != module["cubin_bytes"]:
            raise SystemExit(f"{image.name}: size differs from manifest")
        print(f"{staging.name} {module['unit']} {module['cubin_bytes']} B {module['cubin_sha256']}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="pinned NVIDIA CUDA 12.9.86 wheel")
    parser.add_argument("--out", type=Path, help="write verified images into a fresh directory instead of checking the embedded files")
    args = parser.parse_args()
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    verify(args.wheel, manifest["wheel_sha256"])
    with tempfile.TemporaryDirectory(prefix="eg2-sm120-") as tmp:
        first, second = (Path(tmp) / name for name in ("first", "second"))
        for staging in (first, second):
            staging.mkdir()
            assemble(args.wheel, manifest, staging)
        if args.out:
            args.out.mkdir()
        for module in manifest["modules"]:
            name = module["cubin"]
            data = (first / name).read_bytes()
            if data != (second / name).read_bytes():
                raise SystemExit(f"{name}: independent assemblies differ")
            if args.out:
                (args.out / name).write_bytes(data)
            elif data != (ASSETS / name).read_bytes():
                raise SystemExit(f"{name}: embedded image differs from both assemblies")
    print("PASS: both independent assemblies reproduce all six images")

if __name__ == "__main__":
    main()
