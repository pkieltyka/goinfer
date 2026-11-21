"""Run the registered A B B A batch-pairs experiment; hold the exclusive timing lock around it."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime

HOME = Path.home()
CPU = HOME / "goinfer-logs/pbe1-corpus-2026-11-21"
MODEL = HOME / ".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e"
INPUTS = ["-manifest", str(CPU / "source/manifest.json"), "-fixture", str(CPU / "fixture"), "-model", str(MODEL),
          "-preflight", str(CPU / "preflight.json")]
DIGEST = "420988ad6653a9141a56ef61f12de91adc07255705426a3042f9a41cf413d732"
ORDER = "ABBA" * 5
ARM = {"A": "single", "B": "pairs"}


def host():
    mem = next(int(row.split()[1]) for row in Path("/proc/meminfo").read_text().splitlines()
               if row.startswith("MemAvailable:"))
    temp = None
    for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
        if (sensor / "name").is_file() and (sensor / "name").read_text().strip() == "k10temp":
            temp = int((sensor / "temp1_input").read_text()) / 1000
            break
    return {"time": datetime.now().astimezone().isoformat(), "load_1m": os.getloadavg()[0], "available_kib": mem,
            "cpu_tctl_c": temp}


def device():
    return subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,memory.free,temperature.gpu",
                                    "--format=csv,noheader"], text=True, timeout=5).strip()


def run_cell(exe, arm, path):
    pre = host()
    if pre["load_1m"] > 1.0:
        return False
    pre["gpu"] = device()
    if pre["available_kib"] < 10 * 1024 * 1024 or int(pre["gpu"].split(",")[-2].split()[0]) < 3072:
        raise RuntimeError(f"the preflight fit margin is unavailable: {pre}")
    path.mkdir()
    (path / "start.json").write_text(json.dumps(pre, indent=2) + "\n")
    print(path.name, arm, pre["time"], flush=True)
    with (path / "result.json").open("w") as out, (path / "stderr.log").open("w") as err:
        subprocess.run([str(exe), *INPUTS, "-arm", arm], stdout=out, stderr=err, check=True, timeout=120)
    return True


def read_result(path, arm):
    x = json.loads(path.read_text())
    assert x["backend"] == "cuda" and x["arm"] == arm and x["manifest_sha256"] == DIGEST
    assert len(x["samples"]) == 139 and len(x["queries"]) == 107
    assert len(x["document_groups"]) == (32 if arm == "single" else 16)
    identity = tuple(tuple(row[k] for k in ("id", "role", "tokens", "ids_sha256", "vector_sha256")) for row in x["samples"])
    rankings = tuple((row["id"], row["abstained"], *(t["id"] for t in row["top_5"])) for row in x["queries"])
    timings = {
        "document_ms": sum(g["wall_ns"] for g in x["document_groups"]) / 1e6,
        "warm_query_ms": statistics.median([row["wall_ns"] for row in x["samples"] if row["role"] == "query"][1:]) / 1e6,
        "load_ms": x["load_wall_ns"] / 1e6,
        "attach_ms": x["device_load_wall_ns"] / 1e6,
    }
    return (identity, rankings), timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("destination", type=Path)
    a = parser.parse_args()
    a.destination.mkdir(exist_ok=False)
    start = time.monotonic()
    cells, first = [], None
    for i, arm in enumerate(ORDER):
        path = a.destination / f"{i + 1:02d}-{arm}"
        while not run_cell(a.binary, ARM[arm], path):
            if time.monotonic() - start > 600:
                raise RuntimeError("no quiet window within the registered 10 minutes")
            time.sleep(5)
        identity, metrics = read_result(path / "result.json", ARM[arm])
        if first is None:
            first = identity
        elif identity != first:
            raise RuntimeError(f"{path.name}: the exact token, vector or ranking comparison failed")
        cells.append({"cell": path.name, "arm": arm, **metrics})
    ratios = {"document_ms": [], "warm_query_ms": []}
    for x, y in zip(cells[::2], cells[1::2]):
        single, pairs = (x, y) if x["arm"] == "A" else (y, x)
        for metric in ratios:
            ratios[metric].append(single[metric] / pairs[metric])
    result = {metric: {"median_ratio_single_over_pairs": statistics.median(v), "pairs_faster": sum(r > 1.0 for r in v),
                       "ratios": v} for metric, v in ratios.items()}
    (a.destination / "summary.json").write_text(json.dumps({"cells": cells, "ratios": result, "identity_match": True},
                                                           indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
