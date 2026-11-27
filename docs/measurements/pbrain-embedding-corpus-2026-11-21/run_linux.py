#!/usr/bin/env python3
"""Record one prebuilt native CPU cell while the caller holds the exclusive timing lock."""

import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

GIB_KIB = 1024 * 1024


def proc_kib(path, fields):
    values = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key in fields:
            values[key] = int(value.split()[0])
    return values


def temperature():
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        if (hwmon / "name").read_text().strip() == "k10temp":
            for label in sorted(hwmon.glob("temp*_label")):
                if label.read_text().strip() == "Tctl":
                    return int(label.with_name(label.name.replace("_label", "_input")).read_text()) / 1000
    return None


def host():
    out = proc_kib(Path("/proc/meminfo"), {"MemAvailable", "SwapTotal", "SwapFree"})
    out.update(time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               load_1m=os.getloadavg()[0], tctl_c=temperature())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-gib", required=True, type=float)
    parser.add_argument("--physical-cores", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if len(args.command) < 2 or args.command[0] != "--" or args.fit_gib <= 0 or args.physical_cores <= 0:
        parser.error("give a positive fit estimate, core count, and prebuilt command after --")

    start = host()
    if start["load_1m"] > min(1.0, args.physical_cores / 2) or start["MemAvailable"] < (args.fit_gib + 2) * GIB_KIB:
        print(f"timing cell refused before launch: {json.dumps(start)}", file=sys.stderr)
        return 75

    args.output.mkdir()
    peak_swap = 0
    swap_samples = 0
    begin = time.monotonic()
    with (args.output / "cpu.json.partial").open("x") as out, (args.output / "stderr.log").open("x") as err:
        proc = subprocess.Popen(args.command[1:], stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        status_path = Path(f"/proc/{proc.pid}/status")
        while proc.poll() is None:
            try:
                values = proc_kib(status_path, {"VmSwap"})
            except FileNotFoundError:
                break
            if "VmSwap" in values:
                peak_swap = max(peak_swap, values["VmSwap"])
                swap_samples += 1
            time.sleep(0.1)
        status = proc.wait()
    end = host()
    meta = {
        "start": start, "end": end, "exit_status": status,
        "elapsed_seconds": time.monotonic() - begin,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "sampled_peak_vmswap_kib": peak_swap if swap_samples else None,
        "vmswap_samples": swap_samples,
        "fit_estimate_gib": args.fit_gib, "physical_cores": args.physical_cores,
        "logical_cpus": os.cpu_count(), "kernel": os.uname().release,
        "machine": os.uname().machine, "gomaxprocs": os.environ.get("GOMAXPROCS"),
    }
    (args.output / "host.json").write_text(json.dumps(meta, indent=2) + "\n")
    if status == 0:
        (args.output / "cpu.json.partial").rename(args.output / "cpu.json")
    print(json.dumps(meta))
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
