#!/usr/bin/env python3
"""Record one prebuilt native CPU cell while the caller holds the exclusive timing lock."""

import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

GIB_KIB = 1024 * 1024
TERMINATE_GRACE_SECONDS = 5.0


class _Interrupted(BaseException):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


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


def _terminate_child(proc, grace_seconds):
    if proc.poll() is not None:
        return proc.wait()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            if sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        if sig == signal.SIGTERM:
            try:
                return proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                continue
        return proc.wait()
    return proc.wait()


def run_monitored(command, output_dir, grace_seconds=TERMINATE_GRACE_SECONDS):
    """Run command in its own process group, reaping it on signals or errors before exit."""
    output_dir.mkdir()
    peak_swap = 0
    swap_samples = 0
    interrupted_signal = None
    begin = time.monotonic()

    def _handle_signal(signum, _frame):
        raise _Interrupted(signum)

    handled = [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]
    previous = {sig: signal.getsignal(sig) for sig in handled}
    for sig in handled:
        signal.signal(sig, _handle_signal)

    proc = None
    status = None
    try:
        with (output_dir / "cpu.json.partial").open("x") as out, (output_dir / "stderr.log").open("x") as err:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
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
    except _Interrupted as exc:
        interrupted_signal = exc.signum
        if proc is not None:
            status = _terminate_child(proc, grace_seconds)
    except BaseException:
        if proc is not None:
            _terminate_child(proc, grace_seconds)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            status = _terminate_child(proc, grace_seconds)
        for sig, prev in previous.items():
            signal.signal(sig, prev)

    elapsed = time.monotonic() - begin
    return status, interrupted_signal, elapsed, peak_swap, swap_samples


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

    status, interrupted_signal, elapsed, peak_swap, swap_samples = run_monitored(args.command[1:], args.output)
    end = host()
    meta = {
        "start": start, "end": end, "exit_status": status,
        "elapsed_seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "sampled_peak_vmswap_kib": peak_swap if swap_samples else None,
        "vmswap_samples": swap_samples,
        "fit_estimate_gib": args.fit_gib, "physical_cores": args.physical_cores,
        "logical_cpus": os.cpu_count(), "kernel": os.uname().release,
        "machine": os.uname().machine, "gomaxprocs": os.environ.get("GOMAXPROCS"),
    }
    if interrupted_signal is not None:
        meta["interrupted_signal"] = interrupted_signal
    (args.output / "host.json").write_text(json.dumps(meta, indent=2) + "\n")
    if interrupted_signal is None and status == 0:
        (args.output / "cpu.json.partial").rename(args.output / "cpu.json")
    print(json.dumps(meta))
    if interrupted_signal is not None:
        return 128 + interrupted_signal
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
