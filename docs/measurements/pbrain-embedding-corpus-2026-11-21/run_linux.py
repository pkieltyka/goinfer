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


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _finish_group(proc, pgid, grace_seconds):
    """Terminate the whole child process group, then reap the leader.

    The leader may already be gone while descendants remain, so every stage
    signals the captured process-group id rather than checking only proc.poll().
    Bounded waits keep a stuck group from holding the caller forever.
    """
    status = proc.poll()
    if status is not None and not _group_exists(pgid):
        return status

    _signal_group(pgid, signal.SIGTERM)
    if status is None and not _group_exists(pgid):
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        status = proc.poll()
        if status is not None and not _group_exists(pgid):
            return status
        time.sleep(0.02)

    _signal_group(pgid, signal.SIGKILL)
    if status is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        status = proc.poll()
        if status is not None and not _group_exists(pgid):
            return status
        time.sleep(0.02)
    try:
        return proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return status if status is not None else -signal.SIGKILL


def run_monitored(command, output_dir, grace_seconds=TERMINATE_GRACE_SECONDS):
    """Run command in its own process group, reaping it on signals or errors before exit.

    Signal handlers only record the pending signal; they never raise from the
    handler. Cleanup therefore cannot be interrupted by a second signal and
    always runs to a bounded end, even when the group leader exited before its
    descendants.
    """
    pending = []
    handled = [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]

    def _handle_signal(signum, _frame):
        if not pending:
            pending.append(signum)

    previous = {sig: signal.getsignal(sig) for sig in handled}
    for sig in handled:
        signal.signal(sig, _handle_signal)

    output_dir.mkdir()
    peak_swap = 0
    swap_samples = 0
    begin = time.monotonic()
    proc = None
    pgid = None
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
            pgid = proc.pid
            status_path = Path(f"/proc/{proc.pid}/status")
            while not pending and proc.poll() is None:
                try:
                    values = proc_kib(status_path, {"VmSwap"})
                except FileNotFoundError:
                    break
                if "VmSwap" in values:
                    peak_swap = max(peak_swap, values["VmSwap"])
                    swap_samples += 1
                time.sleep(0.1)
    finally:
        if proc is not None:
            status = _finish_group(proc, pgid, grace_seconds)
        for sig, prev in previous.items():
            signal.signal(sig, prev)

    interrupted_signal = pending[0] if pending else None
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
