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


GROUP_STOPPED = "stopped"
GROUP_LIVE = "live"
GROUP_UNKNOWN = "unknown"


def _group_visible(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


def _proc_complete(proc_root=Path("/proc"), mountinfo=Path("/proc/self/mountinfo")):
    """Whether this proc mount can show members of every UID in our group."""
    if str(proc_root) != "/proc":
        return True  # fixture roots are complete by construction
    try:
        lines = mountinfo.read_bytes().splitlines()
    except OSError:
        return False
    matches = []
    for line in lines:
        before, sep, after = line.partition(b" - ")
        if not sep:
            continue
        fields, tail = before.split(), after.split()
        if len(fields) < 6 or len(tail) < 3 or fields[4] != b"/proc" or tail[0] != b"proc":
            continue
        matches.append(fields[5].split(b",") + tail[2].split(b","))
    if len(matches) != 1:
        return False
    for option in matches[0]:
        if option.startswith(b"hidepid=") and option not in (b"hidepid=0", b"hidepid=off"):
            return False
    return True


def _task_liveness(pid, proc_root=Path("/proc")):
    """Whether every visible task of one zombie thread-group leader is stopped."""
    task_dir = Path(proc_root) / str(pid) / "task"
    try:
        tasks = list(task_dir.iterdir())
    except OSError:
        return GROUP_UNKNOWN
    if not tasks:
        return GROUP_UNKNOWN
    for task in tasks:
        if not task.name.isdigit():
            continue
        try:
            raw = (task / "stat").read_bytes()
        except OSError:
            return GROUP_UNKNOWN
        close = raw.rfind(b")")
        if close < 0:
            return GROUP_UNKNOWN
        fields = raw[close + 2 :].split()
        if not fields:
            return GROUP_UNKNOWN
        if fields[0] != b"Z":
            return GROUP_LIVE
    return GROUP_STOPPED


def _group_liveness(pgid, proc_root=Path("/proc")):
    """Classify one process group from /proc bytes without treating zombies as live.

    A confirmed zombie is stopped. Any unreadable/malformed member state, or a
    group still visible after every member was classified, is unknown, never
    okay. The stat file is parsed as bytes so a comm containing ')' or non-UTF8
    bytes cannot change the fields.
    """
    root = Path(proc_root)
    try:
        entries = list(root.iterdir())
    except OSError:
        return GROUP_UNKNOWN
    saw_member = False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            return GROUP_UNKNOWN
        close = raw.rfind(b")")
        if close < 0:
            return GROUP_UNKNOWN
        fields = raw[close + 2 :].split()
        if len(fields) < 3:
            return GROUP_UNKNOWN
        try:
            pgrp = int(fields[2])
        except ValueError:
            return GROUP_UNKNOWN
        if pgrp != pgid:
            continue
        saw_member = True
        if fields[0] != b"Z":
            return GROUP_LIVE
        task_state = _task_liveness(entry.name, proc_root)
        if task_state != GROUP_STOPPED:
            return task_state
    if saw_member:
        return GROUP_STOPPED if _proc_complete(proc_root) else GROUP_UNKNOWN
    return GROUP_STOPPED if _group_visible(pgid) is False else GROUP_UNKNOWN


def _finish_group(proc, pgid, grace_seconds, proc_root=Path("/proc")):
    """Terminate the owned process group and report whether it is stopped.

    A reaped leader with an absent group is accepted immediately, even if some
    unrelated /proc entry is unreadable. A group still visible during the TERM
    wait is never accepted from a snapshot; it is escalated to KILL. After a
    successful KILL, absence or two fresh separated zombie-only confirmations
    are required. Live or indeterminate state, or an unreaped leader, fails.
    """
    status = proc.poll()
    visible = _group_visible(pgid)
    liveness = _group_liveness(pgid, proc_root)
    if status is not None and visible is False and liveness != GROUP_LIVE:
        return status, True

    term_sent = _signal_group(pgid, signal.SIGTERM)
    if status is None and not term_sent:
        try:
            proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        status = proc.poll()
        visible = _group_visible(pgid)
        liveness = _group_liveness(pgid, proc_root)
        if status is not None and visible is False and liveness != GROUP_LIVE:
            return status, True
        time.sleep(0.02)

    kill_sent = _signal_group(pgid, signal.SIGKILL)
    if status is None and not kill_sent:
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_seconds
    zombie_confirmations = 0
    while time.monotonic() < deadline:
        status = proc.poll()
        visible = _group_visible(pgid)
        liveness = _group_liveness(pgid, proc_root)
        if status is not None and visible is False and liveness != GROUP_LIVE:
            return status, True
        if status is not None and visible is True and liveness == GROUP_STOPPED:
            zombie_confirmations += 1
            if zombie_confirmations >= 2:
                return status, True
            time.sleep(0.02)
            continue
        zombie_confirmations = 0
        time.sleep(0.02)

    try:
        status = proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        status = proc.poll()
    return status, status is not None and _group_visible(pgid) is False and _group_liveness(pgid, proc_root) != GROUP_LIVE


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
    cleanup_ok = False
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
            status, cleanup_ok = _finish_group(proc, pgid, grace_seconds)
        for sig, prev in previous.items():
            signal.signal(sig, prev)

    interrupted_signal = pending[0] if pending else None
    elapsed = time.monotonic() - begin
    return status, interrupted_signal, cleanup_ok, elapsed, peak_swap, swap_samples


def _should_promote(interrupted_signal, status, cleanup_ok):
    return interrupted_signal is None and status == 0 and cleanup_ok


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

    status, interrupted_signal, cleanup_ok, elapsed, peak_swap, swap_samples = run_monitored(args.command[1:], args.output)
    end = host()
    meta = {
        "start": start, "end": end, "exit_status": status,
        "cleanup_ok": cleanup_ok,
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
    if _should_promote(interrupted_signal, status, cleanup_ok):
        (args.output / "cpu.json.partial").rename(args.output / "cpu.json")
    print(json.dumps(meta))
    if interrupted_signal is not None:
        return 128 + interrupted_signal
    if not cleanup_ok:
        return 1
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
