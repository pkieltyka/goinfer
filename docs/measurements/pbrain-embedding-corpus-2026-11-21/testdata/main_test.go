package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestRankQuestion(t *testing.T) {
	docs := []input{{id: "z"}, {id: "a"}, {id: "b"}}
	vectors := [][]float32{{1, 0}, {1, 0}, {0, 1}}
	q := question{ID: "a001", Expect: []string{"a", "b"}}
	got := rankQuestion(q, docs, vectors, []float32{1, 0}, 1)
	if got.Top[0].ID != "a" || got.Top[1].ID != "z" || got.FirstExpected != 1 || got.Recall1 != 0.5 || got.Recall5 != 1 || got.Abstained {
		t.Fatalf("unexpected rank or cutoff result: %+v", got)
	}
	q = question{ID: "u001", Kind: "unanswerable", Half: "held-out", Regression: true}
	got = rankQuestion(q, docs, vectors, []float32{0.6, 0.8}, 0.9)
	if !got.Abstained || got.FirstExpected != 0 || got.Recall5 != 0 || got.Top[0].ID != "b" || !got.Regression {
		t.Fatalf("unexpected abstention result: %+v", got)
	}
}

func TestVectorDigest(t *testing.T) {
	if _, err := vectorDigest([]float32{1, 0}, 2); err != nil {
		t.Fatal(err)
	}
	for _, vector := range [][]float32{{1}, {0, 0}, {float32(math.NaN()), 0}} {
		if _, err := vectorDigest(vector, 2); err == nil {
			t.Errorf("accepted invalid vector %v", vector)
		}
	}
}

func TestRunVerifiesPinnedModelFilesBeforeLoad(t *testing.T) {
	goodBytes := []byte("pinned-model-weights-v1")
	goodSHA := fmt.Sprintf("%x", sha256.Sum256(goodBytes))

	w := workload{
		pinSHA: "manifest-sha-pin",
		pins: manifest{
			Model:         "google/embeddinggemma-2",
			ModelRevision: "pinned-rev",
			ModelFiles: map[string]string{
				"model.safetensors": goodSHA,
			},
		},
		docs: []input{{id: "doc1", role: "document", text: "doc"}},
		qs:   []question{{ID: "q1", Kind: "answerable", Text: "query", Expect: []string{"doc1"}}},
	}
	prep := preparation{
		ManifestSHA: w.pinSHA,
		Tokens: []tokenRow{
			{ID: "q1", Role: "query", Count: 1, SHA: "a"},
			{ID: "doc1", Role: "document", Count: 1, SHA: "b"},
		},
	}

	t.Run("tampered weights fail before load", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("tampered-weights"), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := run(w, dir, prep)
		if err == nil {
			t.Fatal("expected run to reject tampered model.safetensors")
		}
		if !strings.Contains(err.Error(), "verify pinned model before load") || !strings.Contains(err.Error(), "model.safetensors") {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("missing pinned file fails before load", func(t *testing.T) {
		dir := t.TempDir()
		_, err := run(w, dir, prep)
		if err == nil {
			t.Fatal("expected run to reject missing model.safetensors")
		}
		if !strings.Contains(err.Error(), "verify pinned model before load") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestRunLinuxReapsChildOnSignal(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("run_linux.py is Linux-specific")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 unavailable: %v", err)
	}
	scriptPath, err := filepath.Abs("../run_linux.py")
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name        string
		sig         syscall.Signal
		ignoreTerm  bool
		secondDelay time.Duration
	}{
		{name: "SIGTERM terminates cooperative child", sig: syscall.SIGTERM, ignoreTerm: false},
		{name: "SIGHUP terminates cooperative child", sig: syscall.SIGHUP, ignoreTerm: false},
		{name: "SIGTERM escalates to SIGKILL when child ignores SIGTERM", sig: syscall.SIGTERM, ignoreTerm: true},
		{name: "second signal during cleanup still reaps", sig: syscall.SIGTERM, ignoreTerm: true, secondDelay: 100 * time.Millisecond},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := t.TempDir()
			pidFile := filepath.Join(tmp, "child.pid")
			outDir := filepath.Join(tmp, "cell")

			childCode := "import os, signal, time; "
			if tc.ignoreTerm {
				childCode += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
			}
			childCode += fmt.Sprintf("open(%q, 'w').write(str(os.getpid())); time.sleep(60)", pidFile)

			driverCode := fmt.Sprintf(`
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("run_linux", %q)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
status, interrupted, cleanup_ok, elapsed, peak_swap, samples = mod.run_monitored(
    [sys.executable, "-c", %q],
    pathlib.Path(%q),
    grace_seconds=0.3,
)
if interrupted is None and status == 0 and cleanup_ok:
    (pathlib.Path(%q) / "cpu.json.partial").rename(pathlib.Path(%q) / "cpu.json")
sys.exit(128 + interrupted if interrupted is not None else (1 if not cleanup_ok else (status if status >= 0 else 128 - status)))
`, scriptPath, childCode, outDir, outDir, outDir)

			cmd := exec.Command(py, "-c", driverCode)
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}

			var childPID int
			deadline := time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				raw, err := os.ReadFile(pidFile)
				if err == nil && len(strings.TrimSpace(string(raw))) > 0 {
					childPID, err = strconv.Atoi(strings.TrimSpace(string(raw)))
					if err == nil && childPID > 0 {
						break
					}
				}
				time.Sleep(20 * time.Millisecond)
			}
			if childPID <= 0 {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				t.Fatal("child process did not write its PID in time")
			}

			if err := cmd.Process.Signal(tc.sig); err != nil {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				t.Fatalf("signal recorder: %v", err)
			}
			if tc.secondDelay > 0 {
				time.Sleep(tc.secondDelay)
				if err := cmd.Process.Signal(tc.sig); err != nil && !strings.Contains(err.Error(), "process already finished") {
					t.Fatalf("second signal recorder: %v", err)
				}
			}
			waitErr := cmd.Wait()
			if waitErr == nil {
				t.Fatalf("expected non-zero exit after %v", tc.sig)
			}
			exitCode := cmd.ProcessState.ExitCode()
			if want := 128 + int(tc.sig); exitCode != want {
				t.Fatalf("recorder exit code = %d, want %d", exitCode, want)
			}

			// Child must already be dead and reaped before the recorder returned.
			if err := syscall.Kill(childPID, 0); err != syscall.ESRCH {
				_ = syscall.Kill(childPID, syscall.SIGKILL)
				t.Fatalf("child pid %d still alive after recorder exit (kill(0) err=%v)", childPID, err)
			}
			if _, err := os.Stat(filepath.Join(outDir, "cpu.json")); !os.IsNotExist(err) {
				t.Fatalf("interrupted run must not promote cpu.json.partial to cpu.json (stat err=%v)", err)
			}
		})
	}
}

func TestRunLinuxReapsGroupAfterLeaderExit(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("run_linux.py is Linux-specific")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 unavailable: %v", err)
	}
	scriptPath, err := filepath.Abs("../run_linux.py")
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	grandchildFile := filepath.Join(tmp, "grandchild.pid")
	outDir := filepath.Join(tmp, "cell")
	childCode := fmt.Sprintf(`
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(%q, "w").write(str(p.pid))
time.sleep(0.3)
`, grandchildFile)
	driverCode := fmt.Sprintf(`
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("run_linux", %q)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
status, interrupted, cleanup_ok, elapsed, peak_swap, samples = mod.run_monitored(
    [sys.executable, "-c", %q],
    pathlib.Path(%q),
    grace_seconds=0.5,
)
if interrupted is None and status == 0 and cleanup_ok:
    (pathlib.Path(%q) / "cpu.json.partial").rename(pathlib.Path(%q) / "cpu.json")
sys.exit(128 + interrupted if interrupted is not None else (1 if not cleanup_ok else (status if status >= 0 else 128 - status)))
`, scriptPath, childCode, outDir, outDir, outDir)
	cmd := exec.Command(py, "-c", driverCode)
	if err := cmd.Run(); err != nil {
		t.Fatalf("recorder: %v", err)
	}
	raw, err := os.ReadFile(grandchildFile)
	if err != nil {
		t.Fatal(err)
	}
	grandchildPID, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || grandchildPID <= 0 {
		t.Fatalf("grandchild pid %q: %v", raw, err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for !processStopped(grandchildPID) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !processStopped(grandchildPID) {
		_ = syscall.Kill(grandchildPID, syscall.SIGKILL)
		t.Fatalf("grandchild pid %d remained live after leader exit", grandchildPID)
	}
	if _, err := os.Stat(filepath.Join(outDir, "cpu.json")); err != nil {
		t.Fatalf("successful run did not promote cpu.json: %v", err)
	}
}

func TestGroupLivenessTreatsZombiesAsStoppedAndUnknownAsFailure(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("run_linux.py is Linux-specific")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 unavailable: %v", err)
	}
	scriptPath, err := filepath.Abs("../run_linux.py")
	if err != nil {
		t.Fatal(err)
	}
	driverCode := fmt.Sprintf(`
import importlib.util, os, pathlib, shutil, tempfile, types
spec = importlib.util.spec_from_file_location("run_linux", %q)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
root = pathlib.Path(tempfile.mkdtemp())
def stat(pid, state, pgrp):
    p = root / str(pid); p.mkdir()
    (p / "stat").write_text(f"{pid} (comm) {state} 1 {pgrp} 0 0 0")
    task = p / "task" / str(pid); task.mkdir(parents=True)
    (task / "stat").write_text(f"{pid} (comm) {state} 1 {pgrp} 0 0 0")
def set_state(pid, state):
    (root / str(pid) / "stat").write_text(f"{pid} (comm) {state} 1 42 0 0 0")
    (root / str(pid) / "task" / str(pid) / "stat").write_text(f"{pid} (comm) {state} 1 42 0 0 0")
stat(101, "R", 42)
assert mod._group_liveness(42, root) == mod.GROUP_LIVE
set_state(101, "Z")
assert mod._group_liveness(42, root) == mod.GROUP_STOPPED
(root / "101" / "task" / "1002").mkdir()
(root / "101" / "task" / "1002" / "stat").write_text("1002 (worker) S 101 42 0 0 0")
assert mod._group_liveness(42, root) == mod.GROUP_LIVE
shutil.rmtree(root / "101" / "task" / "1002")
(root / "101" / "task" / "1003").mkdir()
assert mod._group_liveness(42, root) == mod.GROUP_UNKNOWN
shutil.rmtree(root / "101" / "task" / "1003")
(root / "101" / "stat").write_text("broken")
assert mod._group_liveness(42, root) == mod.GROUP_UNKNOWN
shutil.rmtree(root / "101")
(root / "102").mkdir()
(root / "102" / "stat").mkdir()
assert mod._group_liveness(42, root) == mod.GROUP_UNKNOWN

class Proc:
    def poll(self): return 0
    def terminate(self): pass
    def kill(self): pass
    def wait(self, timeout=None): return 0

real_liveness = mod._group_liveness
signals = []
mod._signal_group = lambda *_: signals.append(1) or True
for status, visible, ok in ((mod.GROUP_LIVE, False, False), (mod.GROUP_UNKNOWN, False, True),
                            (mod.GROUP_UNKNOWN, True, False),
                            (mod.GROUP_STOPPED, False, True), (mod.GROUP_STOPPED, True, True),
                            (mod.GROUP_STOPPED, None, False)):
    signals.clear()
    mod._group_liveness = lambda *_args, status=status: status
    mod._group_visible = lambda *_args, visible=visible: visible
    got_status, got_ok = mod._finish_group(Proc(), 42, 0.2, root)
    assert got_ok is ok, (status, visible, got_ok)
    assert got_status == 0
    if status == mod.GROUP_STOPPED and visible is True:
        assert signals, "all-zombie snapshot was promoted without termination/recheck"
class CountProc(Proc):
    def poll(self): return None
    def __init__(self):
        self.terminated = 0
        self.killed = 0
    def terminate(self): self.terminated += 1
    def kill(self): self.killed += 1

mod._group_liveness = lambda *_args: mod.GROUP_STOPPED
mod._group_visible = lambda *_args: True
mod._signal_group = lambda *_: True
p = CountProc()
mod._finish_group(p, 42, 0.2, root)
assert p.terminated == 0 and p.killed == 0, (p.terminated, p.killed)
mod._signal_group = lambda *_: False
p = CountProc()
mod._finish_group(p, 42, 0.05, root)
assert p.terminated == 1 and p.killed == 1, (p.terminated, p.killed)
mi = pathlib.Path(tempfile.mkdtemp()) / "mountinfo"
mi.write_text("36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n")
assert mod._proc_complete(pathlib.Path("/proc"), mi) is True
for option in ("hidepid=1", "hidepid=2", "hidepid=noaccess", "hidepid=invisible", "hidepid=ptraceable", "hidepid=other"):
    mi.write_text("36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw," + option + "\n")
    assert mod._proc_complete(pathlib.Path("/proc"), mi) is False, option
for option in ("hidepid=0", "hidepid=off"):
    mi.write_text("36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw," + option + "\n")
    assert mod._proc_complete(pathlib.Path("/proc"), mi) is True, option
mi.write_text("36 35 98:0 / /other rw,nosuid,nodev,noexec,relatime - proc proc rw\n")
assert mod._proc_complete(pathlib.Path("/proc"), mi) is False
mi.write_text("36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n37 36 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n")
assert mod._proc_complete(pathlib.Path("/proc"), mi) is False
mi.write_bytes(b"1 2 3 4 /bad\xff rw - ext4 /dev/x rw\n36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n")
assert mod._proc_complete(pathlib.Path("/proc"), mi) is True
mi.write_bytes(b"1 2 3 4 /bad\xff rw - ext4 /dev/x rw\n36 35 98:0 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=2\n")
assert mod._proc_complete(pathlib.Path("/proc"), mi) is False
mod._group_liveness = real_liveness
hidden = pathlib.Path(tempfile.mkdtemp())
(hidden / "123").mkdir()
(hidden / "123" / "stat").write_text("123 (hidden) Z 1 42 0 0 0")
(hidden / "123" / "task" / "123").mkdir(parents=True)
(hidden / "123" / "task" / "123" / "stat").write_text("123 (hidden) Z 1 42 0 0 0")
mod._proc_complete = lambda *_args: False
assert mod._group_liveness(42, hidden) == mod.GROUP_UNKNOWN
mod._group_visible = lambda *_args: True
mod._signal_group = lambda *_: True
p = CountProc()
status, ok = mod._finish_group(p, 42, 0.05, hidden)
assert ok is False and status == 0
mod._proc_complete = lambda *_args: True
assert mod._group_liveness(42, hidden) == mod.GROUP_STOPPED
assert mod._should_promote(None, 0, False) is False
assert mod._should_promote(None, 0, True) is True
assert mod._should_promote(15, 0, True) is False
print("liveness-controls-ok")
`, scriptPath)
	out, err := exec.Command(py, "-c", driverCode).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "liveness-controls-ok") {
		t.Fatalf("liveness controls: %v\n%s", err, out)
	}
}

func processStopped(pid int) bool {
	if err := syscall.Kill(pid, 0); err == syscall.ESRCH {
		return true
	}
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return false
	}
	close := bytes.LastIndexByte(raw, ')')
	if close < 0 {
		return false
	}
	fields := bytes.Fields(raw[close+2:])
	return len(fields) > 0 && fields[0][0] == 'Z'
}

func TestRunLinuxPersistsCleanupFailure(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("run_linux.py is Linux-specific")
	}
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 unavailable: %v", err)
	}
	scriptPath, err := filepath.Abs("../run_linux.py")
	if err != nil {
		t.Fatal(err)
	}
	driverCode := fmt.Sprintf(`
import importlib.util, json, pathlib, sys, tempfile
spec = importlib.util.spec_from_file_location("run_linux", %q)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = pathlib.Path(tempfile.mkdtemp()) / "cell"
mod.host = lambda: {"load_1m": 0.0, "MemAvailable": 1 << 40}
def fake(command, output):
    output.mkdir()
    (output / "cpu.json.partial").write_text("partial")
    return 0, None, False, 0.1, 0, 0
mod.run_monitored = fake
sys.argv = ["run_linux.py", "--output", str(out), "--fit-gib", "1", "--physical-cores", "1", "--", "true"]
rc = mod.main()
meta = json.loads((out / "host.json").read_text())
assert rc == 1, rc
assert meta["cleanup_ok"] is False, meta
assert not (out / "cpu.json").exists()
assert (out / "cpu.json.partial").exists()
print("cleanup-meta-ok")
`, scriptPath)
	out, err := exec.Command(py, "-c", driverCode).CombinedOutput()
	if err != nil || !strings.Contains(string(out), "cleanup-meta-ok") {
		t.Fatalf("cleanup metadata: %v\n%s", err, out)
	}
}
