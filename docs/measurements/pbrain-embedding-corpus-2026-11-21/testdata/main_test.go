package main

import (
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
		name       string
		sig        syscall.Signal
		ignoreTerm bool
	}{
		{name: "SIGTERM terminates cooperative child", sig: syscall.SIGTERM, ignoreTerm: false},
		{name: "SIGHUP terminates cooperative child", sig: syscall.SIGHUP, ignoreTerm: false},
		{name: "SIGTERM escalates to SIGKILL when child ignores SIGTERM", sig: syscall.SIGTERM, ignoreTerm: true},
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
status, interrupted, elapsed, peak_swap, samples = mod.run_monitored(
    [sys.executable, "-c", %q],
    pathlib.Path(%q),
    grace_seconds=0.3,
)
if interrupted is None and status == 0:
    (pathlib.Path(%q) / "cpu.json.partial").rename(pathlib.Path(%q) / "cpu.json")
sys.exit(128 + interrupted if interrupted is not None else (status if status >= 0 else 128 - status))
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
