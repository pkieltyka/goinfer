// Command embeddinggemma2select selects one text route before an index write.
//
//	go run ./apps/examples/embeddinggemma2select \
//	    -checkpoint <dir> -model-pin <pin> -fingerprint <stored> [-selector cpu] [-device ""]
//
// This example links no GPU backend, so its executable route is cpu. An
// application that imports a backend can pass that backend's selector and an
// exact device ID or alias; see docs/use-from-go.md.
//
// The persisted fingerprint covers the caller-owned model/preprocessing pin,
// selector, verified numerical profile, output width and normalization. It does
// not cover the physical attachment, so a Metal registry ID that changes after
// reboot does not force a rebuild. The resolved device and backend are printed
// separately as provenance.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

type options struct {
	checkpoint  string
	modelPin    string
	fingerprint string
	selector    string
	deviceID    string
}

func main() {
	var opts options
	flag.StringVar(&opts.checkpoint, "checkpoint", "", "checkpoint directory")
	flag.StringVar(&opts.modelPin, "model-pin", "", "caller-owned model/tokenizer/prompt/preprocessing pin")
	flag.StringVar(&opts.fingerprint, "fingerprint", "", "stored route fingerprint")
	flag.StringVar(&opts.selector, "selector", "cpu", "route selector; this example links cpu only")
	flag.StringVar(&opts.deviceID, "device", "", "optional exact device ID or alias; empty uses the selector default")
	flag.Parse()
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(opts options) (err error) {
	if opts.checkpoint == "" || opts.modelPin == "" || opts.fingerprint == "" {
		return errors.New("checkpoint, model-pin and fingerprint are required")
	}
	if opts.selector != "cpu" {
		return fmt.Errorf("this example links no GPU backend, so selector %q is unavailable; use cpu, or import a backend in your application (see docs/use-from-go.md)", opts.selector)
	}
	encoder, err := embeddinggemma2.LoadEncoder(opts.checkpoint)
	if err != nil {
		return err
	}
	ids, err := encoder.Tokenize("first document", "document")
	if err != nil {
		return err
	}
	session, report, err := encoder.SelectText(embeddinggemma2.TextSelectionOptions{
		Targets:        []embeddinggemma2.TextTarget{{DeviceID: opts.deviceID, Selector: opts.selector}},
		AllowEstimates: true,
		AllowSplit:     true,
	}, [][]int{ids})
	if err != nil {
		for _, attempt := range report.Attempts {
			fmt.Fprintf(os.Stderr, "selection target %+v: %v\n", attempt.Target, attempt.Err)
		}
		return err
	}
	defer func() { err = errors.Join(err, session.Close()) }()
	if report.Winner < 0 || report.Winner >= len(report.Attempts) || len(session.First) == 0 {
		return errors.New("selection returned no first-group vectors")
	}
	attempt := report.Attempts[report.Winner]
	width := len(session.First[0])
	actual := routeFingerprint(opts.modelPin, attempt, width)
	if actual != opts.fingerprint {
		return fmt.Errorf("route fingerprint mismatch: have %q, want %q; refusing the first write", actual, opts.fingerprint)
	}
	// A real application writes session.First here, and only here. Device and
	// driver provenance is recorded separately from the numerical fingerprint.
	fmt.Printf("fingerprint=%s\nprovenance: backend=%s device=%q name=%q batched=%t memory_basis=%s first_vectors=%d\n",
		actual, attempt.Device.Backend, attempt.Device.ID, attempt.Device.Name, attempt.Batched,
		attempt.Memory.Basis, len(session.First))
	return nil
}

func routeFingerprint(modelPin string, attempt embeddinggemma2.TextSelectionAttempt, width int) string {
	return fmt.Sprintf("model=%s|selector=%s|profile=%s|width=%d|normalization=unit",
		modelPin, attempt.Target.Selector, attempt.Profile, width)
}
