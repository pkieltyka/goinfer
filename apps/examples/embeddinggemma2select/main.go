// Command embeddinggemma2select selects one text route before an index write.
//
//	go run ./apps/examples/embeddinggemma2select <checkpoint-dir> <route-fingerprint> <selector> [device-id]
//
// The fingerprint is the caller's stored value, for example
// "metal-eg2-batch|metal:4294967296@BOOT|<profile>". A different resolved route
// is refused before any first-group vector is returned or written.
package main

import (
	"fmt"
	"os"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func main() {
	if len(os.Args) < 4 || len(os.Args) > 5 {
		fmt.Fprintln(os.Stderr, "usage: embeddinggemma2select <checkpoint-dir> <route-fingerprint> <selector> [device-id]")
		os.Exit(2)
	}
	dir, wantFingerprint, selector := os.Args[1], os.Args[2], os.Args[3]
	deviceID := ""
	if len(os.Args) == 5 {
		deviceID = os.Args[4]
	}
	encoder, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		fatal(err)
	}
	ids, err := encoder.Tokenize("first document", "document")
	if err != nil {
		fatal(err)
	}
	session, report, err := encoder.SelectText(embeddinggemma2.TextSelectionOptions{
		Targets:        []embeddinggemma2.TextTarget{{DeviceID: deviceID, Selector: selector}},
		AllowEstimates: true,
		AllowSplit:     true,
	}, [][]int{ids})
	if err != nil {
		fmt.Fprintf(os.Stderr, "selection attempts: %d\n", len(report.Attempts))
		for _, attempt := range report.Attempts {
			fmt.Fprintf(os.Stderr, "  %+v: %v\n", attempt.Target, attempt.Err)
		}
		fatal(err)
	}
	defer session.Close()
	if report.Winner < 0 || report.Winner >= len(report.Attempts) {
		fatal(fmt.Errorf("selection returned no winner"))
	}
	attempt := report.Attempts[report.Winner]
	actual := routeFingerprint(attempt)
	if actual != wantFingerprint {
		fatal(fmt.Errorf("route fingerprint mismatch: have %q, want %q; refusing the first write", actual, wantFingerprint))
	}
	// A real application writes session.First here, and only here: the route
	// fingerprint matched, so the index fingerprint can record this route.
	fmt.Printf("route=%s profile=%q device=%q batched=%t first_vectors=%d\n",
		selector, attempt.Profile, attempt.Device.ID, attempt.Batched, len(session.First))
}

func routeFingerprint(attempt embeddinggemma2.TextSelectionAttempt) string {
	return fmt.Sprintf("%s|%s|%s", attempt.Target.Selector, attempt.Device.ID, attempt.Profile)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
