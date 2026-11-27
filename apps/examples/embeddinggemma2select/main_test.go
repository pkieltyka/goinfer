package main

import (
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestRouteFingerprintSeparatesAttachmentProvenance(t *testing.T) {
	attempt := embeddinggemma2.TextSelectionAttempt{
		Target:  embeddinggemma2.TextTarget{Selector: "metal-eg2-batch"},
		Profile: "metal-eg2-combined-v1",
		Device:  embeddinggemma2.DeviceInfo{ID: "metal:4294967296@boot-a", Backend: "metal"},
	}
	first := routeFingerprint("google-bf16-r1", attempt, 768)
	attempt.Device.ID = "metal:4294967296@boot-b"
	attempt.Device.Backend = "metal"
	if got := routeFingerprint("google-bf16-r1", attempt, 768); got != first {
		t.Fatalf("attachment identity changed the fingerprint: %q then %q", first, got)
	}
	if got := routeFingerprint("google-bf16-r2", attempt, 768); got == first {
		t.Fatal("model pin did not change the fingerprint")
	}
	attempt.Profile = "metal-eg2-combined-v2"
	if got := routeFingerprint("google-bf16-r1", attempt, 768); got == first {
		t.Fatal("numerical profile did not change the fingerprint")
	}
	if got := routeFingerprint("google-bf16-r1", attempt, 512); got == first {
		t.Fatal("width did not change the fingerprint")
	}
}
