package embeddinggemma2

import (
	"fmt"
	"testing"

	"github.com/pkieltyka/goinfer/internal/audio"
	"github.com/pkieltyka/goinfer/internal/vision"
	"github.com/pkieltyka/goinfer/multimodal"
)

type selectTestAudio struct{ closed *int }

func (selectTestAudio) Name() string                             { return "fake-selected-towers" }
func (selectTestAudio) Blocks([]float32, int) ([]float32, error) { return nil, nil }
func (a selectTestAudio) Close() error                           { *a.closed++; return nil }

func TestNamedDeviceLeavesTowersOnCPU(t *testing.T) {
	const backend = "fake-selected-towers"
	registerTempAccelerator(t, backend, fakeText{backend})
	registerTempAccelerator(t, backend+":registered", fakeText{backend})
	RegisterSelectedAccelerator(backend, func(_ *Model, token string) (Accelerator, error) {
		if token != "1" {
			return nil, fmt.Errorf("unexpected device %q", token)
		}
		return fakeText{backend}, nil
	})
	var visionOpens, audioOpens, visionCloses, audioCloses int
	RegisterVisionAccelerator(backend, func(*vision.Gemma4Encoder) (VisionAccelerator, error) {
		visionOpens++
		return fakeTower{&visionCloses}, nil
	})
	RegisterAudioAccelerator(backend, func(*audio.Gemma4AudioEncoder) (AudioAccelerator, error) {
		audioOpens++
		return selectTestAudio{&audioCloses}, nil
	})
	t.Cleanup(func() {
		multimodal.UnregisterGemma4Tower(backend)
		accelMu.Lock()
		delete(audAccels, backend)
		delete(selectedAccels, backend)
		accelMu.Unlock()
	})

	e := &Encoder{m: &Model{}}
	if _, err := e.UseAccelerator(backend + ":1"); err != nil {
		t.Fatal(err)
	}
	e.vis, e.aud = &visionTower{}, &audioTower{}
	e.bindVisionAccel()
	e.bindAudioAccel()
	if visionOpens != 0 || audioOpens != 0 || e.VisionDevice() != selectedDeviceTowerReason || e.AudioDevice() != selectedDeviceTowerReason {
		t.Fatalf("late towers: vision %q, audio %q; factory calls %d, %d", e.VisionDevice(), e.AudioDevice(), visionOpens, audioOpens)
	}
	if _, err := e.UseAccelerator(backend); err != nil {
		t.Fatal(err)
	}
	if visionOpens != 1 || audioOpens != 1 || e.VisionDevice() != backend || e.AudioDevice() != backend {
		t.Fatalf("bare selector: vision %q, audio %q; factory calls %d, %d", e.VisionDevice(), e.AudioDevice(), visionOpens, audioOpens)
	}
	if _, err := e.UseAccelerator(backend + ":unused"); err == nil || e.VisionDevice() != backend || e.AudioDevice() != backend {
		t.Fatalf("failed device selection: error %v, vision %q, audio %q", err, e.VisionDevice(), e.AudioDevice())
	}
	if _, err := e.UseAccelerator(backend + ":1"); err != nil {
		t.Fatal(err)
	}
	if visionCloses != 1 || audioCloses != 1 || e.VisionAccelerator() != nil || e.AudioAccelerator() != nil ||
		e.VisionDevice() != selectedDeviceTowerReason || e.AudioDevice() != selectedDeviceTowerReason {
		t.Fatalf("explicit selector: vision %q, audio %q; close calls %d, %d", e.VisionDevice(), e.AudioDevice(), visionCloses, audioCloses)
	}
	if _, err := e.UseAccelerator(backend + ":registered"); err != nil {
		t.Fatal(err)
	}
	if e.VisionDevice() != backend || e.AudioDevice() != backend || visionOpens != 2 || audioOpens != 2 {
		t.Fatalf("exact registry name: vision %q, audio %q; factory calls %d, %d", e.VisionDevice(), e.AudioDevice(), visionOpens, audioOpens)
	}
}
