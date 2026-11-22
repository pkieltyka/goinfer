package embeddinggemma2

import "slices"

// AcceleratorChoice reports the encoder's selected backend and any initialization
// errors encountered while choosing it. Backend is the attached accelerator's
// Name, or "cpu" when no GPU was attached. On CPU, Declined explains any failed
// attempts to attach a GPU.
type AcceleratorChoice struct {
	Backend  string
	Selector string // registry name used by this call, empty on CPU or an existing attachment
	Declined []AcceleratorDecline
}

// AcceleratorDecline records why a requested backend could not attach.
type AcceleratorDecline struct {
	Selector string
	Err      error
}

// UseAutoAccelerator attaches the first usable accelerator and otherwise selects
// CPU. With no preference it tries the linked CUDA and Vulkan accelerators in
// that order. Pass backend names to supply a different order or try other
// registered accelerators. An explicit "cpu" ends the search. Call before
// concurrent use. An existing attachment is kept regardless of preference.
// Errors during embedding are returned by the selected backend.
func (e *Encoder) UseAutoAccelerator(preferred ...string) AcceleratorChoice {
	if e.accel != nil {
		return AcceleratorChoice{Backend: e.accel.Name()}
	}

	choice := AcceleratorChoice{Backend: "cpu"}
	if len(preferred) == 0 {
		linked := Accelerators()
		for _, name := range []string{"cuda", "vulkan"} {
			if slices.Contains(linked, name) {
				preferred = append(preferred, name)
			}
		}
	}
	for _, name := range preferred {
		if name == "cpu" {
			break
		}
		a, err := e.UseAccelerator(name)
		if err != nil {
			choice.Declined = append(choice.Declined, AcceleratorDecline{Selector: name, Err: err})
			continue
		}
		choice.Backend, choice.Selector = a.Name(), name
		break
	}
	return choice
}
