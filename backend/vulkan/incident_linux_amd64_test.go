package vulkan

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

var incidentPath = flag.String("eg2-vulkan-incident", "", "path to the verified prepared-ID incident fixture")

func readIncidentIDs(t *testing.T) [][]int {
	t.Helper()
	body, err := os.ReadFile(*incidentPath)
	if err != nil {
		t.Fatal(err)
	}
	const pin = "eb1e4cbae09cb0f6fb161c264a0113581309decfe0fe3c27f9be3b132eb4c579"
	if hash := fmt.Sprintf("%x", sha256.Sum256(body)); hash != pin {
		t.Fatalf("incident fixture SHA-256 %s, want %s", hash, pin)
	}
	var fixture struct {
		IDs [][]int `json:"prepared_ids"`
	}
	if err := json.Unmarshal(body, &fixture); err != nil {
		t.Fatal(err)
	}
	want := slices.Concat(incidentHeadLengths, incidentTailLengths)
	if len(fixture.IDs) != len(want) {
		t.Fatalf("fixture contains %d inputs, want %d", len(fixture.IDs), len(want))
	}
	for i, ids := range fixture.IDs {
		if len(ids) != want[i] {
			t.Fatalf("input %d has %d positions, want %d", i, len(ids), want[i])
		}
	}
	t.Logf("incident fixture SHA-256 %s", pin)
	return fixture.IDs
}

func TestVulkanIncidentPreparedIDs(t *testing.T) {
	if *incidentPath == "" {
		t.Skip("set -eg2-vulkan-incident to the verified prepared-ID incident fixture")
	}
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		t.Skip("set GOINFER_EG2_DIR to the pinned official checkpoint")
	}
	id := requiredDevice(t)
	ids := readIncidentIDs(t)
	long := make([]int, 2048)
	note := ids[57]
	for i := 1; i+1 < len(long); i++ {
		long[i] = note[1+(i-1)%(len(note)-2)]
	}
	long[0], long[len(long)-1] = note[0], note[len(note)-1]
	for _, shape := range []struct {
		name    string
		ids     [][]int
		layouts [][]int
	}{
		{"subset", [][]int{ids[19], ids[22], ids[28], ids[38], ids[57]}, [][]int{
			{0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {0, 1, 2}, {3, 4},
		}},
		{"tail", ids[len(incidentHeadLengths):], nil},
		{"singleton-2048", [][]int{long}, nil},
		{"head", ids[:len(incidentHeadLengths)], nil},
	} {
		if !t.Run(shape.name, func(t *testing.T) {
			e := loadPinnedOfficial(t, dir)
			cpu, err := e.EmbedIDsBatch(shape.ids)
			if err != nil {
				t.Fatal(err)
			}
			a := attachParityDevice(t, e, id)
			singletons := make([][]float32, len(shape.ids))
			for i, original := range shape.ids {
				input := slices.Clone(original)
				before := a.device.Activity()
				single, err := e.EmbedIDs(input)
				if err != nil {
					t.Fatalf("singleton %d: %v", i, err)
				}
				checkedActivity(t, a, before, []int{len(input)}, false)
				if !slices.Equal(input, original) {
					t.Fatalf("singleton mutated input %d", i)
				}
				checkRows(t, fmt.Sprintf("input %d CPU parity", i), single, cpu[i], e.Model().Dim())
				if t.Failed() {
					return
				}
				var norm float64
				for j, v := range single {
					if delta := math.Abs(float64(v) - float64(cpu[i][j])); !(delta <= 1e-5) {
						t.Fatalf("singleton %d component %d CPU absolute error %g exceeds 1e-5", i, j, delta)
					}
					norm += float64(v) * float64(v)
				}
				if !(math.Abs(norm-1) < 1e-6) {
					t.Fatalf("singleton %d squared norm %g", i, norm)
				}
				singletons[i] = single
			}
			t.Logf("%d singleton vectors in input order, pooled SHA-256 %x", len(singletons), pooledDigest(singletons))
			layouts := shape.layouts
			if layouts == nil {
				order := make([]int, len(shape.ids))
				for i := range order {
					order[i] = i
				}
				layouts = [][]int{order}
			}
			for i, order := range layouts {
				checkIncidentLayout(t, e, a, shape.ids, singletons, order, i)
				if t.Failed() {
					return
				}
			}
		}) {
			return
		}
	}
}

func checkIncidentLayout(t *testing.T, e *embeddinggemma2.Encoder, a *encoder, original [][]int, singletons [][]float32, order []int, layout int) {
	t.Helper()
	inputs := make([][]int, len(order))
	lengths := make([]int, len(order))
	for i, input := range order {
		ids := original[input]
		inputs[i], lengths[i] = slices.Clone(ids), len(ids)
	}
	before := a.device.Activity()
	got, err := e.EmbedIDsBatch(inputs)
	if err != nil {
		t.Fatal(err)
	}
	checkedActivity(t, a, before, lengths, false)
	if len(got) != len(inputs) {
		t.Fatalf("returned %d vectors, want %d", len(got), len(inputs))
	}
	for i, input := range order {
		if !slices.Equal(inputs[i], original[input]) {
			t.Fatalf("batch mutated input %d at position %d", input, i)
		}
		checkPooledBits(t, fmt.Sprintf("layout %d batch slot %d input %d", layout, i, input), got[i], singletons[input])
	}
	if !t.Failed() {
		t.Logf("layout %d: %d bit-identical vectors in request order, pooled SHA-256 %x", layout, len(got), pooledDigest(got))
	}
}
