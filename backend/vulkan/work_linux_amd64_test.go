package vulkan

import (
	"testing"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

var incidentHeadLengths = []int{
	77, 70, 58, 67, 66, 60, 51, 71, 53, 64, 57, 58, 61, 76, 80, 69, 61, 62, 80, 60,
	63, 60, 65, 53, 56, 56, 68, 69, 66, 22, 311, 156, 305, 262, 22, 408, 93, 233,
	310, 228, 135, 293, 244, 25, 220, 356, 111, 235, 23, 335, 286, 141, 325, 444, 332,
}

var incidentTailLengths = []int{483, 79, 524, 473}

func TestVulkanWorkPlan(t *testing.T) {
	for _, shape := range []struct {
		name         string
		lengths      []int
		rows         int
		full, window int
	}{
		{"incident-head", incidentHeadLengths, 7712, 1816150, 1816150},
		{"incident-tail", incidentTailLengths, 1559, 737835, 737703},
		{"singleton-2048", []int{2048}, 2048, 4194304, 1836544},
	} {
		t.Run(shape.name, func(t *testing.T) {
			checkBlocks(t, shape.lengths, 4, 512, shape.rows, shape.full)
			checkBlocks(t, shape.lengths, 4, 256, 512, shape.window)
			a := &encoder{
				dispatchBudget: 4096,
				cfg: embeddinggemma2.Config{
					Layers: 24, Hidden: 512, Intermediate: 2048, PLEDim: 512,
					Heads: 4, EmbeddingDim: 768, SlidingWindow: 512,
				},
			}
			for i := range a.cfg.Layers {
				l := layer{rotation: rotation{dim: 256}, kv: 2}
				if i%6 == 5 {
					l.full, l.dim, l.kv = true, 512, 1
				}
				a.layers = append(a.layers, l)
			}
			f := forward{a: a, rows: shape.rows, lengths: shape.lengths, maxGroupX: 65535}
			plan := f.plan(false)
			if plan.err != nil || len(plan.groups) < ceilDiv(shape.rows, 128) {
				t.Fatalf("expected at least one group per 128 PLE rows, got %d groups: %v", len(plan.groups), plan.err)
			}
			var largest uint64
			for i, group := range plan.groups {
				if group.units == 0 || group.units > 805306368 || group.dispatches > 4096 {
					t.Fatalf("group %d: dispatches=%d units=%d", i, group.dispatches, group.units)
				}
				largest = max(largest, group.units)
			}
			t.Logf("%d rows: %d submissions, largest %d estimated work units", shape.rows, len(plan.groups), largest)
		})
	}
}

func checkBlocks(t *testing.T, lengths []int, heads, dim, radius, wantPairs int) {
	t.Helper()
	sequence, start, pairs := 0, 0, 0
	q := 0
	err := attentionBlocks(lengths, heads, dim, radius, func(b attentionBlock) {
		n := lengths[sequence]
		if b.sequence != start || b.query != q || b.rows < 1 || b.rows > 128 || b.query+b.rows > start+n {
			t.Fatalf("query coverage: got %+v for sequence %d at packed row %d", b, sequence, q)
		}
		first, end := max(start, q-radius), min(start+n, q+b.rows+radius)
		if b.key != first || b.key+b.keys != end {
			t.Fatalf("key tile [%d,%d), want [%d,%d)", b.key, b.key+b.keys, first, end)
		}
		if b.scores(heads, dim) > 805306368 || b.values(heads, dim) > 805306368 || b.softmax(heads) > 805306368 {
			t.Fatalf("attention block %+v exceeds the work cap", b)
		}
		for row := q; row < q+b.rows; row++ {
			pairs += min(end, row+radius+1) - max(first, row-radius)
		}
		q += b.rows
		if q == start+n {
			sequence++
			start += n
		}
	})
	if err != nil || sequence != len(lengths) || pairs != wantPairs {
		t.Fatalf("radius %d: covered %d/%d sequences and %d/%d pairs: %v", radius, sequence, len(lengths), pairs, wantPairs, err)
	}
}

func TestVulkanWorkPlanRejectsOversizeDispatch(t *testing.T) {
	plan := workPlan{budget: 1}
	never := func(*vk.Commands) error { t.Fatal("planning executed work"); return nil }
	plan.add("first", 1, true, never)
	plan.add("second", 1, true, never)
	plan.add("too large", submissionWork+1, true, never)
	if len(plan.groups) != 2 || plan.err == nil {
		t.Fatalf("invalid dispatch plan: %+v", plan)
	}
}
