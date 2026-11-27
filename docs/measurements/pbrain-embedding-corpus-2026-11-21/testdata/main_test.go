package main

import (
	"math"
	"testing"
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
