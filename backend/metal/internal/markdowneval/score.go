package markdowneval

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

type Hit struct {
	ID    string  `json:"id"`
	Score float64 `json:"cosine"`
	Label string  `json:"label"`
}

type Ranking struct {
	ID                string  `json:"id"`
	Hits              []Hit   `json:"hits"`
	FirstRelevantRank int     `json:"first_relevant_rank"`
	RecallAt5         float64 `json:"recall_at_5"`
}

// Rank scores finite unit vectors using cosine similarity and breaks ties by ID.
func (f Fixture) Rank(docs, queries [][]float32, width int) ([]Ranking, error) {
	if len(docs) != len(f.Passages) || len(queries) != len(f.Queries) || len(docs) == 0 {
		return nil, fmt.Errorf("vectors do not cover the fixture")
	}
	for _, group := range [][][]float32{docs, queries} {
		for _, v := range group {
			if err := CheckVector(v, width); err != nil {
				return nil, err
			}
		}
	}
	results := make([]Ranking, 0, len(queries))
	for i, query := range queries {
		q := f.Queries[i]
		r := Ranking{ID: q.ID, Hits: make([]Hit, len(docs))}
		for j, doc := range docs {
			h := Hit{ID: f.Passages[j].ID, Label: "unjudged"}
			switch {
			case slices.Contains(q.Relevant, h.ID):
				h.Label = "relevant"
			case slices.Contains(q.Distractors, h.ID):
				h.Label = "distractor"
			}
			for k, x := range doc {
				h.Score += float64(x) * float64(query[k])
			}
			r.Hits[j] = h
		}
		slices.SortFunc(r.Hits, func(a, b Hit) int {
			if c := cmp.Compare(b.Score, a.Score); c != 0 {
				return c
			}
			return cmp.Compare(a.ID, b.ID)
		})
		for j, h := range r.Hits {
			if h.Label == "relevant" {
				if r.FirstRelevantRank == 0 {
					r.FirstRelevantRank = j + 1
				}
				if j < 5 {
					r.RecallAt5 += 1 / float64(len(q.Relevant))
				}
			}
		}
		results = append(results, r)
	}
	return results, nil
}

func CheckVector(v []float32, width int) error {
	if width <= 0 || len(v) != width {
		return fmt.Errorf("vector width %d, want %d", len(v), width)
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.IsNaN(sum) || math.Abs(sum-1) > 1e-4 {
		return fmt.Errorf("invalid squared L2 norm %g", sum)
	}
	return nil
}
