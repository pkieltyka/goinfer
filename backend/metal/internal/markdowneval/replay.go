package markdowneval

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

const replayManifestSHA = "52dbe674fe616b75a81eb589e3d5db4314f6a95790c27eef984d3bffd87fefdb"

type ReplayPreparation struct {
	Documents   []InputPin `json:"documents"`
	GroupTokens []int      `json:"group_tokens"`
}

type Replay struct {
	Version   int                          `json:"version"`
	CorpusSHA string                       `json:"corpus_sha256"`
	Groups    [][]string                   `json:"groups"`
	Prepared  map[string]ReplayPreparation `json:"prepared"`
	Passages  []Passage                    `json:"-"`
}

// LoadReplay reads the ordered Markdown and preparation pins shared by backend tests.
func LoadReplay(dir string) (Replay, error) {
	meta, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return Replay{}, err
	}
	if fmt.Sprintf("%x", sha256.Sum256(meta)) != replayManifestSHA {
		return Replay{}, fmt.Errorf("replay-v1 manifest digest mismatch")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "corpus.md"))
	if err != nil {
		return Replay{}, err
	}
	var f Replay
	if err := json.Unmarshal(meta, &f); err != nil {
		return f, err
	}
	if f.Version != 1 || f.CorpusSHA != fmt.Sprintf("%x", sha256.Sum256(raw)) {
		return f, fmt.Errorf("replay version or corpus digest mismatch")
	}
	f.Passages, err = parsePassages(string(raw))
	if err != nil {
		return f, err
	}
	if len(f.Passages) != 219 || len(f.Groups) != 55 || len(f.Prepared) != 2 {
		return f, fmt.Errorf("replay needs 219 documents in 55 groups and two model preparations")
	}
	ids := make([]string, len(f.Passages))
	for i, p := range f.Passages {
		ids[i] = fmt.Sprintf("watch-%03d", i+1)
		if p.ID != ids[i] {
			return f, fmt.Errorf("replay document %d has ID %q, want %q", i, p.ID, ids[i])
		}
	}
	for i, group := range f.Groups {
		if !slices.Equal(group, ids[i*4:min(i*4+4, len(ids))]) {
			return f, fmt.Errorf("replay group %d changes the document order", i)
		}
	}
	for _, kind := range []string{"q8_0", "mlx8"} {
		pins := f.Prepared[kind]
		if err := checkInputs(pins.Documents, ids); err != nil {
			return f, fmt.Errorf("%s: %w", kind, err)
		}
		if len(pins.GroupTokens) != len(f.Groups) {
			return f, fmt.Errorf("%s: wrong group token count", kind)
		}
		for i, group := range f.Groups {
			var tokens int
			for _, pin := range pins.Documents[i*4 : i*4+len(group)] {
				tokens += pin.Tokens
			}
			if tokens != pins.GroupTokens[i] || tokens > 2748 {
				return f, fmt.Errorf("%s: group %d has %d tokens, manifest %d, cap 2748", kind, i, tokens, pins.GroupTokens[i])
			}
		}
	}
	return f, nil
}
