//go:build darwin

package metal

import (
	"testing"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func TestEG2StagedFFNEligibility(t *testing.T) {
	c := embeddinggemma2.Config{Hidden: 512, Intermediate: 2048, Layers: 24}
	if !eg2StagedFFNEligible("Apple M4", "applegpu_g16g", eg2MatrixProfile, c) {
		t.Fatal("measured device, graph and profile rejected")
	}
	for _, test := range []struct {
		name, device, arch, profile string
		cfg                         embeddinggemma2.Config
	}{
		{"unknown arch", "Apple M4", "", eg2MatrixProfile, c},
		{"other arch", "Apple M4", "applegpu_g15g", eg2MatrixProfile, c},
		{"other model", "Apple M4 Pro", "applegpu_g16g", eg2MatrixProfile, c},
		{"stock", "Apple M4", "applegpu_g16g", eg2StockProfile, c},
		{"other graph", "Apple M4", "applegpu_g16g", eg2MatrixProfile, embeddinggemma2.Config{Hidden: 512, Intermediate: 4096, Layers: 24}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if eg2StagedFFNEligible(test.device, test.arch, test.profile, test.cfg) {
				t.Fatal("unqualified graph or device admitted")
			}
		})
	}
	for _, tt := range []struct {
		rows int
		want bool
	}{{1, false}, {30, false}, {31, true}, {36, true}, {879, true}, {2042, true}, {2043, false}, {8192, false}} {
		if got := eg2StagedFFNRows(tt.rows); got != tt.want {
			t.Errorf("rows %d: staged=%t, want %t", tt.rows, got, tt.want)
		}
	}
}
