//go:build darwin

package gpudev

import (
	"errors"
	"testing"

	"github.com/ebitengine/purego/objc"
)

// TestSelectAndRetainOwnsExactlyOneMember covers the ownership decision without
// a Metal device: one snapshot, one retain, and no retain when selection fails.
func TestSelectAndRetainOwnsExactlyOneMember(t *testing.T) {
	facts := []DeviceFact{
		{Name: "GPU A", Unified: true, RegistryID: 11},
		{Name: "GPU B", Unified: false, RegistryID: 22},
	}
	ids := []objc.ID{101, 202}
	for _, tc := range []struct {
		name    string
		choose  func([]DeviceFact) (int, error)
		wantID  objc.ID
		wantErr error
		retains int
	}{
		{
			name: "second member",
			choose: func(got []DeviceFact) (int, error) {
				if len(got) != len(facts) || got[1] != facts[1] {
					t.Errorf("choose saw %+v", got)
				}
				return 1, nil
			},
			wantID:  202,
			retains: 1,
		},
		{
			name:    "selection error",
			choose:  func([]DeviceFact) (int, error) { return 0, errors.New("no match") },
			wantErr: errors.New("no match"),
		},
		{
			name:    "out of range",
			choose:  func([]DeviceFact) (int, error) { return 2, nil },
			wantErr: ErrNoDevice,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			retains := 0
			id, fact, err := selectAndRetain(facts, ids, tc.choose, func(id objc.ID) objc.ID {
				retains++
				return id
			})
			if tc.wantErr != nil {
				if err == nil || err.Error() != tc.wantErr.Error() && !errors.Is(err, tc.wantErr) {
					t.Fatalf("error = %v, want %v", err, tc.wantErr)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if retains != tc.retains {
				t.Errorf("retains = %d, want %d", retains, tc.retains)
			}
			if tc.retains == 0 {
				if id != 0 || fact != (DeviceFact{}) {
					t.Errorf("failed selection returned id %d, fact %+v", id, fact)
				}
				return
			}
			if id != tc.wantID || fact != facts[1] {
				t.Errorf("selected id %d fact %+v, want id %d fact %+v", id, fact, tc.wantID, facts[1])
			}
		})
	}
}
