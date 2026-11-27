//go:build cuda && goinfer_testhooks

package eg2

import (
	"os"
	"testing"

	"github.com/pkieltyka/goinfer/backend/cuda/internal/tower"
)

func TestMain(m *testing.M) {
	guard := tower.InstallLedgerGuard(os.Stderr)
	os.Exit(guard.Check(m.Run()))
}
