//go:build cuda && goinfer_testhooks

package tower

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	guard := InstallLedgerGuard(os.Stderr)
	os.Exit(guard.Check(m.Run()))
}
