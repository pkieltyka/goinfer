//go:build cuda

package tower

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	guard := InstallLedgerGuard()
	os.Exit(guard.Check(m.Run()))
}
