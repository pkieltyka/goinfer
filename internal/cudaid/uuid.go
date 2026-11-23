// Package cudaid validates CUDA identifiers without accessing the driver.
package cudaid

import (
	"encoding/hex"
	"strings"
)

// ValidUUID reports whether token is a GPU- or MIG-prefixed CUDA UUID.
func ValidUUID(token string) bool {
	prefix, uuid, ok := strings.Cut(token, "-")
	if !ok || !strings.EqualFold(prefix, "GPU") && !strings.EqualFold(prefix, "MIG") {
		return false
	}
	if len(uuid) != 36 || uuid[8] != '-' || uuid[13] != '-' || uuid[18] != '-' || uuid[23] != '-' {
		return false
	}
	digits := strings.ReplaceAll(uuid, "-", "")
	_, err := hex.DecodeString(digits)
	return len(digits) == 32 && err == nil
}
