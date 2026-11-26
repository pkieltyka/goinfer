//go:build cuda

package cuda

// SelfTestDeviceError reports the device-open error for decoder's standalone startup self-test
// (`check --hardware`), or nil when a device opens and can be released cleanly. It exists only for that
// standalone path: a normal load opens the device in BuildResident, so loaded models never pay a second
// context open. The driver's own error is returned unchanged, so the report can show it.
func (b *cudaBackend) SelfTestDeviceError() error {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		return err
	}
	dev.ReleaseObjects()
	return nil
}
