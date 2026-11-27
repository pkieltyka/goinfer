//go:build cuda

package tower

import (
	"fmt"
	"runtime"
	"runtime/debug"
)

// PTXTargetMajor and PTXTargetMinor are the minimum compute capability for
// the default CUDA kernels. A newer card uses the driver's forward JIT.
const (
	PTXTargetMajor = 7
	PTXTargetMinor = 5
)

// RunJob executes one job on a pinned CUDA executor goroutine, turning a panic
// (e.g. a device allocation failure) into an error.
//
// A runtime.Error (a bug, not a device failure) comes back as an *ExecutorBug,
// which AckJob raises again on the calling goroutine: a bug must fail loudly
// there, not become a decline to a slower path. Any other panic value is
// wrapped with %w when it is an error, so mem.DeviceAllocFailed still sees the
// allocator's failure type.
func RunJob(j func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			// The STACK, not just the value: a recovered executor panic becomes a resident DECLINE printed once to stderr, and
			// an allocation failure with no frame is unactionable. The stack is a few KB on a path that runs at most once per
			// model load.
			stack := debug.Stack()
			switch v := p.(type) {
			case runtime.Error:
				err = &ExecutorBug{bug: v, stack: stack}
			case error:
				err = fmt.Errorf("cuda: executor job panicked: %w\n%s", v, stack)
			default:
				err = fmt.Errorf("cuda: executor job panicked: %v\n%s", p, stack)
			}
		}
	}()
	return j()
}

// ExecutorBug is a runtime.Error recovered on an executor goroutine, with that
// goroutine's stack. It is itself a runtime.Error, so every recover the
// re-raised panic meets on the calling side (mem.RepanicBug) lets it through too.
type ExecutorBug struct {
	bug   runtime.Error
	stack []byte
}

func (b *ExecutorBug) Error() string {
	return fmt.Sprintf("cuda: executor job panicked: %s\nexecutor goroutine stack:\n%s", b.bug.Error(), b.stack)
}

func (b *ExecutorBug) RuntimeError() {}

func (b *ExecutorBug) Unwrap() error { return b.bug }

// AckJob is a job's result as Do returns it: an *ExecutorBug panics on the
// calling goroutine, anything else is returned.
func AckJob(err error) error {
	if b, ok := err.(*ExecutorBug); ok {
		panic(b)
	}
	return err
}
