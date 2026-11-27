//go:build cuda

package tower

import "time"

// The tower profile on CUDA: with
// profiling on, Ops drains its queue at every change of kernel class and attributes the host time since the
// previous drain to the class that was running. the CUDA Event wrapper has no elapsed-time call, so this is a wall-clock split
// with the queue drained at each class boundary, and the launch latency of the first kernel after a drain sits inside
// its class. With profiling off (Ops.prof nil) every hook is a nil check and nothing else changes.

type Class int

const (
	ClsGEMM Class = iota // the projections, with their bias and add epilogues
	ClsAttn              // attention
	ClsNorm              // LayerNorm and RMSNorm
	ClsElem              // RoPE, activations, adds, scales, position adds
	NClass
)

var ClassNames = [NClass]string{"gemm", "attention", "norm", "elementwise"}

type Prof struct {
	Acc         [NClass]time.Duration
	cur         Class
	last        time.Time
	open        bool
	GEMMFLOPs   float64
	AttnFLOPs   float64
	launchCount int
}

// Cls marks the start of an op of class c: a change of class drains the queue and closes the running class's interval.
func (t *Ops) Cls(c Class) {
	p := t.prof
	if p == nil {
		return
	}
	p.launchCount++
	if !p.open {
		p.open, p.cur, p.last = true, c, time.Now()
		return
	}
	if p.cur != c {
		t.profFlush()
		p.cur = c
	}
}

// profFlush drains the queue and charges the time since the last drain to the running class.
func (t *Ops) profFlush() {
	p := t.prof
	if p == nil || !p.open {
		return
	}
	_ = t.Q.Sync() // a launch error is latched in t.Err and reported by Finish
	now := time.Now()
	p.Acc[p.cur] += now.Sub(p.last)
	p.last = now
}

// ProfGEMM and profAttn add a call's FLOPs.
func (t *Ops) ProfGEMM(M, N, K int) {
	if t.prof != nil {
		t.prof.GEMMFLOPs += 2 * float64(M) * float64(N) * float64(K)
	}
}

func (t *Ops) profAttn(np, nH, hd int) {
	if t.prof != nil {
		t.prof.AttnFLOPs += 4 * float64(np) * float64(np) * float64(hd) * float64(nH)
	}
}

// StartProf attaches a fresh profile; StopProf detaches and returns it. A forward ends in Finish(), which closes the last interval on the executor thread (the CUDA context is bound to it,
// so StopProf does not drain from the caller's thread). Both run while no call is in flight.
func (t *Ops) StartProf() { t.prof = &Prof{} }

func (t *Ops) StopProf() *Prof {
	p := t.prof
	t.prof = nil
	return p
}
