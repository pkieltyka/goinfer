package embeddinggemma2

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// ErrSelectionFit means a permitted text target was refused before its forward
// because the batch estimate could not be shown to fit the caller's cap or the
// attached device's known headroom. It is distinct from a native device OOM,
// which is never retried as a different route by SelectText.
var ErrSelectionFit = fmt.Errorf("%w: selection fit refused", ErrDeviceUnavailable)

// errSelectionCleanup marks a losing attachment whose Close failed. The trial
// stops; it never advances to another target after a cleanup failure.
var errSelectionCleanup = errors.New("embeddinggemma2: selection cleanup failed")

// errSelectionOutput marks a forward whose vectors violate the embedding
// output contract. It is terminal: a different route could hide the defect.
var errSelectionOutput = errors.New("embeddinggemma2: selection output invalid")

// TextTarget names one permitted numerical route. Selector is a registered
// accelerator or device selector, or "cpu". DeviceID is an optional exact
// backend-qualified ID or current alias; empty means the selector's documented
// default device. RequiredProfile, when non-empty, must equal the attached
// accelerator's reported profile.
type TextTarget struct {
	DeviceID        string
	Selector        string
	RequiredProfile string
}

// TextSelectionOptions orders the routes SelectText may try. MaxAdditionalBytes
// caps additional device bytes for a forward; a pointer to zero is a real cap.
// AllowEstimates permits approximate allocation and headroom figures to admit a
// batch. AllowSplit permits contiguous whole-document subgroups when a batch is
// refused or a forward certifies a retry-safe out-of-memory failure.
type TextSelectionOptions struct {
	Targets            []TextTarget
	MaxAdditionalBytes *int64
	AllowEstimates     bool
	AllowSplit         bool
}

// TextSelectionAttempt records one target in caller order. Device, Profile,
// Memory and Batched describe the resolved route, with empty or unknown values
// where the backend cannot report them. FittedLengths and Estimates list each
// executed subgroup in order. Err is the raw cause when the target was refused.
type TextSelectionAttempt struct {
	Target        TextTarget
	Device        DeviceInfo
	Profile       string
	ProfileKnown  bool
	Batched       bool
	Memory        DeviceMemory
	MemoryErr     error
	FittedLengths [][]int
	Estimates     []BatchMemoryEstimate
	EstimateErr   error
	Err           error
}

// TextSelectionReport is returned with both success and a terminal error.
// Winner indexes Attempts, or is -1 when no target was selected. Inventory and
// InventoryFailures preserve the #29 probe used to resolve explicit IDs.
type TextSelectionReport struct {
	Winner            int
	Attempts          []TextSelectionAttempt
	Inventory         []DeviceInfo
	InventoryFailures []ProbeFailure
}

// TextSession is one owned numerical route selected before an index write. It
// serializes later pricing, forwarding and Close. First holds the first group's
// completed vectors once. Later batches cannot change device or profile.
type TextSession struct {
	mu             sync.Mutex
	enc            *Encoder
	accel          Accelerator
	target         TextTarget
	device         DeviceInfo
	profile        string
	limitBytes     *int64
	allowEstimates bool
	allowSplit     bool
	closed         bool
	closeErr       error

	First [][]float32
}

// SelectText validates the permitted routes, then tries them in caller order on
// private encoders sharing this encoder's model, tokenizer and prompts. It
// returns an owned session pinned to the first route that admits and completes
// the first batch, together with the report. The receiver encoder and its
// attachment are never changed.
func (e *Encoder) SelectText(opts TextSelectionOptions, first [][]int) (*TextSession, TextSelectionReport, error) {
	report := TextSelectionReport{Winner: -1}
	if e == nil || e.m == nil {
		return nil, report, errors.New("embeddinggemma2: SelectText needs an encoder")
	}
	targets, limit, err := validateTextSelection(opts)
	if err != nil {
		return nil, report, err
	}
	lengths := make([]int, len(first))
	for i, row := range first {
		lengths[i] = len(row)
	}
	if _, _, err := validatePreparedLengths(lengths); err != nil {
		return nil, report, err
	}
	if err := e.m.checkPreparedLengthCap(lengths); err != nil {
		return nil, report, err
	}
	if err := validateTextTokenIDs(first, e.m.cfg.VocabSize); err != nil {
		return nil, report, err
	}
	if textSelectionNeedsInventory(targets) {
		report.Inventory, report.InventoryFailures = AllDevices(ProbeOptions{})
	}
	var causes []error
	for _, target := range targets {
		session, attempt := e.tryTextTarget(target, limit, opts, report.Inventory, first)
		report.Attempts = append(report.Attempts, attempt)
		if attempt.Err == nil && session != nil {
			report.Winner = len(report.Attempts) - 1
			return session, report, nil
		}
		causes = append(causes, attempt.Err)
		if !textSelectionMayAdvance(attempt.Err) {
			return nil, report, attempt.Err
		}
	}
	return nil, report, fmt.Errorf("%w: every permitted text target was refused: %w", ErrDeviceUnavailable, errors.Join(causes...))
}

func validateTextTokenIDs(ids [][]int, vocab int) error {
	for i, row := range ids {
		for _, id := range row {
			if id < 0 || id >= vocab {
				return fmt.Errorf("%w: input %d has token id %d out of range [0, %d)", ErrDeviceUnavailable, i, id, vocab)
			}
		}
	}
	return nil
}

func validateTextSelection(opts TextSelectionOptions) ([]TextTarget, *int64, error) {
	if len(opts.Targets) == 0 {
		return nil, nil, fmt.Errorf("%w: text selection needs at least one target", ErrDeviceUnavailable)
	}
	var limit *int64
	if opts.MaxAdditionalBytes != nil {
		if *opts.MaxAdditionalBytes < 0 {
			return nil, nil, fmt.Errorf("%w: additional byte cap %d is negative", ErrDeviceUnavailable, *opts.MaxAdditionalBytes)
		}
		value := *opts.MaxAdditionalBytes
		limit = &value
	}
	targets := make([]TextTarget, len(opts.Targets))
	copy(targets, opts.Targets)
	for i, target := range targets {
		if target.Selector == "" {
			return nil, nil, fmt.Errorf("%w: target %d has no selector", ErrDeviceUnavailable, i)
		}
		if target.Selector == "cpu" {
			if target.DeviceID != "" {
				return nil, nil, fmt.Errorf("%w: target %d selects CPU with a device ID %q", ErrDeviceUnavailable, i, target.DeviceID)
			}
			if target.RequiredProfile != "" {
				return nil, nil, fmt.Errorf("%w: target %d requires profile %q on CPU, which reports none", ErrDeviceUnavailable, i, target.RequiredProfile)
			}
		}
	}
	return targets, limit, nil
}

func textSelectionNeedsInventory(targets []TextTarget) bool {
	for _, target := range targets {
		if target.DeviceID != "" {
			return true
		}
	}
	return false
}

func textSelectionMayAdvance(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, errSelectionCleanup) || errors.Is(err, ErrAmbiguousDevice) || errors.Is(err, ErrDeviceLost) || errors.Is(err, ErrDeviceOutOfMemory) {
		return false
	}
	if errors.Is(err, ErrSelectionFit) || errors.Is(err, ErrUnsupportedBatchShape) ||
		errors.Is(err, ErrUnsupportedSelector) || errors.Is(err, ErrNoDevice) || errors.Is(err, ErrBackendUnavailable) {
		return true
	}
	return errors.Is(err, ErrDeviceUnavailable)
}

func (e *Encoder) tryTextTarget(target TextTarget, limit *int64, opts TextSelectionOptions, inventory []DeviceInfo, first [][]int) (*TextSession, TextSelectionAttempt) {
	attempt := TextSelectionAttempt{Target: target}
	enc, err := NewEncoder(e.m, e.tok, e.prompts)
	if err != nil {
		attempt.Err = err
		return nil, attempt
	}
	trial := &textTrial{
		enc:            enc,
		target:         target,
		limitBytes:     limit,
		allowEstimates: opts.AllowEstimates,
		allowSplit:     opts.AllowSplit,
		attempt:        attempt,
	}
	accel, device, err := trial.attach(inventory)
	if err != nil {
		trial.attempt.Err = err
		return nil, trial.attempt
	}
	trial.accel = accel
	trial.device = device
	trial.profile, trial.attempt.ProfileKnown = textAttachedProfile(accel)
	trial.attempt.Device = device
	trial.attempt.Profile = trial.profile
	_, trial.attempt.Batched = accel.(batchForwarder)
	trial.refreshMemory()
	if target.RequiredProfile != "" {
		if !trial.attempt.ProfileKnown || trial.profile != target.RequiredProfile {
			err := fmt.Errorf("%w: target requires profile %q, attached %q (known %t)", ErrUnsupportedSelector, target.RequiredProfile, trial.profile, trial.attempt.ProfileKnown)
			trial.attempt.Err = err
			closeTextTrial(trial, &trial.attempt)
			return nil, trial.attempt
		}
	}
	vectors, err := trial.run(first)
	trial.attempt = trial.snapshot()
	if err != nil {
		trial.attempt.Err = err
		closeTextTrial(trial, &trial.attempt)
		return nil, trial.attempt
	}
	trial.attempt.EstimateErr = nil
	session := &TextSession{
		enc:            enc,
		accel:          accel,
		target:         target,
		device:         device,
		profile:        trial.profile,
		limitBytes:     limit,
		allowEstimates: opts.AllowEstimates,
		allowSplit:     opts.AllowSplit,
		First:          vectors,
	}
	return session, trial.attempt
}

func textAttachedProfile(accel Accelerator) (string, bool) {
	reporter, ok := accel.(AttachedProfileReporter)
	if !ok {
		return "", false
	}
	return reporter.AttachedProfile(), true
}

func closeTextTrial(trial *textTrial, attempt *TextSelectionAttempt) {
	if trial.accel == nil {
		return
	}
	if err := trial.accel.Close(); err != nil {
		attempt.Err = errors.Join(attempt.Err, fmt.Errorf("%w: %w: close refused target: %w", ErrDeviceUnavailable, errSelectionCleanup, err))
	}
	trial.accel = nil
}

type textTrial struct {
	enc            *Encoder
	accel          Accelerator
	target         TextTarget
	device         DeviceInfo
	profile        string
	limitBytes     *int64
	allowEstimates bool
	allowSplit     bool
	memory         DeviceMemory
	memoryErr      error
	attempt        TextSelectionAttempt
}

func (t *textTrial) attach(inventory []DeviceInfo) (Accelerator, DeviceInfo, error) {
	if t.target.Selector == "cpu" {
		return nil, DeviceInfo{Backend: "cpu", Name: "cpu"}, nil
	}
	if t.target.DeviceID == "" {
		accel, err := t.enc.UseAccelerator(t.target.Selector)
		if err != nil {
			return nil, DeviceInfo{}, err
		}
		device := DeviceInfo{Backend: accel.Name()}
		if reporter, ok := accel.(AttachedDeviceReporter); ok {
			device = reporter.AttachedDevice()
			if device.Backend == "" {
				device.Backend = accel.Name()
			}
		}
		return accel, device, nil
	}
	row, err := resolveTextDevice(inventory, t.target.DeviceID)
	if err != nil {
		return nil, DeviceInfo{}, err
	}
	accel, err := t.enc.UseDevice(row.ID, t.target.Selector)
	if err != nil {
		return nil, DeviceInfo{}, err
	}
	reporter, ok := accel.(AttachedDeviceReporter)
	if !ok {
		return nil, DeviceInfo{}, rejectTextAttach(accel, fmt.Errorf("%w: %s cannot report the attached device identity", ErrDeviceUnavailable, t.target.Selector))
	}
	device := reporter.AttachedDevice()
	if device.ID == "" || device.ID != row.ID || (device.Backend != "" && device.Backend != row.Backend) {
		return nil, DeviceInfo{}, rejectTextAttach(accel, fmt.Errorf("%w: target resolved %q but attached %+v", ErrDeviceUnavailable, row.ID, device))
	}
	return accel, device, nil
}

func rejectTextAttach(accel Accelerator, cause error) error {
	if err := accel.Close(); err != nil {
		return errors.Join(cause, fmt.Errorf("%w: %w: close refused target: %w", ErrDeviceUnavailable, errSelectionCleanup, err))
	}
	return cause
}

func resolveTextDevice(inventory []DeviceInfo, id string) (DeviceInfo, error) {
	var matched []DeviceInfo
	for _, row := range inventory {
		if row.ID == id || (row.Alias != "" && row.Alias == id) {
			matched = append(matched, row)
		}
	}
	switch len(matched) {
	case 0:
		return DeviceInfo{}, fmt.Errorf("%w: no inventory row matches %q", ErrNoDevice, id)
	case 1:
		if matched[0].ID == "" {
			return DeviceInfo{}, fmt.Errorf("%w: inventory row %q has no exact ID", ErrNoDevice, id)
		}
		return matched[0], nil
	default:
		return DeviceInfo{}, fmt.Errorf("%w: inventory row %q is ambiguous", ErrAmbiguousDevice, id)
	}
}

func (t *textTrial) refreshMemory() {
	t.memory, t.memoryErr = DeviceMemory{}, nil
	if t.accel == nil {
		return
	}
	if reporter, ok := t.accel.(DeviceMemoryReporter); ok {
		t.memory, t.memoryErr = reporter.DeviceMemory()
	}
}

func (t *textTrial) snapshot() TextSelectionAttempt {
	attempt := t.attempt
	attempt.Target = t.target
	attempt.Device = t.device
	attempt.Profile = t.profile
	attempt.Memory = t.memory
	attempt.MemoryErr = t.memoryErr
	return attempt
}

func (t *textTrial) run(ids [][]int) ([][]float32, error) {
	t.refreshMemory()
	lengths := make([]int, len(ids))
	for i, row := range ids {
		lengths[i] = len(row)
	}
	estimate, err := t.enc.BatchMemory(lengths)
	if err != nil {
		t.attempt.EstimateErr = err
		if t.canSplit(ids, err) {
			return t.split(ids)
		}
		return nil, err
	}
	if err := t.admit(estimate); err != nil {
		t.attempt.EstimateErr = err
		if t.canSplit(ids, err) {
			return t.split(ids)
		}
		return nil, err
	}
	vectors, err := t.execute(ids)
	if err != nil {
		if errors.Is(err, ErrBatchOutOfMemory) && t.allowSplit && len(ids) > 1 {
			return t.split(ids)
		}
		return nil, err
	}
	if err := validateTextVectors(vectors, t.enc.m.cfg.EmbeddingDim); err != nil {
		return nil, err
	}
	t.attempt.FittedLengths = append(t.attempt.FittedLengths, lengths)
	t.attempt.Estimates = append(t.attempt.Estimates, estimate)
	return vectors, nil
}

func (t *textTrial) canSplit(ids [][]int, err error) bool {
	if !t.allowSplit || len(ids) < 2 {
		return false
	}
	return errors.Is(err, ErrUnsupportedBatchShape) || errors.Is(err, ErrSelectionFit)
}

func (t *textTrial) split(ids [][]int) ([][]float32, error) {
	mid := len(ids) / 2
	left, err := t.run(ids[:mid])
	if err != nil {
		return nil, err
	}
	right, err := t.run(ids[mid:])
	if err != nil {
		return nil, err
	}
	return append(left, right...), nil
}

func (t *textTrial) execute(ids [][]int) ([][]float32, error) {
	if len(ids) == 1 {
		vector, err := t.enc.EmbedIDs(ids[0])
		if err != nil {
			return nil, err
		}
		return [][]float32{vector}, nil
	}
	return t.enc.EmbedIDsBatch(ids)
}

func (t *textTrial) admit(estimate BatchMemoryEstimate) error {
	if t.target.Selector == "cpu" {
		return nil
	}
	limit, known := int64(0), false
	if t.limitBytes != nil {
		limit, known = *t.limitBytes, true
	}
	switch t.memory.Basis {
	case MemoryLive:
		if !known || t.memory.AvailableBytes < limit {
			limit, known = t.memory.AvailableBytes, true
		}
	case MemoryEstimated:
		if t.allowEstimates && (!known || t.memory.AvailableBytes < limit) {
			limit, known = t.memory.AvailableBytes, true
		}
	}
	if estimate.AdditionalPeakBytes < 0 {
		return fmt.Errorf("%w: negative additional peak %d", ErrSelectionFit, estimate.AdditionalPeakBytes)
	}
	if estimate.Quality == BatchMemoryUnknown {
		if t.limitBytes != nil {
			return fmt.Errorf("%w: unknown allocation estimate cannot certify a %d-byte cap", ErrSelectionFit, *t.limitBytes)
		}
		return nil // no cap: an observed trial is the only available evidence
	}
	if estimate.Quality == BatchMemoryApproximate && !t.allowEstimates {
		if t.limitBytes != nil {
			return fmt.Errorf("%w: approximate estimate %d needs AllowEstimates", ErrSelectionFit, estimate.AdditionalPeakBytes)
		}
		return nil // no cap: ignore an unpermitted approximate figure
	}
	if !known {
		return nil
	}
	if estimate.AdditionalPeakBytes > limit {
		return fmt.Errorf("%w: %s estimate %d exceeds limit %d", ErrSelectionFit, estimate.Quality, estimate.AdditionalPeakBytes, limit)
	}
	return nil
}

func validateTextVectors(vectors [][]float32, dim int) error {
	for i, vector := range vectors {
		if len(vector) != dim {
			return fmt.Errorf("%w: vector %d has width %d, want %d", errSelectionOutput, i, len(vector), dim)
		}
		var norm float64
		for _, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("%w: vector %d has a non-finite value", errSelectionOutput, i)
			}
			norm += float64(value) * float64(value)
		}
		if math.Abs(norm-1) > 1e-4 {
			return fmt.Errorf("%w: vector %d is not unit length (%v)", errSelectionOutput, i, norm)
		}
	}
	return nil
}

// EmbedIDsBatch runs later ids on the selected route. It reprices each executed
// subgroup, may split when the session allows it, and returns no partial result
// if any part fails. It never changes device or profile.
func (s *TextSession) EmbedIDsBatch(ids [][]int) ([][]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("%w: text session is closed", ErrDeviceUnavailable)
	}
	lengths := make([]int, len(ids))
	for i, row := range ids {
		lengths[i] = len(row)
	}
	if _, _, err := validatePreparedLengths(lengths); err != nil {
		return nil, err
	}
	if err := s.enc.m.checkPreparedLengthCap(lengths); err != nil {
		return nil, err
	}
	if err := validateTextTokenIDs(ids, s.enc.m.cfg.VocabSize); err != nil {
		return nil, err
	}
	trial := &textTrial{
		enc:            s.enc,
		accel:          s.accel,
		target:         s.target,
		device:         s.device,
		profile:        s.profile,
		limitBytes:     s.limitBytes,
		allowEstimates: s.allowEstimates,
		allowSplit:     s.allowSplit,
	}
	trial.refreshMemory()
	return trial.run(ids)
}

// Close releases the selected route. It is idempotent; a call on a closed
// session for embedding fails.
func (s *TextSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	if s.accel != nil {
		s.closeErr = s.accel.Close()
	}
	return s.closeErr
}
