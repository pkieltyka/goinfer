package embeddinggemma2

import "fmt"

// ErrBatchOutOfMemory reports a device out-of-memory failure during a prepared
// EmbedIDs or EmbedIDsBatch forward after the accelerator has certified
// recovery: any queued device work drained cleanly, all transient allocations
// from the failed attempt were released, resident weights remain intact (any
// evicted derived caches can be rebuilt on demand), and the encoder remains
// usable for an explicit caller retry with a smaller subset of the same ordered
// inputs.
//
// ErrBatchOutOfMemory wraps ErrDeviceOutOfMemory, so errors.Is matches both
// ErrBatchOutOfMemory (retry-safe) and ErrDeviceOutOfMemory (common device
// OOM). A failed attach or weight upload, or an unrecovered forward OOM,
// matches ErrDeviceOutOfMemory alone (or a fatal/device-lost error) and never
// matches ErrBatchOutOfMemory.
//
// Only the backend completing recovery may attach ErrBatchOutOfMemory, and only
// after cleanup succeeds; callers must never infer retry safety from
// ErrDeviceOutOfMemory alone or from any pre-call memory estimate. Callers
// retrying a failed batch halve between complete inputs in request order on the
// same encoder and stop splitting at one input; a smaller batch can still fail
// with OOM under concurrent device demand.
//
// Per-input length limits (MaxTokens and the model's max_position_embeddings)
// are distinct from backend aggregate batch limits (CUDA's geometry-dependent
// batchRowLimit and Vulkan's 8192 total tokens across inputs), and neither
// matches ErrBatchOutOfMemory.
var ErrBatchOutOfMemory = fmt.Errorf("%w: batch retry-safe", ErrDeviceOutOfMemory)
