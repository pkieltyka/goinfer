//go:build cuda

package cuda

// Qwen2.5-VL's vision tower on CUDA is goinfer's own (qwen25_vision.go, on the tower base with the fused attention),
// registered through vision.RegisterQwenResident from this package. The earlier upstream Qwen CUDA tower used
// unfused attention and its scratch failed to fit beside the decoder at the serving defaults on the 8 GB card.
// The upstream SigLIP CUDA tower also failed parity at real size, and
// cuda/vision_register.go owns the one global vision.RegisterResident hook.
