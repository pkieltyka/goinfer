//go:build cuda && eg2_combined

package cuda

import (
	"errors"
	"fmt"

	"github.com/pkieltyka/goinfer/embeddinggemma2"
)

func init() {
	embeddinggemma2.RegisterAccelerator("cuda-eg2-combined", func(model *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		acc, err := newEG2Accel(model)
		if err != nil {
			return nil, err
		}
		if err := acc.enableCombined(); err != nil {
			return nil, errors.Join(err, acc.Close())
		}
		return acc, nil
	})
}

func (a *eg2Accel) enableCombined() error {
	if err := a.ops.do(func() error {
		lib, err := a.ops.dev.CompileLibrary(eg2CombinedAttentionPTX)
		if err != nil {
			return err
		}
		a.combinedTranspose, err = a.ops.dev.NewComputePipeline(lib, "eg2_transpose_keys")
		if err != nil {
			return err
		}
		a.combinedAttention, err = a.ops.dev.NewComputePipeline(lib, "eg2_attention_transposed")
		return err
	}); err != nil {
		return fmt.Errorf("cuda: load EmbeddingGemma 2 transposed attention: %w", err)
	}
	a.ops.shortTiled = true
	return nil
}
