# EmbeddingGemma 2 sm_120 module images

These six images pair with the committed PTX listed in the manifest. They are
used only by the EmbeddingGemma 2 text attachment on compute capability 12.0.
Other devices and shared CUDA callers use their original PTX. The existing
build constraints continue to govern the FP16 modules. The selector names
and vector profile identities are unchanged.

To check all six images, run the verifier under the shared timing lock
from the repository root and pass the NVIDIA wheel named in the manifest.

    python3 scripts/timing_lock.py run --shared --label eg2-sm120-rebuild -- \
      python3 backend/cuda/build_eg2_sm120.py /path/to/the/pinned.whl

The verifier checks the wheel and assembler hashes, creates two independent
assemblies, and compares both to the committed bytes. The assembler is a
build-time tool. Runtime module loading uses the CUDA driver.

A changed PTX digest sends that module through the PTX loader. To update a
pair, qualify exact vector parity on the affected selector and update the
image, manifest, and loader digest together. Native module compatibility
errors allow one retry using the original PTX.
