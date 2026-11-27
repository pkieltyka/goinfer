### Added

- `backend/cuda/eg2`: lightweight text-only CUDA leaf package for EmbeddingGemma 2 (blank-import, no exported identifiers, no `decoder` or generation-kernel dependency). `backend/cuda` blank-imports it so existing full-backend imports, selectors, profiles, tags, and defaults are unchanged.
- `backend/cuda/internal/tower`: shared internal CUDA tower executor, kernel helpers, and device ledger used by both `backend/cuda` and `backend/cuda/eg2`.
