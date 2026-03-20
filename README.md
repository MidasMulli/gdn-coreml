# GDN-CoreML: GatedDeltaNet SSM on Apple Neural Engine

First working CoreML conversion of Qwen3.5's GatedDeltaNet (SSM) architecture for Apple Neural Engine inference.

ANEMLL and other CoreML converters only support attention-only architectures. Qwen3.5 is a hybrid model — 18 GatedDeltaNet SSM layers + 6 attention layers. This project implements the full SSM recurrence as traceable PyTorch, converts to CoreML via `coremltools`, and runs on ANE.

## What this does

Converts Qwen3.5-0.8B (all 24 layers) to three CoreML `.mlpackage` files:
- **embedding.mlpackage** — token ID to hidden state (485 MB, 248K vocab)
- **ffn_24layers.mlpackage** — all 24 layers: 18 SSM + 6 attention + final norm (951 MB)
- **lm_head.mlpackage** — hidden state to logits (485 MB, tied weights)

Total: ~1.9 GB on disk. Runs at **23.7 ms/token** on M5 Air ANE.

## Why this was hard

GatedDeltaNet layers have recurrent state that standard converters can't trace:
- Conv1d sliding window state (`[18, 6144, 3]`)
- Recurrent state matrix (`[18, 16, 128, 128]`)
- The recurrence: `h_t = exp(g) * h_{t-1} + k * (beta * (v - k^T h_{t-1}))`

The solution: explicit state tensors passed as model inputs/outputs (no Python cache objects), making `torch.jit.trace` work on the full SSM.

Additional bugs found and fixed during development:
- **RMSNorm**: Qwen3.5 uses `output * (1 + weight)`, not `output * weight` — weights initialize to zero
- **Attention gating**: gate applies AFTER attention output, not to query before attention
- **KV cache**: shift-left static slicing for ANE compatibility (no dynamic position indexing)
- **bfloat16 weights**: HF stores bfloat16, must cast to float32 for tracing

## Results

Verified exact numerical match against HuggingFace reference:
```
Our model: next_token=318 (' ('), time=0.31s
HF model: next_token=318 (' (')
PASS: argmax matches

Our top-10 tokens:  [318, 320, 12344, 7, 596, 1697, 37082, 8318, 11, 48660]
HF top-10 tokens:   [318, 320, 12344, 7, 596, 1697, 37082, 8318, 11, 48660]
```

CoreML end-to-end test:
```
CoreML prediction: 318 (' (')
PyTorch prediction: 318 (' (')
Match: PASS
FFN time per token: 23.7ms
```

## Use case: speculative decoding draft model

Built to run Qwen3.5-0.8B on ANE as a speculative decode draft model for larger Qwen3.5 models on GPU. Same tokenizer (248K vocab) eliminates the cross-vocab problem that killed the Qwen3-1.7B approach.

### Acceptance rates (teacher-forcing, 0.8B predicts 9B's chosen tokens)

| Prompt | Top-1 match | Top-5 match |
|--------|-------------|-------------|
| ISDA clause | 70.0% | 88.0% |
| Financial analysis | 54.0% | 78.0% |
| Regulatory | 58.0% | 82.0% |
| Collateral | 56.0% | 84.0% |

### Speculative decode wallclock (M5 Air 16GB, K=2)

| Prompt | Spec tok/s | Baseline tok/s | Speedup | Accept rate |
|--------|-----------|----------------|---------|-------------|
| ISDA clause | 24.6 | 26.0 | 0.94x | 90.3% |
| Collateral | 19.7 | 26.0 | 0.76x | 62.5% |

**On M5 Air (16GB):** 0.8B at 24ms/tok is only 1.75x faster than 9B at 42ms/tok — not enough to overcome verification overhead. Best case 0.94x (near break-even).

**On M5 Pro (64GB) with 70B target:** 0.8B at 24ms/tok vs 70B at ~200ms/tok = 8x faster — firmly in the spec decode sweet spot. This is where the converter pays off.

### Speed

- Decode: **41.3 tok/s** (24.2ms per token)
- Draft K=3: 71.6ms (42 draft tok/s)
- Draft K=5: 119.3ms
- Model load: ~11s (cached after first load)

## Files

| File | Purpose |
|------|---------|
| `gdn_full_model.py` | Full Qwen3.5-0.8B implementation — all 24 layers, weight loader, RoPE, verification test |
| `gdn_convert.py` | CoreML conversion pipeline — embedding, FFN (24 layers), lm_head |
| `gdn_drafter.py` | Autoregressive draft source — loads CoreML models, generates tokens, async threading |
| `gdn_coreml.py` | Step 1 proof — single GDN layer to CoreML (5.0ms, 0.001 max diff) |
| `gdn_debug.py` | Layer-by-layer comparison tool (multi-token, HF hooks) |
| `gdn_debug2.py` | Single-token layer-by-layer debug (found the RMSNorm bug) |
| `benchmark_gdn_*.py` | Full benchmark suite (acceptance, teacher-forcing, speculative, pipelined) |

## Requirements

```
torch
coremltools
safetensors
transformers
numpy
```

Tested with: Python 3.11, coremltools 8.x, PyTorch 2.x, macOS 26.x (Tahoe).

Requires Qwen3.5-0.8B weights from HuggingFace (downloads automatically on first run, or set `MODEL_PATH` in the scripts).

## Usage

```bash
# Step 1: Verify numerical match against HF
python gdn_full_model.py

# Step 2: Convert to CoreML (outputs to ~/models/Qwen3.5-0.8B-coreml/)
python gdn_convert.py

# Step 3 (optional): Test single layer in isolation
python gdn_coreml.py
```

## Architecture details

Qwen3.5-0.8B (hybrid GatedDeltaNet):
- 24 layers: 18 SSM (GatedDeltaNet) + 6 attention (layers 3, 7, 11, 15, 19, 23)
- Hidden: 1024, SSM heads: 16, Attention heads: 8, KV heads: 2
- Head dim: 128 (SSM), 256 (attention), partial RoPE (25%)
- Conv1d kernel: 4, intermediate: 3584, vocab: 248,044
- Total: ~829M parameters

State per decode step:
- 18 conv states: `[6144, 3]` each (shift register for depthwise conv1d)
- 18 recurrent states: `[16, 128, 128]` each (SSM hidden state matrix)
- 6 KV caches: `[2, 2, CTX, 256]` each (shift-left for static slicing)

## Limitations

- Decode only (seq_len=1). No prefill/batch mode — designed for autoregressive speculative decoding.
- Float16 precision. Matches HF at top-10 token level; sub-token logit differences are normal for fp16.
- 248K vocab makes embedding/lm_head large. Vocab pruning to 50K cuts these to ~100MB each (tested, see four-path-mlx repo).
- Hardcoded to Qwen3.5-0.8B dimensions. Adapting to 4B or other sizes requires changing the constants.
- `MODEL_PATH` is hardcoded — update it to point to your local HF cache path.

## Related work

- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) — Four-path speculative decoding server that uses this converter's output as ANE draft source
- [ANEMLL](https://github.com/ANEMLL/ANEMLL) — CoreML converter for attention-only LLMs (Llama, Qwen3, etc.)
- [coremltools](https://github.com/apple/coremltools) — Apple's PyTorch-to-CoreML conversion toolkit

## License

MIT
