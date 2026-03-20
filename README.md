# GDN-CoreML: GatedDeltaNet SSM on Apple Neural Engine

To our knowledge, the first CoreML conversion of a GatedDeltaNet (SSM) architecture for Apple Neural Engine inference.

ANEMLL and other CoreML converters only support attention-only architectures. Qwen3.5 is a hybrid model — 18 GatedDeltaNet SSM layers + 6 attention layers. This project implements the full SSM recurrence as traceable PyTorch, converts to CoreML via `coremltools`, and runs on ANE.

## What this does

Converts Qwen3.5-0.8B (all 24 layers) to three CoreML `.mlpackage` files:
- **embedding.mlpackage** — token ID to hidden state (485 MB, 248K vocab)
- **ffn_24layers.mlpackage** — all 24 layers: 18 SSM + 6 attention + final norm (951 MB)
- **lm_head.mlpackage** — hidden state to logits (485 MB, tied weights)

Total: ~1.9 GB on disk.

## Results

### Numerical verification

Verified exact top-10 token match against HuggingFace reference model:
```
Our model: next_token=318 (' ('), time=0.31s
HF model: next_token=318 (' (')
PASS: argmax matches

Our top-10 tokens:  [318, 320, 12344, 7, 596, 1697, 37082, 8318, 11, 48660]
HF top-10 tokens:   [318, 320, 12344, 7, 596, 1697, 37082, 8318, 11, 48660]
```

### ANE decode speed

| Metric | Value | Notes |
|--------|-------|-------|
| FFN (24 layers) per token | **23.7 ms** | The core model — SSM + attention + norms |
| End-to-end decode | **24.2 ms** (~41.3 tok/s) | Embedding + FFN + lm_head + argmax |
| Draft K=3 | 71.6 ms | 3 autoregressive steps |
| Draft K=5 | 119.3 ms | 5 autoregressive steps |
| Model load | ~11s | Cached after first load |

The 23.7ms figure is the FFN-only time. End-to-end including embedding lookup and lm_head is 24.2ms per token.

### Speculative decode performance

**Bottom line: negative speedup on M5 Air 16GB.** The 0.8B draft model is not fast enough relative to the 9B target to overcome verification overhead.

**Autoregressive speculative decode** (sequential K-token drafting + batch verify):

| Prompt | Spec tok/s | Baseline tok/s | Speedup | Accept rate |
|--------|-----------|----------------|---------|-------------|
| ISDA clause | 24.6 | 26.0 | **0.94x** | 90.3% |
| Collateral | 19.7 | 26.0 | **0.76x** | 62.5% |

**Teacher-forcing** (upper bound — 0.8B predicts 9B's chosen tokens independently at each position, not autoregressive):

| Prompt | Top-1 match | Top-5 match |
|--------|-------------|-------------|
| ISDA clause | 70.0% | 88.0% |
| Financial analysis | 54.0% | 78.0% |
| Regulatory | 58.0% | 82.0% |
| Collateral | 56.0% | 84.0% |

Teacher-forcing measures how often the 0.8B's top prediction matches the 9B's greedy choice at each position independently. This is an upper bound — autoregressive acceptance is lower because errors compound (37-59% measured in autoregressive mode, varying by prompt type).

### Why no speedup on M5 Air

The math: 0.8B draft at 24ms/tok (ANE), 9B target at 42ms/tok (GPU). Speed ratio = 42/24 = **1.75x**. Speculative decoding needs the draft to be 5-10x faster than the target to amortize verification overhead. At 1.75x, each speculative round costs almost as much as just running the target directly.

**On M5 Pro (64GB) with 70B target:** 0.8B at 24ms/tok vs 70B at ~200ms/tok = 200/24 = **8.3x ratio**. At 37-59% autoregressive acceptance and 8x speed ratio, expected speedup is 1.3-1.8x. The converter was built for this configuration.

## Why this was hard

GatedDeltaNet layers have recurrent state that standard converters can't trace:
- Conv1d sliding window state (`[18, 6144, 3]`)
- Recurrent state matrix (`[18, 16, 128, 128]`)
- The recurrence: `h_t = exp(g) * h_{t-1} + k * (beta * (v - k^T h_{t-1}))`

The solution: explicit state tensors passed as model inputs/outputs (no Python cache objects), making `torch.jit.trace` work on the full SSM.

Bugs found and fixed during development:
- **RMSNorm**: Qwen3.5 uses `output * (1 + weight)`, not `output * weight` — weights initialize to zero
- **Attention gating**: gate applies AFTER attention output, not to query before attention
- **KV cache**: shift-left static slicing for ANE compatibility (no dynamic position indexing)
- **bfloat16 weights**: HF stores bfloat16, must cast to float32 for tracing

## Files

| File | Purpose |
|------|---------|
| `gdn_full_model.py` | Full Qwen3.5-0.8B implementation — all 24 layers, weight loader, RoPE, verification test |
| `gdn_convert.py` | CoreML conversion pipeline — embedding, FFN (24 layers), lm_head |
| `gdn_drafter.py` | Autoregressive draft source — loads CoreML models, generates tokens, async threading |
| `gdn_coreml.py` | Step 1 proof — single GDN layer to CoreML (5.0ms, 0.001 max diff) |
| `gdn_debug.py` | Layer-by-layer comparison tool (multi-token, HF hooks) |
| `gdn_debug2.py` | Single-token layer-by-layer debug (found the RMSNorm bug) |
| `benchmark_gdn_acceptance.py` | Autoregressive acceptance measurement |
| `benchmark_gdn_teacher_forcing.py` | Teacher-forcing acceptance measurement |
| `benchmark_gdn_speculative.py` | Sequential speculative decode wallclock |
| `benchmark_gdn_pipelined.py` | Pipelined speculative decode with K sweep |

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

- **Decode only** (seq_len=1). No prefill/batch mode — designed for autoregressive speculative decoding.
- **Float16 precision.** Matches HF at top-10 token level; sub-token logit differences are normal for fp16.
- **Large vocab** (248K) makes embedding/lm_head ~485MB each. Vocab pruning to 50K cuts these to ~100MB each (tested).
- **Hardcoded to Qwen3.5-0.8B dimensions.** The converter handles the full hybrid architecture (18 GatedDeltaNet SSM + 6 attention layers), but dimension constants are specific to 0.8B. Adapting to other Qwen3.5 sizes (4B, etc.) requires updating these constants.
- **Negative speedup on M5 Air 16GB.** The 0.8B is only 1.75x faster than the 9B — insufficient for speculative decoding. Designed for 64GB Pro hardware with 70B target.
- **`MODEL_PATH` is hardcoded** — update it to point to your local HF cache path.

## Requirements

```
torch
coremltools
safetensors
transformers
numpy
```

Tested with: Python 3.11, coremltools 8.x, PyTorch 2.x, macOS 26.3 (Tahoe), MacBook Air M5 16GB.

Requires Qwen3.5-0.8B weights from HuggingFace (downloads automatically on first run, or set `MODEL_PATH` in the scripts).

## Related

- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) — Four-path speculative decoding server that uses this converter's output as ANE draft source
- [orion-ane](https://github.com/MidasMulli/orion-ane) — ANE training + persistent memory daemon + agent framework
- [dual-path-inference](https://github.com/MidasMulli/dual-path-inference) — Initial GPU+ANE concurrency proof-of-concept (archived)
- [ANEMLL](https://github.com/ANEMLL/ANEMLL) — CoreML converter for attention-only LLMs (Llama, Qwen3, etc.)
- [coremltools](https://github.com/apple/coremltools) — Apple's PyTorch-to-CoreML conversion toolkit

## License

MIT
