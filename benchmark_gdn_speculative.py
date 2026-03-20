"""
Realistic speculative decode benchmark: 0.8B CoreML drafts K tokens,
9B verifies, on rejection we feed back the correct token.

This is the real wallclock test — does the 0.8B ANE draft actually speed things up?

Math:
- 9B baseline: ~42ms/token
- 0.8B draft K=3: ~72ms (24ms/tok)
- 9B verify K+1: ~110ms (batch plateau)
- Per round: draft(72ms) + verify(110ms) = 182ms
- At 60% per-token acceptance, expected accepted per round ≈ 1.5 + 1 correction = 2.5 tokens
- Throughput: 2.5 tokens / 182ms ≈ 13.7 tok/s (SLOWER than baseline)

BUT: ANE drafts on separate silicon. If we pipeline:
- Round N verify (110ms) OVERLAPS with round N+1 draft (72ms)
- Effective: max(72, 110) = 110ms per round
- At 2.5 tokens/round: 2.5/110ms ≈ 22.7 tok/s — still slower

The only way this works is if draft is fast enough to hide in the verify window
AND acceptance is high enough. Let's measure the real numbers.
"""

import time
import sys
from pathlib import Path
import functools

sys.path.insert(0, str(Path(__file__).parent))
from gdn_drafter import GDNCoreMLDrafter

MODEL_DIR = Path.home() / "models" / "Qwen3.5-0.8B-coreml"
TARGET_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

PROMPTS = [
    ("ISDA clause", "The ISDA Master Agreement is the most widely used master contract for OTC derivatives transactions. Section 2(a)(iii) of the ISDA"),
    ("Collateral", "The Credit Support Annex specifies that eligible collateral includes cash in USD, EUR, and GBP, as well as"),
    ("Regulatory", "Under Basel III endgame rules, banks must calculate risk-weighted assets using the standardized approach for"),
]

K = 3
GEN_TOKENS = 100


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache

    print(f"Loading 9B target model...")
    target_model, tokenizer = load(TARGET_MODEL)
    print(f"Loading 0.8B CoreML drafter (K={K})...")
    drafter = GDNCoreMLDrafter(str(MODEL_DIR), context_length=64)
    drafter.load()

    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0, kv_group_size=64, kv_bits=None,
    )

    # === Baseline: standard 9B generation ===
    print(f"\n{'='*60}")
    print("Baseline: 9B standard generation")

    for name, prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        prompt_mx = mx.array(tokens, mx.uint32)

        # Baseline
        from mlx_lm.generate import generate_step
        output = []
        t0 = time.perf_counter()
        for tok_val, _ in generate_step(prompt_mx, target_model, max_tokens=GEN_TOKENS):
            tok = tok_val if isinstance(tok_val, int) else tok_val.item()
            output.append(tok)
            if len(output) >= GEN_TOKENS:
                break
        baseline_time = time.perf_counter() - t0
        baseline_tps = len(output) / baseline_time
        print(f"  {name}: {baseline_tps:.1f} tok/s ({len(output)} tokens in {baseline_time*1000:.0f}ms)")

    # === Speculative: 0.8B draft + 9B verify ===
    print(f"\n{'='*60}")
    print(f"Speculative: 0.8B CoreML draft (K={K}) + 9B verify")

    for name, prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        prompt_mx = mx.array(tokens, mx.uint32)

        # Set up 9B cache
        model_cache = cache.make_prompt_cache(target_model)

        def _forward(y, n_predict=1):
            with mx.stream(generation_stream):
                logits = target_model(y[None], cache=model_cache)
                logits = logits[:, -n_predict:, :]
                quantize_fn(model_cache)
                tok = mx.argmax(logits.squeeze(0), axis=-1)
                mx.eval(tok)
                return tok.tolist() if tok.ndim > 0 else [tok.item()]

        # Prefill 9B
        y = prompt_mx.astype(mx.uint32)
        prefill_step = 2048
        with mx.stream(generation_stream):
            while y.size > prefill_step:
                target_model(y[:prefill_step][None], cache=model_cache)
                quantize_fn(model_cache)
                mx.eval([c.state for c in model_cache if hasattr(c, "state")])
                y = y[prefill_step:]
                mx.clear_cache()

        # First token from 9B
        first_tokens = _forward(y)
        first_tok = first_tokens[-1]

        # Prefill drafter with prompt
        drafter.reset()
        for tok in tokens:
            drafter._step(tok)

        output = [first_tok]
        y_9b = mx.array([first_tok], mx.uint32)
        sources = {"draft": 0, "gpu": 1}
        rounds = 0
        total_proposed = 0

        t0 = time.perf_counter()

        while len(output) < GEN_TOKENS:
            rounds += 1

            # Draft K tokens from 0.8B (on ANE)
            draft_tokens = drafter.draft(K=K, last_token=output[-1])
            if not draft_tokens:
                # Fallback to single token
                toks = _forward(y_9b)
                tok = toks[-1]
                output.append(tok)
                sources["gpu"] += 1
                y_9b = mx.array([tok], mx.uint32)
                continue

            total_proposed += len(draft_tokens)

            # Verify with 9B (batch)
            draft_mx = mx.array(draft_tokens, mx.uint32)
            verify_input = mx.concatenate([y_9b, draft_mx])
            verify_tokens = _forward(verify_input, len(draft_tokens) + 1)

            # Count accepted
            n_accepted = 0
            for i in range(len(draft_tokens)):
                if verify_tokens[i] == draft_tokens[i]:
                    n_accepted += 1
                else:
                    break

            n_rejected = len(draft_tokens) - n_accepted

            # Trim cache for rejected
            if n_rejected > 0:
                cache.trim_prompt_cache(model_cache, n_rejected)

            # Rewind drafter state for rejected tokens
            if n_rejected > 0:
                drafter.rewind(n_rejected)
                # Feed the correct rejection token to keep drafter in sync
                correction_tok = verify_tokens[n_accepted]
                drafter._step(correction_tok)

            # Collect accepted tokens
            for i in range(n_accepted):
                output.append(draft_tokens[i])
                sources["draft"] += 1

            # Add the correction/next token from 9B
            correction = verify_tokens[n_accepted]
            output.append(correction)
            sources["gpu"] += 1

            y_9b = mx.array([output[-1]], mx.uint32)

        elapsed = time.perf_counter() - t0
        spec_tps = len(output) / elapsed
        accept_rate = sources["draft"] / total_proposed * 100 if total_proposed > 0 else 0

        text = tokenizer.decode(output[:30])
        print(f"  {name}: {spec_tps:.1f} tok/s ({len(output)} tokens in {elapsed*1000:.0f}ms)")
        print(f"    Acceptance: {sources['draft']}/{total_proposed} ({accept_rate:.1f}%), rounds: {rounds}")
        print(f"    Sources: draft={sources['draft']}, gpu={sources['gpu']}")
        print(f"    Output: {text!r}...")


if __name__ == "__main__":
    main()
