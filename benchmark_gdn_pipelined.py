"""
Pipelined speculative decode: ANE drafts round N+1 in parallel with GPU verifying round N.

Key insight: ANE and GPU are separate silicon. If we overlap:
- GPU verify round N (110ms) runs while ANE drafts round N+1 (72ms for K=3)
- ANE finishes first (72 < 110), so draft is always ready when GPU finishes
- Effective cost: max(draft, verify) = 110ms per round, not 182ms

Problem: ANE needs the last accepted token to start drafting. After rejection,
we need to feed the correct token to the drafter before it can draft again.
This adds 24ms sync overhead.

Approach: pipeline with one-round-behind sync.
"""

import time
import sys
import threading
from pathlib import Path
import functools

sys.path.insert(0, str(Path(__file__).parent))
from gdn_drafter import GDNCoreMLDrafter

MODEL_DIR = Path.home() / "models" / "Qwen3.5-0.8B-coreml"
TARGET_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

PROMPTS = [
    ("ISDA clause", "The ISDA Master Agreement is the most widely used master contract for OTC derivatives transactions. Section 2(a)(iii) of the ISDA"),
    ("Collateral", "The Credit Support Annex specifies that eligible collateral includes cash in USD, EUR, and GBP, as well as"),
]

GEN_TOKENS = 100


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.generate import generate_step, generation_stream, maybe_quantize_kv_cache

    print("Loading models...")
    target_model, tokenizer = load(TARGET_MODEL)
    drafter = GDNCoreMLDrafter(str(MODEL_DIR), context_length=64)
    drafter.load()

    quantize_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0, kv_group_size=64, kv_bits=None,
    )

    # Test different K values
    for K in [1, 2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"K={K} draft tokens per round")

        for name, prompt in PROMPTS:
            tokens = tokenizer.encode(prompt)
            prompt_mx = mx.array(tokens, mx.uint32)

            # --- Baseline ---
            output_base = []
            t0 = time.perf_counter()
            for tok_val, _ in generate_step(prompt_mx, target_model, max_tokens=GEN_TOKENS):
                tok = tok_val if isinstance(tok_val, int) else tok_val.item()
                output_base.append(tok)
                if len(output_base) >= GEN_TOKENS:
                    break
            baseline_tps = len(output_base) / (time.perf_counter() - t0)

            # --- Speculative with pipelining ---
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
            with mx.stream(generation_stream):
                target_model(y[None], cache=model_cache)
                quantize_fn(model_cache)
                mx.eval([c.state for c in model_cache if hasattr(c, "state")])

            # Actually we need to get the first token properly
            # Re-do: use single-step prefill
            model_cache = cache.make_prompt_cache(target_model)
            y = prompt_mx.astype(mx.uint32)
            prefill_step = 2048
            with mx.stream(generation_stream):
                while y.size > prefill_step:
                    target_model(y[:prefill_step][None], cache=model_cache)
                    quantize_fn(model_cache)
                    mx.eval([c.state for c in model_cache if hasattr(c, "state")])
                    y = y[prefill_step:]

            first_tokens = _forward(y)
            first_tok = first_tokens[-1]

            # Prefill drafter
            drafter.reset()
            for tok in tokens:
                drafter._step(tok)

            output = [first_tok]
            y_9b = mx.array([first_tok], mx.uint32)
            total_accepted = 0
            total_proposed = 0
            total_rounds = 0

            # Pipeline: kick off first draft
            drafter.draft_async(K=K, last_token=first_tok)

            t0 = time.perf_counter()

            while len(output) < GEN_TOKENS:
                total_rounds += 1

                # Get ANE draft (should already be done from previous round)
                draft_tokens = drafter.get_draft(timeout=0.2)

                if not draft_tokens:
                    # Fallback: single token
                    toks = _forward(y_9b)
                    tok = toks[-1]
                    output.append(tok)
                    y_9b = mx.array([tok], mx.uint32)
                    # Sync drafter and start next draft
                    drafter._step(tok)
                    drafter.draft_async(K=K, last_token=tok)
                    continue

                total_proposed += len(draft_tokens)

                # Verify with 9B
                draft_mx = mx.array(draft_tokens, mx.uint32)
                verify_input = mx.concatenate([y_9b, draft_mx])
                verify_tokens = _forward(verify_input, len(draft_tokens) + 1)

                n_accepted = 0
                for i in range(len(draft_tokens)):
                    if verify_tokens[i] == draft_tokens[i]:
                        n_accepted += 1
                    else:
                        break

                n_rejected = len(draft_tokens) - n_accepted
                if n_rejected > 0:
                    cache.trim_prompt_cache(model_cache, n_rejected)

                # Collect accepted
                for i in range(n_accepted):
                    output.append(draft_tokens[i])
                total_accepted += n_accepted

                # Correction token
                correction = verify_tokens[n_accepted]
                output.append(correction)

                y_9b = mx.array([output[-1]], mx.uint32)

                # Sync drafter: rewind rejected, feed correction
                if n_rejected > 0:
                    drafter.rewind(n_rejected)
                    drafter._step(correction)
                else:
                    drafter._step(correction)

                # Pipeline: start next ANE draft immediately
                drafter.draft_async(K=K, last_token=output[-1])

            elapsed = time.perf_counter() - t0
            spec_tps = len(output) / elapsed
            accept = total_accepted / total_proposed * 100 if total_proposed > 0 else 0
            speedup = spec_tps / baseline_tps

            print(f"  {name}: {spec_tps:.1f} tok/s (baseline {baseline_tps:.1f}, {speedup:.2f}x)")
            print(f"    Accept: {total_accepted}/{total_proposed} ({accept:.1f}%), rounds: {total_rounds}")


if __name__ == "__main__":
    main()
