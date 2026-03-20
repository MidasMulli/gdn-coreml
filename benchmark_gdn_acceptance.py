"""
Benchmark: GDN CoreML 0.8B acceptance rate against 9B target.

Same tokenizer family (Qwen3.5, 248K vocab) — no cross-vocab issues.
Measures greedy acceptance: does argmax(0.8B) == argmax(9B)?
"""

import time
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gdn_drafter import GDNCoreMLDrafter

MODEL_DIR = Path.home() / "models" / "Qwen3.5-0.8B-coreml"
TARGET_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
CTX = 64

PROMPTS = [
    ("ISDA clause", "The ISDA Master Agreement is the most widely used master contract for OTC derivatives transactions. Section 2(a)(iii) of the ISDA"),
    ("Financial analysis", "Goldman Sachs reported Q4 2025 earnings of $14.2 billion in revenue, driven primarily by strong performance in"),
    ("Regulatory", "Under Basel III endgame rules, banks must calculate risk-weighted assets using the standardized approach for"),
    ("Collateral", "The Credit Support Annex specifies that eligible collateral includes cash in USD, EUR, and GBP, as well as"),
]


def main():
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    print("Loading 9B target model...")
    target_model, tokenizer = load(TARGET_MODEL)
    print("Loading 0.8B CoreML drafter...")
    drafter = GDNCoreMLDrafter(str(MODEL_DIR), context_length=CTX)
    drafter.load()

    for name, prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        print(f"\n{'='*60}")
        print(f"Prompt: {name} ({len(tokens)} tokens)")

        # Prefill drafter
        drafter.reset()
        for tok in tokens[:-1]:
            drafter._step(tok)

        # Get 9B greedy tokens (one at a time for fair comparison)
        target_tokens = []
        prompt_mx = mx.array(tokens, mx.uint32)
        step_gen = generate_step(prompt_mx, target_model, max_tokens=50)
        for tok_val, _ in step_gen:
            target_tokens.append(tok_val if isinstance(tok_val, int) else tok_val.item())
            if len(target_tokens) >= 50:
                break

        # Get 0.8B draft tokens
        draft_tokens = []
        last_tok = tokens[-1]
        for i in range(50):
            logits = drafter._step(last_tok)
            last_tok = int(logits[0, 0].argmax())
            draft_tokens.append(last_tok)

        # Compare
        matches = sum(1 for d, t in zip(draft_tokens, target_tokens) if d == t)
        total = min(len(draft_tokens), len(target_tokens))
        rate = matches / total * 100

        # Chain acceptance (consecutive matches from start)
        chain = 0
        for d, t in zip(draft_tokens, target_tokens):
            if d == t:
                chain += 1
            else:
                break

        target_text = tokenizer.decode(target_tokens[:20])
        draft_text = tokenizer.decode(draft_tokens[:20])

        print(f"  Greedy acceptance: {matches}/{total} ({rate:.1f}%)")
        print(f"  First chain length: {chain}")
        print(f"  9B:  {target_text!r}")
        print(f"  0.8B: {draft_text!r}")

    # Speed test: how fast can we draft K tokens?
    print(f"\n{'='*60}")
    print("Draft speed (K=3, K=5, K=8):")
    drafter.reset()
    tokens = tokenizer.encode(PROMPTS[0][1])
    for tok in tokens:
        drafter._step(tok)

    for K in [3, 5, 8]:
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            draft = drafter.draft(K=K, last_token=tokens[-1])
            times.append(time.perf_counter() - t0)
            drafter.rewind(K)
        avg = np.mean(times) * 1000
        print(f"  K={K}: {avg:.1f}ms ({K*1000/avg:.0f} draft tok/s)")


if __name__ == "__main__":
    main()
