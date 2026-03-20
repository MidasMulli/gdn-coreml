"""
Teacher-forcing acceptance: feed 9B's chosen tokens to 0.8B, check if 0.8B would pick the same next token.
This measures distribution overlap independent of autoregressive drift.
"""

import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gdn_drafter import GDNCoreMLDrafter

MODEL_DIR = Path.home() / "models" / "Qwen3.5-0.8B-coreml"
TARGET_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

PROMPTS = [
    ("ISDA clause", "The ISDA Master Agreement is the most widely used master contract for OTC derivatives transactions. Section 2(a)(iii) of the ISDA"),
    ("Financial analysis", "Goldman Sachs reported Q4 2025 earnings of $14.2 billion in revenue, driven primarily by strong performance in"),
    ("Regulatory", "Under Basel III endgame rules, banks must calculate risk-weighted assets using the standardized approach for"),
    ("Collateral", "The Credit Support Annex specifies that eligible collateral includes cash in USD, EUR, and GBP, as well as"),
]


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    print("Loading 9B target model...")
    target_model, tokenizer = load(TARGET_MODEL)
    print("Loading 0.8B CoreML drafter...")
    drafter = GDNCoreMLDrafter(str(MODEL_DIR), context_length=64)
    drafter.load()

    for name, prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)

        # Get 50 greedy tokens from 9B
        prompt_mx = mx.array(tokens, mx.uint32)
        target_tokens = []
        for tok_val, _ in generate_step(prompt_mx, target_model, max_tokens=50):
            target_tokens.append(tok_val if isinstance(tok_val, int) else tok_val.item())
            if len(target_tokens) >= 50:
                break

        # Teacher-forcing: feed prompt + 9B's tokens to 0.8B, check predictions
        drafter.reset()
        full_seq = tokens + target_tokens
        matches_top1 = 0
        matches_top5 = 0
        total = 0

        for i in range(len(tokens) - 1):
            drafter._step(full_seq[i])

        for i in range(len(target_tokens)):
            pos = len(tokens) - 1 + i
            logits = drafter._step(full_seq[pos])
            logits_flat = logits[0, 0]

            pred_top1 = int(logits_flat.argmax())
            top5 = np.argsort(logits_flat)[-5:][::-1]

            expected = target_tokens[i]
            if pred_top1 == expected:
                matches_top1 += 1
            if expected in top5:
                matches_top5 += 1
            total += 1

        print(f"\n{name} ({len(tokens)} prompt tokens, {total} generated):")
        print(f"  Top-1 match: {matches_top1}/{total} ({matches_top1/total*100:.1f}%)")
        print(f"  Top-5 match: {matches_top5}/{total} ({matches_top5/total*100:.1f}%)")


if __name__ == "__main__":
    main()
