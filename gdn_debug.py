"""Debug: compare our model vs HF layer by layer to find divergence."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn_full_model import (
    Qwen35DecodeStep, load_full_weights, build_rope_cache,
    rms_norm, HIDDEN_SIZE, ATTN_LAYERS, CONV_DIM, CONV_KERNEL,
    NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM,
    NUM_LAYERS
)

MODEL_PATH = "/Users/midas/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
CTX = 256


def compare_after_ssm_only():
    """Run just the first SSM layer on both models and compare."""
    print("=== Comparing hidden states after each layer ===\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokens = tokenizer.encode("The ISDA")
    input_ids = torch.tensor([tokens])

    # Load HF model
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True)
    hf_model.eval()
    lm = hf_model.model

    # Get HF hidden states after each layer using a forward hook
    hf_hidden = {}
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_hidden[name] = output[0].detach()
            else:
                hf_hidden[name] = output.detach()
        return hook

    # Hook into each layer
    for i, layer in enumerate(lm.layers):
        layer.register_forward_hook(make_hook(f"layer_{i}"))

    # Also hook embedding
    lm.embed_tokens.register_forward_hook(make_hook("embedding"))
    lm.norm.register_forward_hook(make_hook("final_norm"))

    with torch.no_grad():
        hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits[0, -1]

    # Now run our model token by token
    print("Loading our model...")
    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    cos_cache, sin_cache = build_rope_cache(CTX)
    conv_states = torch.zeros(18, CONV_DIM, CONV_KERNEL - 1)
    recurrent_states = torch.zeros(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    kv_caches = torch.zeros(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM)

    # Process all tokens
    for pos, tok in enumerate(tokens):
        token_id = torch.tensor([[tok]])
        position = torch.tensor(pos)
        causal_mask = torch.full((1, 1, 1, CTX), float('-inf'))
        causal_mask[:, :, :, :pos+1] = 0.0
        with torch.no_grad():
            logits, conv_states, recurrent_states, kv_caches = model(
                token_id, position, conv_states, recurrent_states, kv_caches,
                cos_cache, sin_cache, causal_mask
            )

    our_logits = logits[0, 0]

    # For the last token: compare HF hidden states (at position -1) with our hidden states
    # HF processes all tokens in parallel, so we compare with position [0, -1, :]
    # Our model processes token by token, and we have the final hidden state

    # Let's check: after embedding, do they match for the last token?
    hf_embed = hf_hidden["embedding"][0, -1]  # last token embedding
    our_embed = model.embed_tokens(torch.tensor([[tokens[-1]]]))[0, 0]
    print(f"\nEmbedding last token: diff = {(hf_embed - our_embed).abs().max():.8f}")

    # The HF model processes all tokens at once, so layer_i output includes all positions
    # The SSM layers depend on sequence order, so the last position's output
    # should match our sequential processing

    # Compare after each layer
    print(f"\n{'Layer':>8} {'Type':>5} {'HF last tok':>15} {'Our output':>15} {'Max diff':>12}")
    print("-" * 60)

    # We can't easily get intermediate states from our sequential model.
    # Instead, let's just compare the final output more carefully.

    print(f"\nFinal logits comparison:")
    print(f"  HF argmax:  {hf_logits.argmax().item()} = {tokenizer.decode([hf_logits.argmax().item()])!r}")
    print(f"  Our argmax: {our_logits.argmax().item()} = {tokenizer.decode([our_logits.argmax().item()])!r}")

    # Check logit correlation
    hf_top = hf_logits.topk(20)
    our_logits_at_hf_top = our_logits[hf_top.indices]
    print(f"\n  HF top-20 tokens and our logits at those positions:")
    for i in range(20):
        tok = hf_top.indices[i].item()
        hf_val = hf_top.values[i].item()
        our_val = our_logits_at_hf_top[i].item()
        word = tokenizer.decode([tok])
        print(f"    {tok:>6} ({word:>10}): HF={hf_val:8.3f}  Ours={our_val:8.3f}  diff={hf_val-our_val:8.3f}")

    # Test with just 1 token
    print("\n\n=== Single token test (position 0 only) ===")
    model2 = Qwen35DecodeStep(context_length=CTX)
    model2 = load_full_weights(model2)
    model2.eval()

    conv_states2 = torch.zeros(18, CONV_DIM, CONV_KERNEL - 1)
    recurrent_states2 = torch.zeros(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    kv_caches2 = torch.zeros(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM)

    token_id = torch.tensor([[tokens[0]]])
    position = torch.tensor(0)
    causal_mask = torch.full((1, 1, 1, CTX), float('-inf'))
    causal_mask[:, :, :, :1] = 0.0

    with torch.no_grad():
        logits2, _, _, _ = model2(
            token_id, position, conv_states2, recurrent_states2, kv_caches2,
            cos_cache, sin_cache, causal_mask
        )

    # HF with single token
    input_ids_1 = torch.tensor([[tokens[0]]])
    with torch.no_grad():
        hf_out_1 = hf_model(input_ids_1)
        hf_logits_1 = hf_out_1.logits[0, 0]

    our_logits_1 = logits2[0, 0]
    print(f"Single token:")
    print(f"  HF argmax:  {hf_logits_1.argmax().item()} = {tokenizer.decode([hf_logits_1.argmax().item()])!r}")
    print(f"  Our argmax: {our_logits_1.argmax().item()} = {tokenizer.decode([our_logits_1.argmax().item()])!r}")

    diff = (hf_logits_1 - our_logits_1).abs()
    print(f"  Logit diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    if diff.max() < 0.01:
        print("  PASS: Single token matches!")
    else:
        print(f"  FAIL: Single token diverges")
        # Find where the divergence starts
        hf_top1 = hf_logits_1.topk(10)
        our_at_hf = our_logits_1[hf_top1.indices]
        print(f"  Top-10 comparison:")
        for i in range(10):
            tok = hf_top1.indices[i].item()
            print(f"    {tok:>6}: HF={hf_top1.values[i].item():8.3f}  Ours={our_at_hf[i].item():8.3f}")


if __name__ == "__main__":
    compare_after_ssm_only()
