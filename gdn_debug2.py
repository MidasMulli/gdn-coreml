"""Direct layer-by-layer comparison: run one layer at a time on both models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gdn_full_model import (
    Qwen35DecodeStep, load_full_weights, build_rope_cache, rms_norm,
    HIDDEN_SIZE, CONV_DIM, CONV_KERNEL, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
    ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM, RMS_EPS
)

MODEL_PATH = "/Users/midas/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
CTX = 256


def main():
    print("Loading models...")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True)
    hf_model.eval()
    lm = hf_model.model

    our_model = Qwen35DecodeStep(context_length=CTX)
    our_model = load_full_weights(our_model)
    our_model.eval()

    # Single token
    token_id = 760  # "The"
    input_ids = torch.tensor([[token_id]])

    # Embedding
    hf_embed = lm.embed_tokens(input_ids)  # [1, 1, 1024]
    our_embed = our_model.embed_tokens(input_ids)  # [1, 1, 1024]
    print(f"Embedding diff: {(hf_embed - our_embed).abs().max():.8f}")

    # Now run layer 0 (SSM) on both
    # HF: needs position_embeddings, cache, etc.
    # Let's just get the hidden state after layer 0 from HF using hooks

    hf_hidden_after = {}
    def make_hook(name):
        def hook(module, input, output):
            # Decoder layer returns FloatTensor (per typing)
            hf_hidden_after[name] = output.detach() if not isinstance(output, tuple) else output[0].detach()
        return hook

    for i in range(3):  # Hook first 3 layers
        lm.layers[i].register_forward_hook(make_hook(f"layer_{i}"))

    # HF forward with single token
    with torch.no_grad():
        hf_out = hf_model(input_ids)

    # Our model: run layer 0 manually
    hidden = our_embed.clone()

    # Layer 0 is SSM
    ssm_layer = our_model.ssm_layers["0"]
    conv = torch.zeros(1, CONV_DIM, CONV_KERNEL - 1)
    rec = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)

    with torch.no_grad():
        our_out_0, _, _ = ssm_layer(hidden, conv, rec)

    hf_out_0 = hf_hidden_after["layer_0"]

    print(f"\nAfter layer 0 (SSM):")
    print(f"  HF shape: {hf_out_0.shape}, Our shape: {our_out_0.shape}")
    diff = (hf_out_0[0, 0] - our_out_0[0, 0]).abs()
    print(f"  Max diff: {diff.max():.8f}, Mean: {diff.mean():.8f}")
    print(f"  HF stats: mean={hf_out_0.mean():.6f}, std={hf_out_0.std():.6f}")
    print(f"  Our stats: mean={our_out_0.mean():.6f}, std={our_out_0.std():.6f}")

    if diff.max() > 0.01:
        print("  DIVERGENCE at layer 0!")
        # Debug further: check pre-norm output
        hf_norm_w = lm.layers[0].input_layernorm.weight
        our_norm_w = ssm_layer.input_layernorm_weight

        print(f"\n  Layer 0 input_layernorm weight diff: {(hf_norm_w - our_norm_w).abs().max():.8f}")

        # Apply norms manually
        hf_normed = lm.layers[0].input_layernorm(hidden)
        our_normed = rms_norm(hidden, our_norm_w, RMS_EPS)
        print(f"  After input_layernorm: {(hf_normed - our_normed).abs().max():.8f}")

        # Check the SSM output (just the attention part, without residual/MLP)
        hf_gdn = lm.layers[0].linear_attn
        with torch.no_grad():
            hf_ssm_out = hf_gdn(hf_normed, cache_params=None, cache_position=None, attention_mask=None)
        print(f"  HF SSM raw output shape: {hf_ssm_out.shape}")
        print(f"  HF SSM raw output stats: mean={hf_ssm_out.mean():.6f}, std={hf_ssm_out.std():.6f}")

        # Our SSM doesn't expose the raw attention output easily, but we verified
        # single layer match in gdn_coreml.py. The issue might be in the RMSNorm.

        # Check HF's RMSNorm implementation
        print(f"\n  HF RMSNorm type: {type(lm.layers[0].input_layernorm).__name__}")
        # Check if HF uses a different norm than us
        hf_norm_class = type(lm.layers[0].input_layernorm)
        print(f"  HF norm forward source:")
        import inspect
        src = inspect.getsource(hf_norm_class.forward)
        print(f"  {src[:200]}...")

    # Continue to layer 1
    with torch.no_grad():
        our_out_1, _, _ = our_model.ssm_layers["1"](our_out_0,
            torch.zeros(1, CONV_DIM, CONV_KERNEL - 1),
            torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM))

    hf_out_1 = hf_hidden_after["layer_1"]
    diff_1 = (hf_out_1[0, 0] - our_out_1[0, 0]).abs()
    print(f"\nAfter layer 1 (SSM):")
    print(f"  Max diff: {diff_1.max():.8f}")


if __name__ == "__main__":
    main()
