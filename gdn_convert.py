"""
Convert verified Qwen3.5-0.8B to CoreML for ANE.

Strategy: Convert in parts (like ANEMLL):
1. Embedding (token_id → hidden_states)
2. FFN layers (hidden_states + states → hidden_states + new_states)
3. LM head (hidden_states → logits)

For the FFN, we process ALL layers in one model to avoid state coordination overhead.
State is passed as separate inputs/outputs (not ct.StateType) for initial testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import time
from pathlib import Path
from safetensors.torch import load_file

from gdn_full_model import (
    Qwen35DecodeStep, load_full_weights, build_rope_cache,
    HIDDEN_SIZE, NUM_LAYERS, ATTN_LAYERS, CONV_DIM, CONV_KERNEL,
    NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM,
    VOCAB_SIZE, INTERMEDIATE_SIZE, RMS_EPS, ROTARY_DIM,
    rms_norm, l2norm, GDNLayer, AttnLayer
)

MODEL_PATH = "/Users/midas/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
CTX = 64  # Short context for draft model
OUT_DIR = Path("/Users/midas/models/Qwen3.5-0.8B-coreml")


class FFNWrapper(nn.Module):
    """Wraps all 24 layers for tracing. No Python control flow — everything is explicit."""

    def __init__(self, model: Qwen35DecodeStep):
        super().__init__()
        # Copy all layers as named modules for tracing
        self.ssm_layers = model.ssm_layers
        self.attn_layers = model.attn_layers
        self.final_norm_weight = model.final_norm_weight

        # Pre-compute layer order for trace-friendly iteration
        self.layer_order = []
        ssm_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            if i in ATTN_LAYERS:
                self.layer_order.append(('attn', str(i), attn_idx))
                attn_idx += 1
            else:
                self.layer_order.append(('ssm', str(i), ssm_idx))
                ssm_idx += 1

    def forward(
        self,
        hidden_states,      # [1, 1, HIDDEN_SIZE]
        conv_states,        # [18, CONV_DIM, 3]
        recurrent_states,   # [18, 16, 128, 128]
        kv_caches,          # [6, 2, 2, CTX, 256]
        cos_cur,            # [1, 1, 1, ROTARY_DIM]
        sin_cur,            # [1, 1, 1, ROTARY_DIM]
        causal_mask,        # [1, 1, 1, CTX]
    ):
        new_conv_list = []
        new_rec_list = []
        new_kv_list = []

        ssm_idx = 0
        attn_idx = 0

        for i in range(NUM_LAYERS):
            if i in ATTN_LAYERS:
                kv = kv_caches[attn_idx]
                hidden_states, new_kv = self.attn_layers[str(i)](
                    hidden_states, kv, cos_cur, sin_cur, causal_mask
                )
                new_kv_list.append(new_kv)
                attn_idx += 1
            else:
                conv = conv_states[ssm_idx].unsqueeze(0)
                rec = recurrent_states[ssm_idx].unsqueeze(0)
                hidden_states, new_conv, new_rec = self.ssm_layers[str(i)](
                    hidden_states, conv, rec
                )
                new_conv_list.append(new_conv.squeeze(0))
                new_rec_list.append(new_rec.squeeze(0))
                ssm_idx += 1

        # Final norm
        hidden_states = rms_norm(hidden_states, self.final_norm_weight, RMS_EPS)

        new_conv_states = torch.stack(new_conv_list, dim=0)
        new_recurrent_states = torch.stack(new_rec_list, dim=0)
        new_kv_caches = torch.stack(new_kv_list, dim=0)

        return hidden_states, new_conv_states, new_recurrent_states, new_kv_caches


def convert_embedding():
    """Convert embedding layer to CoreML."""
    print("=== Converting embedding ===")

    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    class EmbedWrapper(nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed
        def forward(self, token_id):
            return self.embed(token_id)

    wrapper = EmbedWrapper(model.embed_tokens).half()
    token_id = torch.tensor([[0]], dtype=torch.int32)

    traced = torch.jit.trace(wrapper, token_id)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="token_id", shape=(1, 1), dtype=np.int32)],
        outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path = OUT_DIR / "embedding.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return mlmodel


def convert_lm_head():
    """Convert LM head to CoreML."""
    print("\n=== Converting LM head ===")

    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    class LMHeadWrapper(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, hidden_states):
            return F.linear(hidden_states, self.weight)

    wrapper = LMHeadWrapper(model.embed_tokens.weight).half()
    hidden = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float16)

    traced = torch.jit.trace(wrapper, hidden)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=(1, 1, HIDDEN_SIZE), dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    out_path = OUT_DIR / "lm_head.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return mlmodel


def convert_ffn():
    """Convert all 24 layers to CoreML."""
    print("\n=== Converting FFN (24 layers) ===")

    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    wrapper = FFNWrapper(model).half()

    # Create trace inputs (no dynamic position — use pre-computed cos/sin and shift-left cache)
    hidden = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float16)
    conv_states = torch.zeros(18, CONV_DIM, CONV_KERNEL - 1, dtype=torch.float16)
    rec_states = torch.zeros(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16)
    kv_caches = torch.zeros(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM, dtype=torch.float16)
    cos_cur = torch.zeros(1, 1, 1, ROTARY_DIM, dtype=torch.float16)
    sin_cur = torch.zeros(1, 1, 1, ROTARY_DIM, dtype=torch.float16)
    causal_mask = torch.full((1, 1, 1, CTX), float('-inf'), dtype=torch.float16)
    causal_mask[:, :, :, -1:] = 0.0

    print("Tracing FFN wrapper...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (
            hidden, conv_states, rec_states, kv_caches,
            cos_cur, sin_cur, causal_mask
        ))
    print(f"Traced in {time.time()-t0:.1f}s")

    print("Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, HIDDEN_SIZE), dtype=np.float16),
            ct.TensorType(name="conv_states", shape=(18, CONV_DIM, CONV_KERNEL-1), dtype=np.float16),
            ct.TensorType(name="recurrent_states", shape=(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16),
            ct.TensorType(name="kv_caches", shape=(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM), dtype=np.float16),
            ct.TensorType(name="cos_cur", shape=(1, 1, 1, ROTARY_DIM), dtype=np.float16),
            ct.TensorType(name="sin_cur", shape=(1, 1, 1, ROTARY_DIM), dtype=np.float16),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, CTX), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="new_conv_states", dtype=np.float16),
            ct.TensorType(name="new_recurrent_states", dtype=np.float16),
            ct.TensorType(name="new_kv_caches", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    print(f"Converted in {time.time()-t0:.1f}s")

    out_path = OUT_DIR / "ffn_24layers.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")
    return mlmodel


def test_end_to_end(embed_model, ffn_model, head_model):
    """Test the full CoreML pipeline."""
    print("\n=== End-to-end CoreML test ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokens = tokenizer.encode("The ISDA")

    cos_cache, sin_cache = build_rope_cache(CTX)
    conv_states = np.zeros((18, CONV_DIM, CONV_KERNEL-1), dtype=np.float16)
    rec_states = np.zeros((18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16)
    kv_caches = np.zeros((6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM), dtype=np.float16)

    total_time = 0
    for pos, tok in enumerate(tokens):
        token_id = np.array([[tok]], dtype=np.int32)
        hidden = embed_model.predict({"token_id": token_id})["hidden_states"]

        # Pre-computed RoPE for current position
        cos_cur = cos_cache[pos:pos+1].unsqueeze(0).unsqueeze(0).half().numpy()
        sin_cur = sin_cache[pos:pos+1].unsqueeze(0).unsqueeze(0).half().numpy()

        # Shift-left mask: valid positions are at the end
        mask = np.full((1, 1, 1, CTX), np.float16(-65504.0), dtype=np.float16)
        mask[:, :, :, CTX-pos-1:] = 0.0

        t0 = time.time()
        result = ffn_model.predict({
            "hidden_states": hidden,
            "conv_states": conv_states,
            "recurrent_states": rec_states,
            "kv_caches": kv_caches,
            "cos_cur": cos_cur,
            "sin_cur": sin_cur,
            "causal_mask": mask,
        })
        total_time += time.time() - t0

        hidden_out = result["output_hidden_states"]
        conv_states = result["new_conv_states"]
        rec_states = result["new_recurrent_states"]
        kv_caches = result["new_kv_caches"]

        logits = head_model.predict({"hidden_states": hidden_out})["logits"]

    next_token = int(logits[0, 0].argmax())
    print(f"CoreML prediction: {next_token} ({tokenizer.decode([next_token])!r})")
    print(f"FFN time per token: {total_time/len(tokens)*1000:.1f}ms")
    print(f"Total FFN time: {total_time*1000:.1f}ms for {len(tokens)} tokens")

    # Compare with PyTorch
    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    conv_pt = torch.zeros(18, CONV_DIM, CONV_KERNEL-1)
    rec_pt = torch.zeros(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    kv_pt = torch.zeros(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM)

    for pos, tok in enumerate(tokens):
        token_id = torch.tensor([[tok]])
        cos_cur_pt = cos_cache[pos:pos+1].unsqueeze(0).unsqueeze(0)
        sin_cur_pt = sin_cache[pos:pos+1].unsqueeze(0).unsqueeze(0)
        mask = torch.full((1, 1, 1, CTX), float('-inf'))
        mask[:, :, :, CTX-pos-1:] = 0.0
        with torch.no_grad():
            logits_pt, conv_pt, rec_pt, kv_pt = model(
                token_id, conv_pt, rec_pt, kv_pt, cos_cur_pt, sin_cur_pt, mask
            )

    pt_token = logits_pt[0, 0].argmax().item()
    print(f"PyTorch prediction: {pt_token} ({tokenizer.decode([pt_token])!r})")
    print(f"Match: {'PASS' if next_token == pt_token else 'FAIL'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embed = convert_embedding()
    head = convert_lm_head()
    ffn = convert_ffn()
    test_end_to_end(embed, ffn, head)


if __name__ == "__main__":
    main()
