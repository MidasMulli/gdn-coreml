"""
Full Qwen3.5-0.8B CoreML converter — all 24 layers + embedding + lm_head.

Builds on gdn_coreml.py (single layer proven). This assembles:
- Token embedding
- 18 GatedDeltaNet (SSM) layers
- 6 Full Attention layers
- Final RMSNorm
- LM head

For decode (seq_len=1), uses explicit state tensors — no Python cache objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from safetensors.torch import load_file

# Qwen3.5-0.8B config
HIDDEN_SIZE = 1024
NUM_LAYERS = 24
NUM_V_HEADS = 16    # SSM value heads
NUM_K_HEADS = 16    # SSM key heads
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM      # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM    # 2048
CONV_DIM = KEY_DIM * 2 + VALUE_DIM      # 6144
CONV_KERNEL = 4
RMS_EPS = 1e-6
# Attention params
ATTN_NUM_HEADS = 8
ATTN_NUM_KV_HEADS = 2
ATTN_HEAD_DIM = 256  # from config: head_dim=256
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(ATTN_HEAD_DIM * PARTIAL_ROTARY)  # 64
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248044  # full vocab (not pruned)

# Layer type map: True = attention, False = SSM
ATTN_LAYERS = {3, 7, 11, 15, 19, 23}

MODEL_PATH = "/Users/midas/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rms_norm(x, weight, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    # Qwen3.5 RMSNorm: weight is initialized to 0, actual scale is (1 + weight)
    return (x_normed * (1.0 + weight.float())).to(x.dtype)


class GDNLayer(nn.Module):
    """Single GatedDeltaNet SSM layer (decode step, seq_len=1)."""
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, CONV_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, VALUE_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.out_proj = nn.Linear(VALUE_DIM, HIDDEN_SIZE, bias=False)
        self.conv_weight = nn.Parameter(torch.randn(CONV_DIM, CONV_KERNEL))
        self.dt_bias = nn.Parameter(torch.ones(NUM_V_HEADS))
        self.A_log = nn.Parameter(torch.zeros(NUM_V_HEADS))
        self.norm_weight = nn.Parameter(torch.ones(HEAD_V_DIM))
        self.input_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.post_attn_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        # MLP
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states, conv_state, recurrent_state):
        """
        hidden_states: [1, 1, 1024]
        conv_state: [1, 6144, 3]
        recurrent_state: [1, 16, 128, 128]
        """
        residual = hidden_states

        # Pre-norm
        hidden_states = rms_norm(hidden_states, self.input_layernorm_weight, RMS_EPS)

        # --- GatedDeltaNet attention ---
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Conv1d update
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        conv_input = torch.cat([conv_state, mixed_qkv_t], dim=2)
        new_conv_state = conv_input[:, :, 1:]
        conv_w = self.conv_weight.unsqueeze(1)
        conv_out = F.conv1d(conv_input, conv_w, groups=CONV_DIM, padding=0)
        conv_out = F.silu(conv_out)
        mixed_qkv = conv_out.transpose(1, 2)

        # Split Q, K, V
        query, key, value = mixed_qkv.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        query = query.reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        key = key.reshape(1, 1, NUM_K_HEADS, HEAD_K_DIM)
        value = value.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM)

        # Gates
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # L2 norm + scale
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = HEAD_K_DIM ** -0.5

        # Transpose to [1, H, 1, dim]
        query = (query * scale).transpose(1, 2).float()
        key = key.transpose(1, 2).float()
        value = value.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()
        g = g.transpose(1, 2).float()

        # Single step recurrence
        q_t = query[:, :, 0]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)

        state = recurrent_state.float()
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        attn_out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        new_recurrent_state = state.half()

        # RMSNormGated
        attn_out_flat = attn_out.reshape(-1, HEAD_V_DIM)
        z_flat = z.reshape(-1, HEAD_V_DIM).float()
        variance = attn_out_flat.pow(2).mean(-1, keepdim=True)
        attn_out_flat = attn_out_flat * torch.rsqrt(variance + RMS_EPS)
        attn_out_flat = self.norm_weight * attn_out_flat.half()
        attn_out_flat = attn_out_flat * F.silu(z_flat).half()

        hidden_states = self.out_proj(attn_out_flat.reshape(1, 1, VALUE_DIM))

        # Residual
        hidden_states = residual + hidden_states

        # --- MLP ---
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attn_layernorm_weight, RMS_EPS)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_conv_state, new_recurrent_state


class AttnLayer(nn.Module):
    """Single attention layer (decode step, seq_len=1).

    Simplified: no sliding window, uses basic single-token attention against KV cache.
    """
    def __init__(self, context_length=256):
        super().__init__()
        self.context_length = context_length
        # Attention projections (q_proj doubles for gate)
        self.q_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(ATTN_NUM_HEADS * ATTN_HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.q_norm_weight = nn.Parameter(torch.ones(ATTN_HEAD_DIM))
        self.k_norm_weight = nn.Parameter(torch.ones(ATTN_HEAD_DIM))
        # Norms and MLP
        self.input_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.post_attn_layernorm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states, kv_cache, cos_cur, sin_cur, causal_mask):
        """
        hidden_states: [1, 1, 1024]
        kv_cache: [2, NUM_KV_HEADS, context_length, HEAD_DIM]
        cos_cur, sin_cur: [1, 1, 1, ROTARY_DIM] — pre-computed for current position
        causal_mask: [1, 1, 1, context_length] — 0 for valid, -inf for masked
        Returns: hidden_states, new_kv_cache
        """
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm_weight, RMS_EPS)

        # Q, K, V projections
        q_out = self.q_proj(hidden_states)  # [1, 1, 4096]
        k = self.k_proj(hidden_states)      # [1, 1, 512]
        v = self.v_proj(hidden_states)      # [1, 1, 512]

        # Split q_proj output into query and gate
        q_out = q_out.reshape(1, 1, ATTN_NUM_HEADS, ATTN_HEAD_DIM * 2)
        q, gate = q_out.chunk(2, dim=-1)  # each [1, 1, 8, 256]
        gate = gate.reshape(1, 1, -1)     # [1, 1, 2048]

        k = k.reshape(1, 1, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM)
        v = v.reshape(1, 1, ATTN_NUM_KV_HEADS, ATTN_HEAD_DIM)

        # QK Norm
        q = rms_norm(q, self.q_norm_weight, RMS_EPS)
        k = rms_norm(k, self.k_norm_weight, RMS_EPS)

        # RoPE (partial rotary) — cos/sin pre-computed for current position
        q_rot = q[..., :ROTARY_DIM]
        q_pass = q[..., ROTARY_DIM:]
        k_rot = k[..., :ROTARY_DIM]
        k_pass = k[..., ROTARY_DIM:]

        q1, q2 = q_rot[..., :ROTARY_DIM//2], q_rot[..., ROTARY_DIM//2:]
        q_rotated = torch.cat([-q2, q1], dim=-1)
        q_rot = q_rot * cos_cur + q_rotated * sin_cur

        k1, k2 = k_rot[..., :ROTARY_DIM//2], k_rot[..., ROTARY_DIM//2:]
        k_rotated = torch.cat([-k2, k1], dim=-1)
        k_rot = k_rot * cos_cur + k_rotated * sin_cur

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # Transpose to [1, H, 1, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Shift-left KV cache + append (static slicing, ANE-compatible)
        old_k = kv_cache[0:1]
        old_v = kv_cache[1:2]
        new_k_cache = torch.cat([old_k[:, :, 1:, :], k], dim=2)
        new_v_cache = torch.cat([old_v[:, :, 1:, :], v], dim=2)
        new_kv_cache = torch.cat([new_k_cache, new_v_cache], dim=0)

        # Repeat KV for GQA
        num_groups = ATTN_NUM_HEADS // ATTN_NUM_KV_HEADS
        k_full = new_k_cache.repeat_interleave(num_groups, dim=1)
        v_full = new_v_cache.repeat_interleave(num_groups, dim=1)

        # Attention with causal mask
        scale = ATTN_HEAD_DIM ** -0.5
        attn_weights = torch.matmul(q, k_full.transpose(-1, -2)) * scale
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_out = torch.matmul(attn_weights, v_full)  # [1, 8, 1, 256]
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, ATTN_NUM_HEADS * ATTN_HEAD_DIM)

        # Apply gate AFTER attention (attn_output_gate=true)
        attn_out = attn_out * torch.sigmoid(gate)

        hidden_states = self.o_proj(attn_out)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attn_layernorm_weight, RMS_EPS)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class Qwen35DecodeStep(nn.Module):
    """Full Qwen3.5-0.8B decode step — one token in, logits out.

    All state is explicit (no cache objects). For CoreML tracing.
    """
    def __init__(self, context_length=256):
        super().__init__()
        self.context_length = context_length

        # Embedding (will be large — 248K * 1024)
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

        # Layers
        self.ssm_layers = nn.ModuleDict()
        self.attn_layers = nn.ModuleDict()
        for i in range(NUM_LAYERS):
            if i in ATTN_LAYERS:
                self.attn_layers[str(i)] = AttnLayer(context_length)
            else:
                self.ssm_layers[str(i)] = GDNLayer()

        # Final norm
        self.final_norm_weight = nn.Parameter(torch.ones(HIDDEN_SIZE))

        # LM head (tied to embedding in Qwen3.5-0.8B)
        # We'll handle tying after weight loading

    def forward(
        self, token_id,
        # SSM states
        conv_states,       # [18, CONV_DIM, 3]
        recurrent_states,  # [18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM]
        # Attention states
        kv_caches,         # [6, 2, NUM_KV_HEADS, CTX, HEAD_DIM]
        # Pre-computed RoPE for current position
        cos_cur, sin_cur,  # [1, 1, 1, ROTARY_DIM]
        # Causal mask
        causal_mask,       # [1, 1, 1, CTX]
    ):
        """Single decode step."""
        hidden_states = self.embed_tokens(token_id)  # [1, 1, 1024]

        ssm_idx = 0
        attn_idx = 0
        new_conv_states = []
        new_recurrent_states = []
        new_kv_caches = []

        for i in range(NUM_LAYERS):
            if i in ATTN_LAYERS:
                kv = kv_caches[attn_idx]
                hidden_states, new_kv = self.attn_layers[str(i)](
                    hidden_states, kv, cos_cur, sin_cur, causal_mask
                )
                new_kv_caches.append(new_kv)
                attn_idx += 1
            else:
                conv = conv_states[ssm_idx].unsqueeze(0)       # [1, CONV_DIM, 3]
                rec = recurrent_states[ssm_idx].unsqueeze(0)   # [1, 16, 128, 128]
                hidden_states, new_conv, new_rec = self.ssm_layers[str(i)](
                    hidden_states, conv, rec
                )
                new_conv_states.append(new_conv.squeeze(0))      # [CONV_DIM, 3]
                new_recurrent_states.append(new_rec.squeeze(0))  # [16, 128, 128]
                ssm_idx += 1

        # Final norm
        hidden_states = rms_norm(hidden_states, self.final_norm_weight, RMS_EPS)

        # LM head (use embedding weights — tied)
        logits = F.linear(hidden_states, self.embed_tokens.weight)  # [1, 1, VOCAB_SIZE]

        # Pack states
        new_conv_states = torch.stack(new_conv_states, dim=0)
        new_recurrent_states = torch.stack(new_recurrent_states, dim=0)
        new_kv_caches = torch.stack(new_kv_caches, dim=0)

        return logits, new_conv_states, new_recurrent_states, new_kv_caches


def load_full_weights(model: Qwen35DecodeStep):
    """Load all weights from HF Qwen3.5-0.8B."""
    print("Loading weights from safetensors...")
    sf_files = list(Path(MODEL_PATH).glob("*.safetensors"))
    weights = {}
    for f in sf_files:
        weights.update(load_file(str(f)))

    prefix = "model.language_model.layers."

    ssm_idx = 0
    attn_idx = 0
    for i in range(NUM_LAYERS):
        lp = f"{prefix}{i}."

        if i in ATTN_LAYERS:
            layer = model.attn_layers[str(i)]
            layer.q_proj.weight.data = weights[lp + "self_attn.q_proj.weight"].float()
            layer.k_proj.weight.data = weights[lp + "self_attn.k_proj.weight"].float()
            layer.v_proj.weight.data = weights[lp + "self_attn.v_proj.weight"].float()
            layer.o_proj.weight.data = weights[lp + "self_attn.o_proj.weight"].float()
            layer.q_norm_weight.data = weights[lp + "self_attn.q_norm.weight"].float()
            layer.k_norm_weight.data = weights[lp + "self_attn.k_norm.weight"].float()
            attn_idx += 1
        else:
            layer = model.ssm_layers[str(i)]
            layer.in_proj_qkv.weight.data = weights[lp + "linear_attn.in_proj_qkv.weight"].float()
            layer.in_proj_z.weight.data = weights[lp + "linear_attn.in_proj_z.weight"].float()
            layer.in_proj_b.weight.data = weights[lp + "linear_attn.in_proj_b.weight"].float()
            layer.in_proj_a.weight.data = weights[lp + "linear_attn.in_proj_a.weight"].float()
            layer.out_proj.weight.data = weights[lp + "linear_attn.out_proj.weight"].float()
            conv_w = weights[lp + "linear_attn.conv1d.weight"].float()
            if conv_w.dim() == 3:
                conv_w = conv_w.squeeze(1)
            layer.conv_weight.data = conv_w
            layer.dt_bias.data = weights[lp + "linear_attn.dt_bias"].float()
            layer.A_log.data = weights[lp + "linear_attn.A_log"].float()
            layer.norm_weight.data = weights[lp + "linear_attn.norm.weight"].float()
            ssm_idx += 1

        # Shared across both layer types
        target = model.attn_layers[str(i)] if i in ATTN_LAYERS else model.ssm_layers[str(i)]
        target.input_layernorm_weight.data = weights[lp + "input_layernorm.weight"].float()
        target.post_attn_layernorm_weight.data = weights[lp + "post_attention_layernorm.weight"].float()
        target.gate_proj.weight.data = weights[lp + "mlp.gate_proj.weight"].float()
        target.up_proj.weight.data = weights[lp + "mlp.up_proj.weight"].float()
        target.down_proj.weight.data = weights[lp + "mlp.down_proj.weight"].float()

    # Embedding (tied with lm_head)
    model.embed_tokens.weight.data = weights["model.language_model.embed_tokens.weight"].float()

    # Final norm
    model.final_norm_weight.data = weights["model.language_model.norm.weight"].float()

    print(f"Loaded weights for {NUM_LAYERS} layers + embedding + norm")
    return model


def build_rope_cache(context_length=256, base=10000000):
    """Build cosine/sine cache for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    t = torch.arange(context_length).float()
    freqs = torch.outer(t, inv_freq)
    cos_cache = freqs.cos()  # [CTX, ROTARY_DIM/2]
    sin_cache = freqs.sin()
    # Expand to full rotary dim by repeating (since we split into halves)
    cos_cache = torch.cat([cos_cache, cos_cache], dim=-1)  # [CTX, ROTARY_DIM]
    sin_cache = torch.cat([sin_cache, sin_cache], dim=-1)
    return cos_cache, sin_cache


def test_full_model():
    """Test full model: load weights, run decode, verify against HF."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CTX = 256
    print("Building model...")
    model = Qwen35DecodeStep(context_length=CTX)
    model = load_full_weights(model)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.0f}M")

    # Build RoPE cache
    cos_cache, sin_cache = build_rope_cache(CTX)

    # Test with a real token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    test_text = "The ISDA"
    tokens = tokenizer.encode(test_text)
    print(f"Test tokens: {tokens}")

    # Initialize states
    conv_states = torch.zeros(18, CONV_DIM, CONV_KERNEL - 1)
    recurrent_states = torch.zeros(18, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    kv_caches = torch.zeros(6, 2, ATTN_NUM_KV_HEADS, CTX, ATTN_HEAD_DIM)

    # Run tokens through our model
    print("\nRunning our model...")
    t0 = time.time()
    for pos, tok in enumerate(tokens):
        token_id = torch.tensor([[tok]])
        # Pre-compute RoPE for current position
        cos_cur = cos_cache[pos:pos+1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, ROTARY_DIM]
        sin_cur = sin_cache[pos:pos+1].unsqueeze(0).unsqueeze(0)
        # Shift-left mask: valid positions are the last (pos+1) entries
        causal_mask = torch.full((1, 1, 1, CTX), float('-inf'))
        causal_mask[:, :, :, CTX-pos-1:] = 0.0
        with torch.no_grad():
            logits, conv_states, recurrent_states, kv_caches = model(
                token_id, conv_states, recurrent_states, kv_caches,
                cos_cur, sin_cur, causal_mask
            )
    our_next_token = logits[0, 0].argmax().item()
    our_time = time.time() - t0
    print(f"Our model: next_token={our_next_token} ({tokenizer.decode([our_next_token])!r}), time={our_time:.2f}s")

    # Compare with HF model
    print("\nRunning HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True
    )
    hf_model.eval()
    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        hf_out = hf_model(input_ids)
        hf_logits = hf_out.logits[0, -1]
    hf_next_token = hf_logits.argmax().item()
    print(f"HF model: next_token={hf_next_token} ({tokenizer.decode([hf_next_token])!r})")

    # Compare logits
    our_top10 = logits[0, 0].topk(10)
    hf_top10 = hf_logits.topk(10)
    print(f"\nOur top-10 tokens:  {our_top10.indices.tolist()}")
    print(f"HF top-10 tokens:   {hf_top10.indices.tolist()}")
    print(f"Our top-10 logits:  {[f'{v:.2f}' for v in our_top10.values.tolist()]}")
    print(f"HF top-10 logits:   {[f'{v:.2f}' for v in hf_top10.values.tolist()]}")

    match = our_next_token == hf_next_token
    print(f"\n{'PASS' if match else 'FAIL'}: argmax {'matches' if match else 'does not match'}")

    del hf_model
    return model, match


if __name__ == "__main__":
    model, match = test_full_model()
    if match:
        print("\n=== Numerical verification passed. Ready for CoreML conversion. ===")
    else:
        print("\n=== Debugging needed before CoreML conversion. ===")
