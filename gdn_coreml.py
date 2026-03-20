"""
GatedDeltaNet CoreML converter for Qwen3.5-0.8B.

Builds a custom decode-step wrapper that can be traced and converted to CoreML.
The wrapper takes explicit state tensors (no Python cache objects) so torch.jit.trace works.

Step 1: Single GatedDeltaNet layer → CoreML → ANE → verify numerical match
Step 2: Full 24-layer model (18 SSM + 6 attention)
Step 3: End-to-end with embedding + lm_head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

# Qwen3.5-0.8B dimensions
HIDDEN_SIZE = 1024
NUM_V_HEADS = 16
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM      # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM    # 2048
CONV_DIM = KEY_DIM * 2 + VALUE_DIM      # 6144
CONV_KERNEL = 4
RMS_EPS = 1e-6


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class GDNDecodeStep(nn.Module):
    """Single decode step for one GatedDeltaNet layer.

    Takes explicit state tensors as input/output (no cache objects).
    All shapes are for batch=1, seq_len=1.
    """
    def __init__(self):
        super().__init__()
        # Projections
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, KEY_DIM * 2 + VALUE_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, VALUE_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, NUM_V_HEADS, bias=False)
        self.out_proj = nn.Linear(VALUE_DIM, HIDDEN_SIZE, bias=False)

        # Conv1d weights (depthwise)
        self.conv_weight = nn.Parameter(torch.randn(CONV_DIM, CONV_KERNEL))
        # No conv bias in Qwen3.5

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(NUM_V_HEADS))
        self.A_log = nn.Parameter(torch.zeros(NUM_V_HEADS))

        # RMSNormGated
        self.norm_weight = nn.Parameter(torch.ones(HEAD_V_DIM))

    def forward(self, hidden_states, conv_state, recurrent_state):
        """
        Args:
            hidden_states: [1, 1, HIDDEN_SIZE]
            conv_state:    [1, CONV_DIM, CONV_KERNEL-1]  (last 3 tokens of conv input)
            recurrent_state: [1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM]
        Returns:
            output: [1, 1, HIDDEN_SIZE]
            new_conv_state: [1, CONV_DIM, CONV_KERNEL-1]
            new_recurrent_state: [1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM]
        """
        B = 1
        # --- Projections ---
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [1, 1, 6144]
        z = self.in_proj_z(hidden_states)              # [1, 1, 2048]
        b = self.in_proj_b(hidden_states)              # [1, 1, 16]
        a = self.in_proj_a(hidden_states)              # [1, 1, 16]

        # --- Causal Conv1d Update ---
        # mixed_qkv: [1, 1, 6144] -> [1, 6144, 1]
        mixed_qkv_t = mixed_qkv.transpose(1, 2)       # [1, 6144, 1]

        # Concatenate conv_state + new token
        conv_input = torch.cat([conv_state, mixed_qkv_t], dim=2)  # [1, 6144, 4]

        # Update conv_state: last 3 elements
        new_conv_state = conv_input[:, :, 1:]          # [1, 6144, 3]

        # Depthwise conv1d: weight is [CONV_DIM, CONV_KERNEL]
        # F.conv1d expects weight [out_channels, in_channels/groups, kernel_size]
        # For depthwise: groups=CONV_DIM, so weight is [CONV_DIM, 1, CONV_KERNEL]
        conv_w = self.conv_weight.unsqueeze(1)         # [6144, 1, 4]
        conv_out = F.conv1d(conv_input, conv_w, groups=CONV_DIM, padding=0)  # [1, 6144, 1]

        # SiLU activation
        conv_out = F.silu(conv_out)

        # Back to [1, 1, 6144]
        mixed_qkv = conv_out.transpose(1, 2)

        # --- Split into Q, K, V ---
        query, key, value = mixed_qkv.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)

        # Reshape to heads
        query = query.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)    # [1, 1, 16, 128]
        key = key.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        value = value.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)

        # --- Gates ---
        beta = b.sigmoid()                             # [1, 1, 16]
        # g = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [1, 1, 16]

        # --- L2 normalize Q, K ---
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        # Scale query
        scale = HEAD_K_DIM ** -0.5
        query = query * scale

        # --- Transpose to [B, H, 1, dim] for recurrence ---
        query = query.transpose(1, 2).float()          # [1, 16, 1, 128]
        key = key.transpose(1, 2).float()
        value = value.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()            # [1, 16, 1]
        g = g.transpose(1, 2).float()                  # [1, 16, 1]

        # --- GatedDeltaNet Recurrence (single step) ---
        # Squeeze the seq_len=1 dimension for clarity
        q_t = query[:, :, 0]                           # [1, 16, 128]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]                           # [1, 16, 128]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [1, 16, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)           # [1, 16, 1]

        state = recurrent_state.float()

        # State decay
        state = state * g_t                            # [1, 16, 128, 128]

        # Retrieve from state
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [1, 16, 128]

        # Delta update
        delta = (v_t - kv_mem) * beta_t                # [1, 16, 128]

        # Write to state
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # [1, 16, 128, 128]

        # Read from state
        output = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # [1, 16, 128]

        new_recurrent_state = state.half()

        # --- RMSNormGated ---
        # output: [1, 16, 128] -> [16, 128] for norm (B=1)
        output_flat = output.reshape(-1, HEAD_V_DIM)    # [16, 128]
        z_flat = z.reshape(-1, HEAD_V_DIM).float()       # [16, 128]

        # RMSNorm
        variance = output_flat.pow(2).mean(-1, keepdim=True)
        output_flat = output_flat * torch.rsqrt(variance + RMS_EPS)
        output_flat = self.norm_weight * output_flat.half()

        # Gate
        output_flat = output_flat * F.silu(z_flat).half()

        # --- Output projection ---
        output_flat = output_flat.reshape(B, 1, VALUE_DIM)  # [1, 1, 2048]
        output = self.out_proj(output_flat)              # [1, 1, 1024]

        return output, new_conv_state, new_recurrent_state


def load_hf_weights(layer_module, layer_idx: int, model_path: str):
    """Load HuggingFace weights into our decode step module."""
    from safetensors.torch import load_file

    # Find the safetensors file
    sf_files = list(Path(model_path).glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    weights = load_file(str(sf_files[0]))

    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

    # Linear projections (cast to float32 — HF stores as bfloat16)
    layer_module.in_proj_qkv.weight.data = weights[prefix + "in_proj_qkv.weight"].float()
    layer_module.in_proj_z.weight.data = weights[prefix + "in_proj_z.weight"].float()
    layer_module.in_proj_b.weight.data = weights[prefix + "in_proj_b.weight"].float()
    layer_module.in_proj_a.weight.data = weights[prefix + "in_proj_a.weight"].float()
    layer_module.out_proj.weight.data = weights[prefix + "out_proj.weight"].float()

    # Conv1d weight: HF shape is [CONV_DIM, 1, CONV_KERNEL], we want [CONV_DIM, CONV_KERNEL]
    conv_w = weights[prefix + "conv1d.weight"].float()
    if conv_w.dim() == 3:
        conv_w = conv_w.squeeze(1)
    layer_module.conv_weight.data = conv_w

    # SSM params
    layer_module.dt_bias.data = weights[prefix + "dt_bias"].float()
    layer_module.A_log.data = weights[prefix + "A_log"].float()

    # Norm weight
    layer_module.norm_weight.data = weights[prefix + "norm.weight"].float()

    return layer_module


def test_numerical_match(model_path: str, layer_idx: int = 0):
    """Test that our decode step matches the HuggingFace implementation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== Testing numerical match for SSM layer {layer_idx} ===")

    # Load our module with HF weights
    our_module = GDNDecodeStep()
    our_module = load_hf_weights(our_module, layer_idx, model_path)
    our_module.eval()

    # Create test input
    torch.manual_seed(42)
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE) * 0.1
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL - 1)
    recurrent_state = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)

    # Run our module
    with torch.no_grad():
        our_out, our_conv, our_rec = our_module(hidden_states, conv_state, recurrent_state)

    print(f"Our output shape: {our_out.shape}")
    print(f"Our output stats: mean={our_out.mean():.6f}, std={our_out.std():.6f}")
    print(f"Our conv_state shape: {our_conv.shape}")
    print(f"Our recurrent_state shape: {our_rec.shape}")

    # Now test against HF model
    print("\nLoading HF model for reference...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True
    )
    hf_model.eval()
    lm = hf_model.model

    # Get the specific GatedDeltaNet layer
    hf_layer = lm.layers[layer_idx]
    hf_gdn = hf_layer.linear_attn

    # Create a minimal cache-like object for HF
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    # Run HF layer's GatedDeltaNet with explicit inputs
    # We need to match the decode path (use_precomputed_states=True)
    # But first, let's test prefill path (seq_len=1 without cache) for simplicity
    with torch.no_grad():
        hf_out = hf_gdn(hidden_states, cache_params=None, cache_position=None, attention_mask=None)

    print(f"\nHF output shape: {hf_out.shape}")
    print(f"HF output stats: mean={hf_out.mean():.6f}, std={hf_out.std():.6f}")

    # Compare
    diff = (our_out - hf_out).abs()
    print(f"\nDifference: max={diff.max():.8f}, mean={diff.mean():.8f}")
    if diff.max() < 0.01:
        print("PASS: Numerical match within tolerance")
    elif diff.max() < 0.1:
        print("WARN: Close but not exact (likely float precision)")
    else:
        print("FAIL: Significant numerical divergence")

    del hf_model
    return diff.max().item()


def convert_single_layer_coreml(model_path: str, layer_idx: int = 0):
    """Convert a single GatedDeltaNet decode step to CoreML."""
    import coremltools as ct

    print(f"\n=== Converting SSM layer {layer_idx} to CoreML ===")

    # Build and load weights
    module = GDNDecodeStep()
    module = load_hf_weights(module, layer_idx, model_path)
    module.eval()

    # Trace inputs (batch=1, seq_len=1)
    hidden_states = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float16)
    conv_state = torch.zeros(1, CONV_DIM, CONV_KERNEL - 1, dtype=torch.float16)
    recurrent_state = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16)

    # Cast module to fp16 for tracing
    module = module.half()

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(module, (hidden_states, conv_state, recurrent_state))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, HIDDEN_SIZE), dtype=np.float16),
            ct.TensorType(name="conv_state", shape=(1, CONV_DIM, CONV_KERNEL - 1), dtype=np.float16),
            ct.TensorType(name="recurrent_state", shape=(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float16),
            ct.TensorType(name="new_conv_state", dtype=np.float16),
            ct.TensorType(name="new_recurrent_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )

    # Save
    out_path = Path(model_path).parent / "gdn_layer0_test.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved to {out_path}")

    # Test on ANE
    print("\nTesting on ANE...")
    torch.manual_seed(42)
    test_hidden = (torch.randn(1, 1, HIDDEN_SIZE) * 0.1).half().numpy()
    test_conv = np.zeros((1, CONV_DIM, CONV_KERNEL - 1), dtype=np.float16)
    test_rec = np.zeros((1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16)

    t0 = time.time()
    result = mlmodel.predict({
        "hidden_states": test_hidden,
        "conv_state": test_conv,
        "recurrent_state": test_rec,
    })
    t1 = time.time()

    print(f"ANE inference time: {(t1-t0)*1000:.1f}ms")
    print(f"Output shape: {result['output'].shape}")
    print(f"Output stats: mean={result['output'].mean():.6f}")

    # Compare with PyTorch
    module_fp32 = GDNDecodeStep()
    module_fp32 = load_hf_weights(module_fp32, layer_idx, model_path)
    module_fp32.eval()
    with torch.no_grad():
        pt_out, _, _ = module_fp32(
            torch.from_numpy(test_hidden).float(),
            torch.from_numpy(test_conv).float(),
            torch.from_numpy(test_rec).float(),
        )

    diff = np.abs(result['output'] - pt_out.half().numpy())
    print(f"PyTorch vs CoreML diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    return mlmodel


def main():
    model_path = "/Users/midas/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"

    # Step 1: Verify numerical match
    max_diff = test_numerical_match(model_path, layer_idx=0)

    if max_diff > 0.5:
        print("\nNumerical match failed. Debugging before CoreML conversion.")
        return

    # Step 2: Convert to CoreML and test on ANE
    convert_single_layer_coreml(model_path, layer_idx=0)


if __name__ == "__main__":
    main()
