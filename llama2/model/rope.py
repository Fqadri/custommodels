
import torch
from torch import nn

# Interleaved RoPE rotates adjacent even/odd dimensions, matching original LLaMA.

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Step 1: Calculate one frequency for each adjacent pair.
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    positions = torch.arange(context_length)

    # Step 2: Calculate each pair's angle as token position times its frequency.
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Both coordinates of each adjacent pair use the same angle.
    angles = angles.repeat_interleave(2, dim=1)  # Shape: (context_length, head_dim)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

# Pre-req: Invoke precompute_rope_params to get cos and sin (i.e. angles to rotate) before calling compute_rope.
def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    _batch_size, _num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Create adjacent pairs (x0, x1), (x2, x3), and so on.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # This selects the cos and sin values corresponding to the even indices of the head dimension and reshapes them for broadcasting.
    cos = cos[:seq_len, 0::2].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim // 2)
    sin = sin[:seq_len, 0::2].unsqueeze(0).unsqueeze(0)

    # Rotate each pair: (a*cos - b*sin, a*sin + b*cos).
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    return x_rotated.to(dtype=x.dtype)


# Cache the interleaved RoPE tables once and reuse them for queries and keys.
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta_base: float = 10_000,
        context_length: int = 4096,
    ) -> None:
        super().__init__()
        cos, sin = precompute_rope_params(
            head_dim=head_dim,
            theta_base=theta_base,
            context_length=context_length,
        )
        # These derived tables move with the module but need not be saved in checkpoints.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return compute_rope(x, self.cos, self.sin)


# The following is an example of applying RoPE to the q and k tensors
if __name__ == "__main__":
    # Settings
    batch_size = 2
    context_len = 5
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)

    print("Rotary position embedded queries shape:", queries_rot.shape)
    print("Rotary position embedded keys shape:", keys_rot.shape)