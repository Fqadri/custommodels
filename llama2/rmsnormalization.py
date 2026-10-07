import torch
from torch import nn


# Root Mean Square Layer Normalization (RMSNorm) implementation in PyTorch
# Avoids computing the mean and variance like traditional Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps # Small constant to avoid division by zero
        self.emb_dim = emb_dim

        # Pure normalization strips variance information. The learnable weight allows the network to learn optimal feature scales for each dimension independently during backpropagation.
        # Certain feature dimensions inside a Transformer represent crucial signals that need higher relative magnitudes than others. The weight vector lets the model amplify important channels while dampening noisy ones.
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=dtype))

    def forward(self, x):
        # Accumulate squared values in float32 for low-precision inputs to avoid overflow.
        norm_input = x
        if x.dtype in (torch.float16, torch.bfloat16):
            norm_input = x.float()
        means = norm_input.pow(2).mean(dim=-1, keepdim=True)
        x_normed = norm_input * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)