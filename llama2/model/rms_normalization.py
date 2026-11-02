import torch
from torch import nn


# Root Mean Square Layer Normalization (RMSNorm) implementation in PyTorch
# Avoids computing the mean and variance like traditional Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps # Small constant to avoid division by zero
        self.emb_dim = emb_dim

        # Normalization controls the vector’s overall magnitude; scale gives the model flexibility to adjust each feature afterward.
        # During training, backpropagation adjusts these values to amplify or reduce (i.e. scale) individual features.
        self.scale = nn.Parameter(torch.ones(emb_dim, dtype=dtype))

    def forward(self, x):
        # Accumulate squared values in float32 for low-precision inputs to avoid overflow.
        norm_input = x
        if x.dtype in (torch.float16, torch.bfloat16):
            norm_input = x.float()
        means = norm_input.pow(2).mean(dim=-1, keepdim=True)
        x_normed = norm_input * torch.rsqrt(means + self.eps)
        return (x_normed * self.scale).to(dtype=x.dtype)