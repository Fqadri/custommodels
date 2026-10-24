import math
import torch

# LoRALayer class implemented allows us to transform the layer inputs 'x'
class LoRALayer(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, rank, alpha, *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Initialize the low-rank matrices A and B for LoRA
        self.A = torch.nn.Parameter(
            torch.empty(in_dim, rank, device=device, dtype=dtype)
        )
        self.B = torch.nn.Parameter(
            torch.zeros(rank, out_dim, device=device, dtype=dtype)
        )

        # random A, zero B gives a neutral starting point while allowing learning.
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # Initialize A with random values usingKaiming uniform distribution

        # scaling hyperparameter applied to the output of the low-rank adaptation
        # primarily dictates to what extent the output of the adapted layer can affect the original layer's output.
        # recall, (Input X Original Weights) + (Input X A X B). so the latter is the contribution from the LoRA adaptation.
        # higher alpha means the LoRA adaptation has a stronger influence on the output. So, can be used for regulating the impact of the low-rank adaptation.
        self.alpha = alpha

        # rank governs the inner dimnsion of matrics A and B in LoRA. 
        # This determines the number of additional parameters introduced and is a key factor in determining the balance between 
        # model adaptability and parameter efficiency
        self.rank = rank

    def forward(self, x):
        # This scaling is not necessary, but it's more canonical and convenient
        # as this lets us compare runs across different ranks without retuning learning rates
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x