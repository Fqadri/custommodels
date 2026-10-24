
# Example: batch size 3, input/output width 4, rank 2.
# The training setup freezes the base weights; A and B remain trainable.
#
#                         Frozen base path
# X [3 x 4] --+--> Linear(X) -----------------------------+
#             |                                          |
#             |              LoRA path                   v
#             +--> A [4 x 2] --> B [2 x 4] --> (* s) --> (+) --> Y [3 x 4]
#
# Linear(X) = X @ W.T + b, where W is [4 x 4] (b is optional).
# LoRA path:  X @ A [3 x 2] -> X @ A @ B [3 x 4] -> scale by s.
# s = alpha / rank
# Y = Linear(X) + s * (X @ A @ B)

# In Multi-Lora serving Inference engines:
# The base weights are loaded once, rather than duplicated for every adapter.
# Each adapter contains its own small A/B matrices for the layers it modifies.
# Each request selects which adapter to use—or uses the base model alone.

import torch
from gpt2.finetuning_instruction.LoRA.lora_layer import LoRALayer

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

    def forward(self, x):
        # combines standard Linear layer with the LoRA layer.
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear): # replaces linear layer with LinearWithLoRA layer
            setattr(model, name, LinearWithLoRA(module, rank, alpha)) 
        else:
            replace_linear_with_lora(module, rank, alpha) # recursively replaces linear layers in the model