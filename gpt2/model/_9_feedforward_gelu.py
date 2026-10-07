
import torch
import torch.nn as nn

# Here we will go over the FeedForward network as used in transformers and implement it in PyTorch.

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    # x is of shape (B, T, d_model) where B is batch size, T is sequence length, and d_model is the feature dimension.
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))))
    

# FeedForward network as used in transformers. Consists of two (neuron network layers) linear layers with a GELU activation in between.
class FeedForward(nn.Module):
    def __init__(
            self,
            cfg,
            debug=False,
            dtype = None # dtype setting to be able to instantiate the model with a lower precision later
            ):
        
        super().__init__()
        dmodel = cfg["dim_model"]
        self.layers = nn.Sequential(
            nn.Linear(dmodel, 4 * dmodel, dtype=dtype), # Expand the feature dimension from d_model to 4*d_model (common to get better representation). Think of this layer having 4*d_model neurons.
            GELU(),
            nn.Linear(4 * dmodel, dmodel, dtype=dtype) # Project back to d_model. Think of this layer having d_model neurons.
        )

    # x - input to the feed-forward network. Shape (batch of input sequences - B, num_tokens - T, dim_model - d_model)
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(123)  # For reproducibility
    batch_example = torch.randn(1, 2, 768) # Example batch of input sequences. Shape 1 X 2 X 768 (B X T X D). 1 input sequences each of length 2 tokens and 768 dimensions.

    cfg = {
        "dim_model": 768,    # Dimension of the model (d_model).
    }

    feedforward = FeedForward(cfg) # d_model = 768 as in GPT-2 small
    out = feedforward(batch_example)

    print("Output shape after FeedForward:", out.shape)  #Should print (1, 2, 768)
    print("Output after FeedForward:", out)
