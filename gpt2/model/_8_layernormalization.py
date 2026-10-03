
import torch
import torch.nn as nn

# Here we implement Layer Normalization from scratch.

class LayerNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        # Shift and Scale are two trainable parameters that allow the model to adjust the normalized output.
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-5 # Small constant to avoid division by zero
        # No bias term is needed here since shift serves that purpose.

    # This gives back the normalized, scaled and shifted output.
    # x is of shape (B, T, d_model) where B is batch size, T is sequence length, and d_model is the feature dimension.
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) #last dimension is d_model
        var = x.var(dim=-1, keepdim=True, unbiased=False) #unbiased=False uses N in the denominator instead of N-1 (Bessel's correction). Does not make much difference for large N.
        normalized_x = (x - mean) / torch.sqrt(var + self.eps) # Normalize the input
        return self.scale * normalized_x + self.shift # Scale and shift the normalized output
    

if __name__ == "__main__":
    torch.manual_seed(123)  # For reproducibility
    batch_example = torch.randn(2, 5) # Example batch of input sequences. Shape 2 X 5 (B X T). 2 input sequences each of length 5 tokens.

    layer_norm = LayerNorm(d_model=5)
    normalized_output = layer_norm(batch_example)
    mean = normalized_output.mean(dim=-1, keepdim=True)
    var = normalized_output.var(dim=-1, keepdim=True, unbiased=False)

    torch.set_printoptions(sci_mode=False)  # Disable scientific notation for better readability.
    print("Output after LayerNorm:", normalized_output)    
    print("Mean after LayerNorm:", mean)  # Should be close to 0
    print("Variance after LayerNorm:", var)  # Should be close to 1