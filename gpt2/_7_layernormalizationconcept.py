
import torch
import torch.nn as nn


# Here we will go over LayerNorm  concept which happens in multiple places. 
# After applying activation to the raw output (logits), we perform layer normalization to stabilize and speed up training.

torch.manual_seed(123)  # For reproducibility
batch_example = torch.randn(2, 5)  # Example batch of input sequences. Shape 2 X 5 (B X T). 2 input sequences each of length 5 tokens.

# Example layer: Linear transformation followed by ReLU activation.
layer = nn.Sequential(nn.Linear(5, 5), nn.ReLU()) 
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)#dim=1 or dim=-1 calculates mean across the column dimension (i.e. mean of row). keepdim to maintain the same number of dimensions.
var = out.var(dim=-1, keepdim=True)
print("Mean:", mean)
print("Variance:", var)

normalized_out = (out - mean) / torch.sqrt(var) # Normalize the output using mean and variance. Now each row will have mean 0 and variance 1. Note each row represents a token in the sequence, and we are normalizing across the feature dimension (columns) for each token.

torch.set_printoptions(sci_mode=False)  # Disable scientific notation for better readability.
print("Normalized Output:", normalized_out)

# Let us check how we did.
mean = normalized_out.mean(dim=-1, keepdim=True)
var = normalized_out.var(dim=-1, keepdim=True)

print("Mean after normalization:", mean)  # Should be close to 0
print("Variance after normalization:", var)  # Should be close to 1