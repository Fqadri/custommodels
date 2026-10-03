
import torch
import torch.nn as nn

# Here we implement a basic version of self-attention mechanism with no multi-heads, no masking, and no dropout for simplicity.

# This code implements a self-attention mechanism in PyTorch. nn.Module is the base class for all neural network modules in PyTorch.
class SelfAttention_v1(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()

        # Initialize trainable weight matrices for query, key, and value transformations.
        
        # nn.Linear has an optimized weight initialization scheme contributing to more stable and effective model training than nn.Parameter.
        # This model will have 3 separate linear layers, each transforming the input x to a different representation (Q, K, V).
        # requires_grad=True is the default behavior for nn.Linear, so we don't need to specify it explicitly.
        # dim_in and dim_out are usually the same representing the dimensionality of the input and output embeddings.
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)  # No bias term for simplicity. Shape is (dim_out, dim_in)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)   
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias) 

    # The forward method defines the computation performed at every call.
    # It takes an input tensor x and computes the self-attention output.
    def forward(self, x):
        keys = self.W_key(x)  # Transform input to key vectors. Uses built-in .forward() so no manual things needed as with nn.Parameter.
        queries = self.W_query(x)  # Transform input to query vectors. if x has shape (seq_len, dim_in), keys and queries will have shape (seq_len, dim_out)
        values = self.W_value(x)  # Transform input to value vectors

        attn_scores = queries @ keys.T  # Compute attention scores
        attn_weights = torch.nn.functional.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1) #dim=-1 normalizes the scores along the last dimension
        context_vector = attn_weights @ values  # Compute context vector
        return context_vector

# Set random seed for reproducibility
torch.manual_seed(123)

# Using 1 input sequence with 3 dimensions for simplicity.
# Input sequence is "Your journey starts with one step" which has 6 tokens.
inputs = torch.tensor([
    [0.43, 0.15, 0.89],   #Your
    [0.55, 0.87, 0.66],   #journey
    [0.57, 0.85, 0.64],   #starts
    [0.22, 0.58, 0.33],   #with
    [0.77, 0.25, 0.10],   #one
    [0.05, 0.80, 0.55]    #step
])

sa_v1 = SelfAttention_v1(3, 3)  # Create an instance of the SelfAttention class with input and output dimensions of 3
print(sa_v1(inputs))