

import torch
import torch.nn as nn


# Here we implement Causal Self-Attention with Dropout for illustration purposes.

# This code implements a Causal self-attention mechanism in PyTorch with Dropout. nn.Module is the base class for all neural network modules in PyTorch.
class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()

        # Initialize trainable weight matrices for query, key, and value transformations.
        # nn.Linear has an optimized weight initialization scheme contributing to more stable and effective model training than nn.Parameter.
        # requires_grad=True is the default behavior for nn.Linear, so we don't need to specify it explicitly.
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)  # No bias term for simplicity
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)   
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias) 

    # The forward method defines the computation performed at every call. 
    # We have to do manual implementation of forward here because it involves transformer specific operations.
    # It takes an input tensor x and computes the self-attention output.
    def forward(self, x):
        context_length = x.shape[0]
        keys = self.W_key(x)  # Transform input to key vectors. uses built-in .forward()
        queries = self.W_query(x)  # Transform input to query vectors. uses built-in .forward()
        values = self.W_value(x)  # Transform input to value vectors. uses built-in .forward()

        attn_scores = queries @ keys.T  # Compute attention scores

        # ----------- Create a causal mask to ensure that each token can only attend to itself and previous tokens. ---------------
        
        # triu creates an upper triangular matrix, and diagonal=1 means that the diagonal is included.
        # The masked positions are filled with a very large negative value (-torch.inf).
        # We choose -torch.inf because softmax will treat those columns as 0 probability. Works more efficiently.

        # torch.ones(context_length, context_length) creates a square matrix of ones with dimensions equal to the context length.
        # torch.triu(..., diagonal=1) creates an upper triangular matrix, where the diagonal is set to 1, meaning that the diagonal and above it will be ones, and below it will be zeros.
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)

        print("Attention Scores Matrix MASKED:\n", attn_scores)
        
        # ---------- End of Causal Masking ----------------

        # Normalize the attention scores using softmax to get attention weights, scaling by the square root of the key dimension.
        attn_weights = torch.nn.functional.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1) #dim=-1 normalizes the scores along the last dimension

        print("Attention Weights Matrix:\n", attn_weights)

        # Apply Dropout to the attention weights which prevents overfitting during training.
        dropout = torch.nn.Dropout(p=0.5)  # Dropout rate of 50%
        attn_weights = dropout(attn_weights) # This will also rescale the attention weights so that they sum to 2 due to # the dropout rate of 0.5.

        print("After Dropout: Attention Weights Matrix:\n", attn_weights)

        context_vector = attn_weights @ values  # Compute context vector
        return context_vector
    
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

ca = CausalAttention(3, 3) 
print(ca(inputs))