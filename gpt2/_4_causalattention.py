

import torch
import torch.nn as nn

# Here we will implement Causal Self-Attention mechanism with Dropout with actual code.

# This code implements a Causal self-attention mechanism in PyTorch with Dropout. Supports Batched input sequences.
class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        # Initialize the 3 linear layers each transforming the input x to a different representation (Q, K, V).
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias) 
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)   
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias) 

        # Dropout layer to prevent overfitting during training
        self.dropout = nn.Dropout(dropout)  

        # Register buffer offers advantages. Stores mask as part of the model — like a parameter, but non-trainable. 
        # Buffers are automatically moved to the GPU or CPU along with the model. When you call model.to('cuda'), the mask automatically moves to the GPU along with the model. So you dont have to manually move the mask to the GPU.
        # Included when you torch.save(model.state_dict())
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # x = input that is batch of input sequences.
    def forward(self, x):
        b, num_tokens, dim_in = x.shape  # b is the batch size, num_tokens is the number of tokens in input sequence, and dim_in is the input dimension.

        keys = self.W_key(x)  #uses built-in .forward()
        queries = self.W_query(x)  
        values = self.W_value(x) 

        # We transpose dimensions 1 and 2, keeping the batch dimension at the first position.
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply the causal mask to ensure that each token can only attend to previous tokens and itself.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.nn.functional.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)  
 
        attn_weights = self.dropout(attn_weights) 

        context_vector = attn_weights @ values 

        return context_vector
    
torch.manual_seed(123)  

# Using 3 dimensions for simplicity.
inputs = torch.tensor([
    [0.43, 0.15, 0.89],   #Your
    [0.55, 0.87, 0.66],   #journey
    [0.57, 0.85, 0.64],   #starts
    [0.22, 0.58, 0.33],   #with
    [0.77, 0.25, 0.10],   #one
    [0.05, 0.80, 0.55]    #step
])

batchedInputs = torch.stack([inputs, inputs], dim=0)  # Create a batch of 2 input sequences

ca = CausalAttention(3, 3, context_length=6, dropout=0.5) 
context_vector = ca(batchedInputs)

print("Context Vector for Batched Inputs:\n", context_vector)
print("Shape of Context Vector:", context_vector.shape)  # Should be (batch_size, num_tokens, dim_out)