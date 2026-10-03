import torch
import torch.nn as nn

# Here we will go over the concept of Multi-Head Attention.

# First define a single head of Causal Attention.
class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias) 
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)   
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias) 

        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting during training
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # x = input that is batch of input sequences.
    # forward method is called num_heads times in the MultiHeadAttentionSimple class. Is inefficient but simple.
    def forward(self, x):
        b, num_tokens, dim_in = x.shape

        # Shape of each of the following tensors will be (b, num_tokens, dim_out).
        keys = self.W_key(x)  
        queries = self.W_query(x)  
        values = self.W_value(x) 

        attn_scores = queries @ keys.transpose(1, 2)
        
        # Apply the causal mask to ensure that each position can only attend to previous positions (including itself)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.nn.functional.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)  
 
        attn_weights = self.dropout(attn_weights) 

        context_vector = attn_weights @ values 
        return context_vector


# Wrapper class for multiple attention heads.
# This class allows us to create multiple attention heads and concatenate their outputs.
# It is a simplified version of the MultiHeadAttention class.
class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Create linear (or stacked) layers of CausalAttention heads.
        self.heads = nn.ModuleList(
            [
                CausalAttention(dim_in, dim_out, context_length, dropout, qkv_bias) 
                for _ in range(num_heads)  # Create num_heads attention heads
            ])

    def forward(self, x):
         # Concatenate the outputs of all attention heads. These are however processed SEQUENTIALLY, not in parallel.
         # The CausalAttention class processes the input sequence independently and results from each head are concatenated.
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)  # For reproducibility

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

context_length = batchedInputs.shape[1]  # Number of tokens in each input sequence

# Create an instance of the MultiHeadAttentionSimple class with 2 heads
mha = MultiHeadAttentionSimple(dim_in=3, dim_out=3, context_length=context_length, dropout=0.5, num_heads=2)  

# Compute the context vectors for the batch of input sequences
context_vectors = mha(batchedInputs)  

# Should be (2, 6, 6). 2 is for the 2 input sequences matrices. Basically, (batch_size, num_tokens, dim_out X num_heads).
# 6 rows each is for each token in the input sequence and 
# final 6 is for the concatenated output of 2 heads, each with 3 dimensions.
print("\nContext Vectors Shape:", context_vectors.shape)  
print("\nContext Vectors:\n", context_vectors) 
