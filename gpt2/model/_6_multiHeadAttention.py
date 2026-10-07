
import torch

from common.multi_head_attention import MultiHeadAttention

if __name__ == "__main__":
    # Example usage:
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

    batch_size, context_length, dim_in = batchedInputs.shape

    mha = MultiHeadAttention(dim_in=3, dim_out=6, context_length=context_length, dropout=0.5, num_heads=2)  

    # Compute the context vectors for the batch of input sequences
    context_vectors = mha(batchedInputs)

    print(f"Shape of context_vectors: {context_vectors.shape}")
    print(f"context_vectors: {context_vectors}")