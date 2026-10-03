
import torch
import torch.nn

# Here we will go over the concept of self-attention mechanism in transformers. This is a simplified version of self-attention to illustrate the concept.

# Example input sequence representing a sequence of 6 token embeddings. Using 3 dimensions for simplicity.
# See previous code on how to create token embeddings.
inputs = torch.tensor([
    [0.43, 0.15, 0.89],   #Your
    [0.55, 0.87, 0.66],   #journey
    [0.57, 0.85, 0.64],   #starts
    [0.22, 0.58, 0.33],   #with
    [0.77, 0.25, 0.10],   #one
    [0.05, 0.80, 0.55]    #step
])

torch.manual_seed(123)  # For reproducibility

# Step 1: Initialize the attention weights matrics as a learnable parameter. In training set requires_grad=True, in inference set requires_grad=False.
# Input dimensions are 3, output dimensions are 3 which is normally the case.

# nn.Parameter wrapper is crucial because it tells PyTorch that this tensor should be treated as a model parameter. 
# This means the tensor will automatically be included in the model's parameter list (accessible via model.parameters()), 
# moved to the appropriate device when you call model.to(device), and most importantly, tracked for gradient computation during backpropagation.
# torch.randn part generates a tensor filled with random numbers drawn from a standard normal distribution (mean=0, std=1). 
# When requires_grad=True, PyTorch will compute gradients for this tensor during the backward pass, allowing optimizers to update its values during training.
W_query = torch.nn.Parameter(torch.randn(3, 3), requires_grad=True) # Shape is (3, 3)
W_key = torch.nn.Parameter(torch.randn(3, 3), requires_grad=True)
W_value = torch.nn.Parameter(torch.randn(3, 3), requires_grad=True)

# Compute the query, key, and value matrices by multiplying the input embeddings with the respective weight matrices.
# Each row in the query, key, and value matrices corresponds to a specific token in the input sequence.
query = inputs @ W_query  # Shape: (6, 3)
key = inputs @ W_key      # Shape: (6, 3)
value = inputs @ W_value  # Shape: (6, 3)  

print("Query Matrix:\n", query)
print("Key Matrix:\n", key)
print("Value Matrix:\n", value)

# Step 2: Compute the attention scores by taking the dot product of the query and key matrices.
# Each row in the attention scores matrix represents the attention scores for a specific token in the input sequence in relation to all other tokens.
# 1st row for "Your", then (1, 1) is attention score between "Your" and "Your". (1, 2) is attention score between "Your" and "journey" and so on.
# Transposing the key matrix allows us to compute the dot product between each query and all keys without using a loop.
attention_scores = query @ key.T  # Shape: (6, 6)
print("Attention Scores Matrix:\n", attention_scores)

# Step 3: Apply the softmax function to the attention scores to obtain the attention weights.
# This normalizes the scores so that they sum to 1 for each token, allowing us to interpret them as probabilities.
attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # Shape: (6, 6). No change, just normalization. All rows sum to 1.
print("Attention Weights Matrix:\n", attention_weights)

# Step 4: Compute the Context vector by multiplying the attention weights with the value matrix.
# This step combines the value information based on the attention weights, resulting in a weighted sum of the value vectors.
# Each row in the attention output matrix represents the context vector for a specific token in the input sequence. 
# For example, the first row corresponds to the context vector for "Your". 
attention_output = attention_weights @ value  # Shape: (6, 3). Same as input. 6 tokens with 3 dimensions each.
print("Context Vector Matrix:\n", attention_output)