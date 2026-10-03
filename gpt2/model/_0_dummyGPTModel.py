
import torch
import torch.nn as nn
import tiktoken

# Here we will create a dummy GPT model to illustrate the architecture of GPT models. This is not a functional model but serves to show how the components fit together.

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["dim_model"]) # Create token embedding layer.
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["dim_model"]) # Create position embedding layer.
        
        self.dropout = nn.Dropout(cfg["drop_rate"]) # Dropout layer to prevent overfitting.
        
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )# Stack of transformer encoder layers.

        self.final_norm = DummyLayerNorm(cfg["dim_model"]) # Layer normalization before the output layer.
        
        # Output layer to project back to vocabulary size. Since, final prediction (probabilities) is a token from the vocabulary.
        self.output_layer = nn.Linear(cfg["dim_model"], cfg["vocab_size"])

    # in_idx shape: (b, num_tokens). Input that is batch of input sequences. Each sequence contains tokens
    # We have to do manual implementation of forward here because it involves transformer specific operations.
    def forward(self, in_idx):
        b, context_length = in_idx.shape # b is batch i.e. rows, num_tokens is columns
        
        #Input Embedding Layer
        token_embedding = self.token_embedding(in_idx) # Get token embeddings for input indices.
        pos_embedding = self.position_embedding(
            torch.arange(context_length, device=in_idx.device)) # Get position embeddings. .device ensures it is on same device as in_idx.
        
        input_embedding = token_embedding + pos_embedding # Get input embedding for the sequences.
        
        print("Input Embedding shape:", input_embedding.shape)  # Should print (B, num_tokens - T, dim_model - d_model)
        print("Input Embedding:", input_embedding)  # Print the input embeddings.
        
        #Dropout layer to prevent overfitting.
        input_embedding = self.dropout(input_embedding) # Apply dropout to input embeddings to prevent overfitting.

        print("Input Embedding after Dropout shape:", input_embedding.shape)  # Should print (B, num_tokens - T, dim_model - d_model)
        print("Input Embedding after Dropout:", input_embedding)  # Print the input embeddings.

        #Transformer Blocks
        input_embedding = self.transformer_blocks(input_embedding) # Pass through transformer blocks.

        #Final Layer Normalization
        input_embedding = self.final_norm(input_embedding) # Apply layer normalization.

        # Output Layer that projects to vocabulary size. Shape is (B, num_tokens - T, vocab_size)
        # Each row, in batch, represents the tokens in the batch sequence and each column represents the vocab tokens.
        logits = self.output_layer(input_embedding) 
        
        return logits #logit is the raw, unnormalized scores outputted by the last layer of the model. After softmax activation these become probabilities.

# Transformer block contains 
# a) multi-head self-attention and dropout,
# b) feed-forward neural network
# c) layer normalization, and 
# d) shortcut connections.
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    #x is the input to the transformer block. Shape (batch of input sequences - B, num_tokens - T, dim_model - d_model)
    def forward(self, x):
        return x
        
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Size of the vocabulary.
    "dim_model": 768,    # Dimension of the model (d_model).
    "num_layers": 12,    # Number of transformer layers or blocks.
    "num_heads": 12,     # Number of attention heads per layer.
    "context_length": 1024,  # Maximum context length (number of tokens in input sequence).
    "drop_rate": 0.1,     # Dropout rate to prevent overfitting.
    "qkv_bias": False       # Whether to include bias terms in Q, K, V projections.
}

# Input sequence length or batch size
B = 1
batch = []
tokenizer = tiktoken.get_encoding("gpt2")  # Using GPT-2 tokenizer

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))

if B == 2:
    batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)  # Shape (B, num_tokens)
print("Input batch:", batch)  # Should print (B, num_tokens)


torch.manual_seed(123)  # For reproducibility
model = DummyGPTModel(GPT_CONFIG_124M)  # Instantiate the model with the configuration.
logits = model(batch)  # Forward pass through the model to get logits.
print("Logits shape:", logits.shape)  # Should print (B, num_tokens - T, vocab_size)
print("Logits:", logits)  # Print the logits.