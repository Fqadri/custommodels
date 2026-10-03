

import torch
import torch.nn as nn

from . import _6_multiHeadAttention as mha
from . import _9_feedforward as ff
from . import _8_layernormalization as ln

# Here we define a single transformer block, which consists of a multi-head self-attention layer followed by a feed-forward neural network.
# Each of these components is followed by layer normalization and dropout for regularization.

class TransformerBlock(nn.Module) : 
    def __init__(self,cfg, debug=False) : 
        super().__init__()

        self.attention = mha.MultiHeadAttention(
            dim_in=cfg["dim_model"], 
            dim_out=cfg["dim_model"], 
            context_length=cfg["context_length"], 
            dropout=cfg["drop_rate"], 
            num_heads=cfg["num_heads"], 
            qkv_bias=cfg["qkv_bias"],
            debug=debug
        )

        self.ff = ff.FeedForward(cfg, debug=debug) # Also referred to as MLP
        self.norm1 = ln.LayerNorm(cfg["dim_model"])
        self.norm2 = ln.LayerNorm(cfg["dim_model"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    # x - input to the transformer block. Shape (batch of input sequences - B, num_tokens - T, dim_model - d_model)
    # LayerNorm is applied before self-attention and feed-forward network components. This is known as Pre-LN Transformer.
    # Older architectures such as the original transformer model applied LayerNorm after these components. This is known as Post-LN Transformer. Worse training.
    def forward(self, x) :
        shortcut = x
        x = self.norm1(x)

        x = self.attention(x) # shape (B, T, d_model)
        x = self.dropout(x)

        x = x + shortcut       # Add the original input (shortcut connection)

        shortcut = x           # Shortcut connection to feed-forward network.
        x = self.norm2(x)
        
        x = self.ff(x)         # Also referred to as MLP
        x = self.dropout(x)
        
        x = x + shortcut       # Add the original input (shortcut connection)

        return x


if __name__ == "__main__":
    # Example usage
    GPT2_CONFIG_124M = {
        "vocab_size": 50257,  # Size of the vocabulary.
        "dim_model": 768,    # Dimension of the model (d_model).
        "num_layers": 12,    # Number of transformer layers or blocks.
        "num_heads": 12,     # Number of attention heads per layer.
        "context_length": 1024,  # Maximum context length (number of tokens in input sequence).
        "drop_rate": 0.1,     # Dropout rate to prevent overfitting.
        "qkv_bias": False       # Whether to include bias terms in Q, K, V projections.
    }

    torch.manual_seed(123)  # For reproducibility
    batch_example = torch.randn(1, 2, 768) # Example batch of input sequences. Shape 1 X 2 X 768 (B X T X D). 1 input sequences each of length 2 tokens and 768 dimensions.

    block = TransformerBlock(GPT2_CONFIG_124M)
    out = block(batch_example)
    
    num_parameters_in_attention = sum(p.numel() for p in block.attention.parameters())
    num_parameters_in_ff = sum(p.numel() for p in block.ff.parameters())
    num_parameters_in_block = sum(p.numel() for p in block.parameters())

    print("Output shape after TransformerBlock:", out.shape)  #Should print (1, 2, 768)
    print("Output after TransformerBlock:", out)

    print("Total number of trainable parameters in attention layer of the block:", num_parameters_in_attention)
    print("Total number of trainable parameters in feed-forward layer of the block:", num_parameters_in_ff)
    print("Total number of trainable parameters in the entire transformer block:", num_parameters_in_block)