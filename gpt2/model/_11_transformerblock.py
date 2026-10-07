

import torch

from common.multi_head_attention import MultiHeadAttention
from common.transformer_block import TransformerBlock as SharedTransformerBlock

from . import _8_layernormalization as ln
from . import _9_feedforward_gelu as ff


# GPT-2 supplies attention, GELU feed-forward, and LayerNorm to the shared block.
class TransformerBlock(SharedTransformerBlock):
    def __init__(
            self,
            cfg, 
            debug=False,
            dtype = None) : 

        # Create components for the transformer block: attention, feed-forward network, and layer norms.
        attention = MultiHeadAttention(
            dim_in=cfg["dim_model"], 
            dim_out=cfg["dim_model"], 
            context_length=cfg["context_length"], 
            dropout=cfg["drop_rate"], 
            num_heads=cfg["num_heads"], 
            qkv_bias=cfg["qkv_bias"],
            debug=debug,
            dtype=dtype # dtype setting to be able to instantiate the model with a lower precision later 
        )

        feed_forward = ff.FeedForward(cfg, debug=debug, dtype=dtype)
        
        norm1 = ln.LayerNorm(cfg["dim_model"], dtype=dtype)
        norm2 = ln.LayerNorm(cfg["dim_model"], dtype=dtype)
        
        # Initialize the shared transformer block with the specified components.
        super().__init__(
            attention=attention,
            ff=feed_forward,
            norm1=norm1,
            norm2=norm2,
            drop_rate=cfg["drop_rate"],
        )


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