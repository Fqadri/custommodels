from common.multi_head_attention import MultiHeadAttention
from common.transformer_block import TransformerBlock as SharedTransformerBlock

from .feedforward_swiglu import FeedForward
from .rmsnormalization import RMSNorm
from .rope import RotaryEmbedding


# LLaMA supplies bias-free rotary attention, SwiGLU, and RMSNorm to the shared block.
class TransformerBlock(SharedTransformerBlock):
    def __init__(self, cfg):
        emb_dim = cfg["emb_dim"]
        num_heads = cfg["n_heads"]
        if emb_dim <= 0 or num_heads <= 0 or emb_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be positive and divisible by a positive number of heads."
            )

        # Create components for the transformer block: attention, feed-forward network, and layer norms.
        rope = RotaryEmbedding(
            head_dim=emb_dim // num_heads,
            context_length=cfg["context_length"],
        )

        
        attention = MultiHeadAttention(
            dim_in=emb_dim,
            dim_out=emb_dim,
            context_length=cfg["context_length"],
            dropout=0.0,
            num_heads=num_heads,
            qkv_bias=False,
            dtype=cfg["dtype"],
            out_bias=False,
            rope=rope,
        )


        feed_forward = FeedForward(cfg)
        
        norm_eps = cfg.get("norm_eps", 1e-5)
        norm1 = RMSNorm(emb_dim, eps=norm_eps, dtype=cfg["dtype"])
        norm2 = RMSNorm(emb_dim, eps=norm_eps, dtype=cfg["dtype"])
        
        
        # Initialize the shared transformer block with the specified components.
        super().__init__(
            attention=attention,
            ff=feed_forward,
            norm1=norm1,
            norm2=norm2,
            drop_rate=0.0,
        )
