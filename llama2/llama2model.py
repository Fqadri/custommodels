
from torch import nn

from .rmsnormalization import RMSNorm
from .transformer_block import TransformerBlock

# Llama 1 and 2 model essentially are same with 2 supporting higher context length. When
# When compared to GPT2 it replaced:
#  a) the absolute positional embeddings with RoPE (Rotary Positional Embeddings)
#  b) the activation function with Gated SwiGLU in the feed-forward layer.
#  c) the LayerNorm with RMSNorm.

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = RMSNorm(
            cfg["emb_dim"], eps=cfg.get("norm_eps", 1e-5), dtype=cfg["dtype"]
        )
        
        # Output Layer (also known as LM Head)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        # batch_size, seq_len = in_idx.shape
        if in_idx.shape[-1] > self.context_length:
            raise ValueError(
                f"Input exceeds the model's context length of {self.context_length} tokens."
            )
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds  # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits