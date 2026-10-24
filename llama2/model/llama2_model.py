import torch
from torch import nn

from common.generation import (
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
)
from common.tokenization import text_to_token_ids, token_ids_to_text

from ..save_load_model.load_llama2_modelweights import (
    download_llama2,
    load_weights_into_llama,
)
from .llama2_tokenizer import LlamaTokenizer
from .rms_normalization import RMSNorm
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


LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,
    "context_length": 4096,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 11008,
    "dtype": torch.bfloat16,
}


def get_llama_model() -> tuple[Llama2Model, LlamaTokenizer]:
    tokenizer_file, _ = download_llama2()
    tokenizer = LlamaTokenizer(str(tokenizer_file))
    if tokenizer.vocab_size != LLAMA2_CONFIG_7B["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"the model requires {LLAMA2_CONFIG_7B['vocab_size']}."
        )

    model = Llama2Model(LLAMA2_CONFIG_7B)
    return model, tokenizer


def get_llama_model_using_weights(
    instructionfinetuned: bool = False,
) -> tuple[Llama2Model, LlamaTokenizer]:
    tokenizer_file, weight_file = download_llama2(
        instructionfinetuned=instructionfinetuned
    )
    tokenizer = LlamaTokenizer(str(tokenizer_file))
    if tokenizer.vocab_size != LLAMA2_CONFIG_7B["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"the model requires {LLAMA2_CONFIG_7B['vocab_size']}."
        )

    model = Llama2Model(LLAMA2_CONFIG_7B)
    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weight_file)
    return model, tokenizer


def run_llama2_model() -> None:
    torch.manual_seed(123)
    model, tokenizer = get_llama_model_using_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    model.eval()
    start_context = "Every effort moves you"

    # Greedy decoding produces the same result for both runs.
    for _ in range(2):
        token_ids = generate_text_from_inputsample(
            model,
            text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=LLAMA2_CONFIG_7B["context_length"],
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        print("Generated text:\n", generated_text)

    token_ids = generate_text_from_inputsample_with_sampling(
        model,
        text_to_token_ids(start_context, tokenizer),
        max_new_tokens=15,
        context_size=LLAMA2_CONFIG_7B["context_length"],
        temperature=0.5,
        top_k=25,
        eos_id=tokenizer.eos_id,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text with sampling:\n", generated_text)