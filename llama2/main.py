import argparse
from pathlib import Path

import torch

from common.generation import (
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
)
from common.tokenization import text_to_token_ids, token_ids_to_text
from llama2.download_load_weights import download_llama2, load_weights_into_llama

from .llama2model import Llama2Model
from .llamatokenizer import LlamaTokenizer

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

def get_llama_model() -> tuple[Llama2Model, LlamaTokenizer]:
    # Get tokenizer for Llama from HF
    tokenizer_file, weight_file = download_llama2()
    
    tokenizer = LlamaTokenizer(str(tokenizer_file))  # Use the supplied SentencePiece tokenizer model file.
    if tokenizer.vocab_size != LLAMA2_CONFIG_7B["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"the model requires {LLAMA2_CONFIG_7B['vocab_size']}."
        )

    # Initialize custom LLaMA 2 model with the specified configuration
    model = Llama2Model(LLAMA2_CONFIG_7B)

    return model, tokenizer

# Gets Llama model using weights from HF.
def get_llama_model_using_weights(instructionfinetuned: bool = False) -> tuple[Llama2Model, LlamaTokenizer]:
    # Get tokenizer for Llama from HF
    tokenizer_file, weight_file = download_llama2(instructionfinetuned=instructionfinetuned)
        
    tokenizer = LlamaTokenizer(str(tokenizer_file))  # Use the supplied SentencePiece tokenizer model file.
    if tokenizer.vocab_size != LLAMA2_CONFIG_7B["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"the model requires {LLAMA2_CONFIG_7B['vocab_size']}."
        )
    
    model = Llama2Model(LLAMA2_CONFIG_7B)
    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weight_file)
    return model, tokenizer

def main(argv: list[str] | None = None) -> None:
    
    torch.manual_seed(123)
    
    # model, tokenizer = get_llama_model()

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

if __name__ == "__main__":
    main()

