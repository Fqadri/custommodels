import torch

from common.generation import (
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
)
from common.tokenization import text_to_token_ids, token_ids_to_text

from ._12_gpt2model import GPT2_CONFIG_124M, GPT2Model
from .gpt2_tokenizer import GPT2Tokenizer


def main(dtype: torch.dtype = torch.float32) -> None:
    torch.manual_seed(123)
    tokenizer = GPT2Tokenizer()
    model = GPT2Model(GPT2_CONFIG_124M, debug=False, dtype=dtype)
    model.eval()

    start_context = "Every effort moves you"

    # Greedy decoding produces the same result for both runs.
    for _ in range(2):
        token_ids = generate_text_from_inputsample(
            model,
            text_to_token_ids(start_context, tokenizer),
            max_new_tokens=10,
            context_size=GPT2_CONFIG_124M["context_length"],
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        print("Generated text:\n", generated_text)

    token_ids = generate_text_from_inputsample_with_sampling(
        model,
        text_to_token_ids(start_context, tokenizer),
        max_new_tokens=15,
        context_size=GPT2_CONFIG_124M["context_length"],
        temperature=0.5,
        top_k=25,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text with sampling:\n", generated_text)


if __name__ == "__main__":
    main()
