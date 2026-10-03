import tiktoken
import torch

from gpt2._12_gpt2model import (
    GPT2_CONFIG_124M,
    GPT2Model,
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
)


def main() -> None:
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT2Model(GPT2_CONFIG_124M, debug=False)
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
