from pathlib import Path

import torch

from ..model._12_gpt2model import (
    GPT2Model,
    generate_text_from_inputsample,
    text_to_token_ids,
    token_ids_to_text,
)
from ..pretraining_eval._2_trainingandevaluation import train_model


def main() -> None:
    config = {
        "vocab_size": 50257,
        "dim_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 256,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    torch.manual_seed(42)
    model = GPT2Model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    start_context = "Every effort moves you"
    _, _, _, tokenizer = train_model(
        model, config["context_length"], optimizer, 10, start_context
    )
    device = next(model.parameters()).device
    model.eval()
    input_ids = text_to_token_ids(start_context, tokenizer).to(device)
    token_ids = generate_text_from_inputsample(
        model, input_ids, max_new_tokens=15, context_size=config["context_length"]
    )
    print(
        "Generated text with greedy decoding:\n",
        token_ids_to_text(token_ids, tokenizer),
    )

    checkpoint_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "gpt2"
        / "model_and_optimizer.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GPT2Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    token_ids = generate_text_from_inputsample(
        model, input_ids, max_new_tokens=15, context_size=config["context_length"]
    )
    print(
        "Generated text AFTER loading weights:\n",
        token_ids_to_text(token_ids, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()


if __name__ == "__main__":
    main()
