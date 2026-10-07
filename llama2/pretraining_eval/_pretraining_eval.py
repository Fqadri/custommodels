import argparse
from pathlib import Path

import torch

from common.generation import (
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
)
from common.pretraining import (
    calc_loss_loader,
    create_train_val_dataloaders,
    plot_losses,
    train_model,
)
from common.tokenization import text_to_token_ids, token_ids_to_text

from ..llama2model import Llama2Model
from ..llamatokenizer import LlamaTokenizer

DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("the-verdict.txt")

# A LLaMA-style learning configuration at the GPT-2 demo's scale, not the 7B model.
LLAMA2_CONFIG_SMALL = {
    "vocab_size": 32000,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "hidden_dim": 2048,
    "dtype": torch.float32,
}


def _load_tokenizer(tokenizer_file: str | Path) -> LlamaTokenizer:
    tokenizer = LlamaTokenizer(tokenizer_file)
    if tokenizer.vocab_size != LLAMA2_CONFIG_SMALL["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocabulary has {tokenizer.vocab_size} entries; "
            f"the model requires {LLAMA2_CONFIG_SMALL['vocab_size']}."
        )
    return tokenizer


# Show initial train/validation losses without updating the model's weights.
def demonstrate_training_validation_loss(
    tokenizer_file: str | Path, data_path: str | Path = DEFAULT_DATA_PATH
) -> None:
    tokenizer = _load_tokenizer(tokenizer_file)
    train_loader, val_loader, _tokenizer = create_train_val_dataloaders(
        LLAMA2_CONFIG_SMALL["context_length"], 2,
        tokenizer=tokenizer, data_path=data_path,
    )

    print("\nTrain Loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    print("\nValidation Loader:")
    for inputs, targets in val_loader:
        print(inputs.shape, targets.shape)

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Llama2Model(LLAMA2_CONFIG_SMALL)
    model.to(device)
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain a small float32 LLaMA-style model, then generate text."
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True,
        help="Path to a 32,000-token LLaMA SentencePiece tokenizer.model file.",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help="Training corpus; defaults to this folder's copy of the-verdict.txt.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if not args.tokenizer.is_file():
        parser.error(f"Tokenizer file does not exist: {args.tokenizer}")
    if not args.data.is_file():
        parser.error(f"Training corpus does not exist: {args.data}")
    try:
        tokenizer = _load_tokenizer(args.tokenizer)
    except ValueError as error:
        parser.error(str(error))

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Place the model on its training device before constructing the optimizer.
    model = Llama2Model(LLAMA2_CONFIG_SMALL)
    model.to(device)
    print(f"Device: {device}")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-1)
    start_context = "Every effort moves you"

    # Pretrain from random weights, including periodic evaluation and epoch-end samples.
    train_losses, val_losses, tokens_seen, tokenizer = train_model(
        model,
        LLAMA2_CONFIG_SMALL["context_length"],
        optimizer,
        args.epochs,
        start_context,
        tokenizer=tokenizer,
        data_path=args.data,
    )

    epochs_seen = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)

    # Run greedy inference using the weights learned during pretraining.
    model.eval()
    token_ids = generate_text_from_inputsample(
        model,
        text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=15,
        context_size=LLAMA2_CONFIG_SMALL["context_length"],
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text with greedy decoding:\n", generated_text.replace("\n", " "))

    # Sample a second continuation, stopping if the tokenizer's EOS token is generated.
    token_ids = generate_text_from_inputsample_with_sampling(
        model,
        text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=15,
        context_size=LLAMA2_CONFIG_SMALL["context_length"],
        temperature=0.5,
        top_k=25,
        eos_id=tokenizer.eos_id,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(
        "Generated text with sampling:\n", generated_text.replace("\n", " ")
    )


if __name__ == "__main__":
    main()
