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

from ..model._12_gpt2model import GPT2Model
from ..model.gpt2_tokenizer import GPT2Tokenizer

DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("the-verdict.txt")


# Example usage
# Training and Validation Loss Concept. We compute training loss while training and then we compute validation loss on unseen data to see how well the model is doing.
# In other words, We train the model to get to good weights and then we evaluate model i.e. validate on unseen data to see how well it performs.
# This demonstration measures initial losses; it does not perform optimizer updates.
def demonstrate_training_validation_loss() -> None:

    GPT2_CONFIG_124M = {
        "vocab_size": 50257,
        "dim_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 256,  # Change to 256 for faster training and reducing computational costs.
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    tokenizer = GPT2Tokenizer()
    train_loader, val_loader, _tokenizer = create_train_val_dataloaders(
        GPT2_CONFIG_124M["context_length"], 2, tokenizer=tokenizer,
        data_path=DEFAULT_DATA_PATH,
    )

    # Should be 9 batches in train loader and 1 batch in val loader.
    print("\nTrain Loader:")
    for x, y in train_loader:
        print(
            x.shape, y.shape
        )  # x is input and y is target. Shape is (batch_size, context_size)

    print("\nValidation Loader:")
    for x, y in val_loader:
        print(
            x.shape, y.shape
        )  # x is input and y is target. Shape is (batch_size, context_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)  # For reproducibility
    model = GPT2Model(GPT2_CONFIG_124M, debug=False)
    model.to(device)

    with torch.no_grad():  # No need to track gradients for validation.
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}\n")


# Example usage of the model for training and evaluation.
def main() -> None:

    GPT2_CONFIG_124M = {
        "vocab_size": 50257,
        "dim_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "context_length": 256,  # Change to 256 for faster training and reducing computational costs.
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    torch.manual_seed(42)

    # Initialize the GPT-2 model with the specified configuration.
    model = GPT2Model(GPT2_CONFIG_124M)

    # AdamW is a popular optimizer for training transformer models.
    optimizer = torch.optim.AdamW(
        model.parameters(),  # returns all training weight parameters of the model
        lr=3e-4,  # learning rate. Common values are between 1e-3 and 1e-5. Helps control how much to change the model weights at each step.
        weight_decay=1e-1,  # helps prevent overfitting by penalizing large weights.
    )

    # Train Model for 10 epochs.
    num_epochs = 10
    start_context = "Every effort moves you"
    tokenizer = GPT2Tokenizer()
    train_losses, val_losses, tokens_seen, tokenizer = train_model(
        model, GPT2_CONFIG_124M["context_length"], optimizer, num_epochs, start_context,
        tokenizer=tokenizer,
        data_path=DEFAULT_DATA_PATH,
    )
    device = next(model.parameters()).device

    epochs_seen = torch.linspace(
        0, num_epochs, len(train_losses)
    )  # Create a tensor with evenly spaced values from 0 to num_epochs
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)

    # Run inference with the trained model.
    model.eval()
    token_ids = generate_text_from_inputsample(
        model,
        text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT2_CONFIG_124M["context_length"],
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text with greedy decoding:\n", generated_text.replace("\n", " "))

    # Using sampling to generate text after model training.
    token_ids = generate_text_from_inputsample_with_sampling(
        model,
        text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT2_CONFIG_124M["context_length"],
        temperature=0.5,  # Higher the temperature, more random the output. Lower the temperature, more deterministic the output.
        top_k=25,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(
        "Generated text with sampling:\n", generated_text.replace("\n", " ")
    )  # compact display by removing newlines


if __name__ == "__main__":
    main()
