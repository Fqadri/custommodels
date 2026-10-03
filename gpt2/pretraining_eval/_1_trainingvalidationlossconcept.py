import os

import tiktoken
import torch

# Here we will go over training and validation loss concept.
# We compute training loss while training and then we compute validation loss on unseen data to see how well the model is doing.
from torch.utils.data import DataLoader, TensorDataset

from ..model._12_gpt2model import GPT2Model


# Calculate loss for a single batch.
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(
        device
    )  # Send the input and target batches to the device (CPU or GPU).
    target_batch = target_batch.to(device)

    logits = model(
        input_batch
    )  # logits shape: (b, num_tokens, vocab_size).  PyTorch builds a computation graph automatically during the forward pass.

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(
            0, 1
        ),  # Flatten the logits to shape (b * num_tokens, vocab_size) because we calculate the overall loss for all tokens in the batch.
        target_batch.flatten(),
    )  # Flatten the target to shape (b * num_tokens)

    return loss


# Calculate loss over an entire data loader (multiple batches).
# Goal is to train the model so that this loss is minimized (close to 0)
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        raise ValueError("Cannot evaluate loss on an empty data loader.")
    elif num_batches is not None and num_batches <= 0:
        raise ValueError("num_batches must be positive.")
    elif num_batches is None:
        num_batches = len(
            data_loader
        )  # If num_batches is not specified, we iterate over all batches in the data loader.
    else:
        num_batches = min(
            num_batches, len(data_loader)
        )  # Reduce the number of batches to match the total number of batches in data loader.

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()  # Sum loss for each batch.

    return total_loss / num_batches


def create_dataloader_v1(
    text, batch_size, context_size, stride, shuffle=True, drop_last=True, num_workers=0
):
    if context_size <= 0 or stride <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "Batch, context, and stride sizes must be positive; workers must be non-negative."
        )
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"}), dtype=torch.long
    )
    if len(token_ids) <= context_size:
        raise ValueError("Text must contain more tokens than the context size.")
    windows = token_ids.unfold(0, context_size + 1, stride)
    dataset = TensorDataset(windows[:, :-1], windows[:, 1:])
    if drop_last and len(dataset) < batch_size:
        raise ValueError("Not enough text for a complete training batch.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def create_train_val_dataloaders(context_length, batch_size):
    file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")
    with open(file_path, "r") as file:
        raw_text = file.read()

    # Text / Tokens are too small for training but this is for illustration only.
    # In practice, you would use a much larger dataset or load a pre-existing weights for the model.
    tokenizer = tiktoken.get_encoding("gpt2")
    total_characters = len(raw_text)
    total_tokens = len(tokenizer.encode(raw_text))
    print("Total characters in text:", total_characters)
    print("Total tokens in text:", total_tokens)

    # Split the data into training and validation sets. Use 90% of data for training, 10% for validation (for model evaluation)
    train_ratio = 0.90
    train_size = int(train_ratio * total_characters)

    train_data = raw_text[:train_size]
    val_data = raw_text[train_size:]

    # Now, we use data loaders to create batches of data for training and validation.
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,  # In practice, training with batch sizes of 1024 or larger is common.
        context_size=context_length,
        stride=context_length,  # no overlap
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        context_size=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader, tokenizer


# Example usage
# Training and Validation Loss Concept. We compute training loss while training and then we compute validation loss on unseen data to see how well the model is doing.
# In other words, We train the model to get to good weights and then we evaluate model i.e. validate on unseen data to see how well it performs.
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

    train_loader, val_loader, _tokenizer = create_train_val_dataloaders(
        GPT2_CONFIG_124M["context_length"], 2
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


if __name__ == "__main__":
    main()
