from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

# Here we will go over training loop for model including evaluation and text generation.
# After training we will have a Pre-trained model that is only good at Text Completion.
from ..model._12_gpt2model import (
    GPT2Model,
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
)
from ._1_trainingvalidationlossconcept import (
    calc_loss_batch,
    calc_loss_loader,
    create_train_val_dataloaders,
)


# Helps evaluate whether the training improves the model. This is called Evaluation.
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    was_training = model.training
    model.eval()  # Set the model to evaluation mode
    try:
        with torch.no_grad():
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter
            )
    finally:
        model.train(was_training)
    return train_loss, val_loss


# Similar to evaluate_model (which gives numerical feedback), this function is a convienience function that provides text that we can use to track whether the model improves during training.
def generate_and_print_sample(model, tokenizer, start_context, device):
    was_training = model.training
    model.eval()
    context_size = model.position_embedding.weight.shape[
        0
    ]  # Access model attribute directly to get context size
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    try:
        with torch.no_grad():
            token_ids = generate_text_from_inputsample(
                model, encoded, max_new_tokens=50, context_size=context_size
            )
    finally:
        model.train(was_training)
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(
        "Generated text:\n", generated_text.replace("\n", " ")
    )  # compact display by removing newlines


def plot_losses(
    epochs_seen,
    tokens_seen,
    train_losses,
    val_losses,
    output_path: str | Path | None = None,
):
    fig, ax1 = plt.subplots(
        figsize=(5, 3)
    )  # Create a figure and axis object fig size is in inches
    ax1.plot(epochs_seen, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs_seen, val_losses, label="Validation Loss", color="orange")

    ax1.set_xlabel("Epochs Seen")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot to set the scale
    ax2.set_xlabel("Tokens Seen")
    fig.tight_layout()  # Adjust layout to prevent overlap
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path)
    plt.close(fig)


# Training look with enhancements in training.
# 1. Learning Warmup for first few epochs to stabilize training
# 2. Cosine Decay for learning rate to help model converge better
# 3. Gradient Clipping to prevent exploding gradients
def train_model_enhanced(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    warmup_steps=20,
    initial_lr=3e-05,
    min_lr=1e-6,
):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen = 0
    global_step = -1

    peak_lr = optimizer.param_groups[0][
        "lr"
    ]  # Get the initial learning rate from the optimizer and use it as peak lr.
    total_training_steps = len(train_loader) * num_epochs
    lr_increment = (
        (peak_lr - initial_lr) / warmup_steps if warmup_steps > 0 else 0
    )  # Calculate the increment for each warmup step

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            global_step += 1

            # Adjust learning rate based on warmup and cosine decay schedule. if in warmup phase, increase lr linearly. After warmup, apply cosine decay.
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment  # Linear warmup
            else:
                progress = (global_step - warmup_steps) / (
                    total_training_steps - warmup_steps
                )
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (
                    1 + torch.cos(torch.tensor(progress * 3.141592653589793))
                )  # Cosine decay (3.14159 is pi)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr  # Update the learning rate in the optimizer

            track_lrs.append(lr)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Gradient clipping to prevent exploding gradients. Only apply after warmup phase.
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Now that gradients are computed and clipped (if applicable), update the model weights.

            tokens_seen += input_batch.numel()

            # This is same as in simple loop. No changes.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}, Step {global_step:06d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        generate_and_print_sample(model, tokenizer, start_context, device)

    return train_losses, val_losses, track_tokens_seen, track_lrs


# Training loop with evaluation and text generation.
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,  # evaluate the model every eval_freq epochs
    eval_iter,  # number of batches to use, from train_loader, for evaluating model
    start_context,  # initial text to see how the model is doing
    tokenizer,
):
    if num_epochs <= 0 or eval_freq <= 0 or eval_iter <= 0:
        raise ValueError("Epoch count and evaluation intervals must be positive.")
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation data loaders must be non-empty.")
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Starts the main training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        # Use training data loader to get batches of data. Data Loader will shuffle the data and create batches and ensures that each sequence is seen only once per epoch.
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear previous gradients

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )  # Calculate loss for the current batch
            loss.backward()  # PyTorch will calculates the loss gradients w.r.t. all model parameters that have requires_grad=True and stores them in the .grad attribute of each parameter (every element).
            optimizer.step()  # Update model weights based on gradients stored in .grad attribute of each parameter

            tokens_seen += input_batch.numel()  # Count the number of tokens processed
            global_step += 1

            if (
                global_step % eval_freq == 0
            ):  # optional evaluation frequence to see how the model is doing
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}, Step {global_step:06d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        # generate text after each epoch to see how the model is doing
        if start_context:
            generate_and_print_sample(model, tokenizer, start_context, device)

    # lists of training and validation losses, and tokens seen over time (based on evaluation frequency) for plotting.
    return train_losses, val_losses, track_tokens_seen


def train_model(model, context_length, optimizer, num_epochs, start_context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, val_loader, tokenizer = create_train_val_dataloaders(
        context_length=context_length, batch_size=2
    )
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, tokenizer


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

    model = GPT2Model(GPT2_CONFIG_124M)

    # AdamW is a popular optimizer for training transformer models.
    optimizer = torch.optim.AdamW(
        model.parameters(),  # returns all training weight parameters of the model
        lr=3e-4,  # learning rate. Common values are between 1e-3 and 1e-5. Helps control how much to change the model weights at each step.
        weight_decay=1e-1,  # helps prevent overfitting by penalizing large weights.
    )

    num_epochs = 10
    start_context = "Every effort moves you"
    train_losses, val_losses, tokens_seen, tokenizer = train_model(
        model, GPT2_CONFIG_124M["context_length"], optimizer, num_epochs, start_context
    )
    device = next(model.parameters()).device

    epochs_seen = torch.linspace(
        0, num_epochs, len(train_losses)
    )  # Create a tensor with evenly spaced values from 0 to num_epochs
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)

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
