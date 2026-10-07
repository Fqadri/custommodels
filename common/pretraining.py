from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, TensorDataset

from .generation import generate_text_from_inputsample
from .tokenization import text_to_token_ids, token_ids_to_text

# Here we will go over training loop for model including evaluation and text generation.
# After training we will have a Pre-trained model that is only good at Text Completion.

# Here we will go over training and validation loss concept.
# We compute training loss while training and then we compute validation loss on unseen data to see how well the model is doing.

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
    text, batch_size, context_size, stride, shuffle=True, drop_last=True, num_workers=0,
    *, tokenizer,
):
    if context_size <= 0 or stride <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "Batch, context, and stride sizes must be positive; workers must be non-negative."
        )
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
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

def create_train_val_dataloaders(
    context_length, batch_size, *, tokenizer, data_path: str | Path
):
    with open(data_path, "r") as file:
        raw_text = file.read()

    # Text / Tokens are too small for training but this is for illustration only.
    # In practice, you would use a much larger dataset or load a pre-existing weights for the model.
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
        tokenizer=tokenizer,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        context_size=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        tokenizer=tokenizer,
    )

    return train_loader, val_loader, tokenizer

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
def generate_and_print_sample(
    model, tokenizer, start_context, device, *, context_size
):
    # Use the caller's context size instead of inspecting model-specific layers.
    if context_size <= 0:
        raise ValueError("context_size must be positive.")
    was_training = model.training
    model.eval()
    try:
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
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
    *,
    context_size,
):
    if context_size <= 0:
        raise ValueError("context_size must be positive.")
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

        generate_and_print_sample(
            model, tokenizer, start_context, device, context_size=context_size
        )

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
    *,
    context_size,
):
    if context_size <= 0:
        raise ValueError("context_size must be positive.")
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
            generate_and_print_sample(
                model, tokenizer, start_context, device, context_size=context_size
            )

    # lists of training and validation losses, and tokens seen over time (based on evaluation frequency) for plotting.
    return train_losses, val_losses, track_tokens_seen


def train_model(
    model, context_length, optimizer, num_epochs, start_context, *,
    tokenizer, data_path: str | Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare next-token training data
    # Pre-training is self-supervised, where the model learns to predict the next token in a sequence.
    train_loader, val_loader, tokenizer = create_train_val_dataloaders(
        context_length=context_length, batch_size=2, tokenizer=tokenizer,
        data_path=data_path,
    )

    # Train the model using the simple training loop defined earlier.
    # During training, measure training and validation loss every eval_freq steps.
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
        context_size=context_length,
    )

    return train_losses, val_losses, tokens_seen, tokenizer

