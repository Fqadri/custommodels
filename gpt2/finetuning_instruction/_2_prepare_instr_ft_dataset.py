from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ..model.gpt2_tokenizer import GPT2Tokenizer

from ._1_prompt_data_format import (
    DEFAULT_DATA_PATH,
    format_input_to_alpaca_prompt,
    load_data,
)

# Here we will go over preparing training dataset for supervised instruction finetuning and generate response using pretrained model.

# This class takes in the raw data entries, formats them into prompt style format for the model and pretokenizes them.
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        # for each raw data entry, format it into prompty style format (here Alpaca) and then tokenize it.
        for entry in data:
            instruction_plus_input = format_input_to_alpaca_prompt(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )  # Input sequence is both instruction+input+response tokenized.

    def __len__(self):
        return len(self.data)  # return the number of entries in the dataset

    def __getitem__(self, idx):
        return self.encoded_texts[
            idx
        ]  # return token ids corresponding to the input sequence at index idx


# Collate function combines a list of input sequences into a batch. We do custom to handle sophistication around Padding and Truncation.
# batch - list of token id sequences (lists of integers) for each entry in the batch.
# pad_token_id - token id used for padding shorter sequences. Also used as end-of-sequence (EOS) token.
# ignore_index - index to be ignored in the loss computation (usually for padding tokens).
# allowed_max_length - maximum allowed length for sequences (if any).
# device - device to which the tensors should be moved (e.g., 'cpu' or 'cuda').
def custom_collate(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    if not batch or any(not item for item in batch):
        raise ValueError("A batch must contain non-empty token sequences.")
    if allowed_max_length is not None and allowed_max_length <= 0:
        raise ValueError("allowed_max_length must be positive.")
    batch_max_length = max(
        len(item) + 1 for item in batch
    )  # Find the longest sequence in the batch.
    inputs_list = []
    targets_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # Add an end-of-sequence token.

        padded = new_item + [pad_token_id] * (
            batch_max_length - len(new_item)
        )  # Pad the sequence to the max length.
        inputs = torch.tensor(
            padded[:-1]
        )  # Convert input sequence to tensor with the last token truncated.
        targets = torch.tensor(padded[1:])  # Target sequence shifted by one position.

        # For Target sequence,
        # Mask out all (replace with ignore_index) EXECPT the first padding tokens in targets by ignore_index. This helps avoid computing loss on them which unecessarily increases the loss otherwise. Good to have only meaningful tokens contribute to the loss.
        # Retaining the last padding token allows the model to learn when to generate an end-of-sequence token in response to instructions which we can use for stopping criteria during text generation.
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally, truncate sequences to allowed_max_length if specified. This is useful if you plan to work with datasets that exceed the 1024 token context size supported by GPT2 based models.
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(
        device
    )  # Convert entire input list of padded sequences to a tensor and move to device.
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor

def create_train_val_test_dataloaders(
    data, batch_size, tokenizer, device, allowed_max_length=1024, num_workers=0
):

    if batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "batch_size must be positive and num_workers must be non-negative."
        )
    if allowed_max_length is not None and allowed_max_length <= 0:
        raise ValueError("allowed_max_length must be positive.")
    if num_workers and torch.device(device).type != "cpu":
        raise ValueError(
            "Use CPU collation with worker processes; move batches in the training loop."
        )

    train_portion = int(0.85 * len(data))  # 85% for training
    test_portion = int(0.10 * len(data))  # 10 % for testing.

    train_data = data[:train_portion]  # Pick 85% of the entries for training.
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    if not train_data or not test_data or not val_data:
        raise ValueError(
            "The dataset must have at least 10 entries for non-empty train/test/validation splits."
        )
    if len(train_data) < batch_size:
        raise ValueError(
            "batch_size exceeds the training split size; reduce batch_size."
        )
    collate_fn = partial(
        custom_collate,
        allowed_max_length=allowed_max_length,
        pad_token_id=tokenizer.eot_token,  #PAD token is EOT/EOS token in GPT2.
        device=device,
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_data, val_data, test_data, train_loader, val_loader, test_loader

def prepare_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    batch_size: int = 8,
    context_length: int = 1024,
    num_workers: int = 0,
) -> None:
    torch.manual_seed(123)
    tokenizer = GPT2Tokenizer()
    train_data, val_data, test_data, train_loader, val_loader, test_loader = (
        create_train_val_test_dataloaders(
            load_data(data_path),
            batch_size,
            tokenizer,
            "cpu",
            allowed_max_length=context_length,
            num_workers=num_workers,
        )
    )
    print(
        f"Train size: {len(train_data)}, Test size: {len(test_data)}, Validation size: {len(val_data)}"
    )
    for name, loader in (
        ("Train", train_loader),
        ("Validation", val_loader),
        ("Test", test_loader),
    ):
        inputs, targets = next(iter(loader))
        print(
            f"{name} batch: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}"
        )

def print_train_val_test_dataset() -> None:

    torch.manual_seed(123)

    # Create input sequences of varying lengths
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)

    inputs, targets = custom_collate(batch)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer()
    num_workers = 0
    batch_size = 8
    data = load_data()

    train_data, val_data, test_data, train_loader, _val_loader, _test_loader = (
        create_train_val_test_dataloaders(
            data,
            batch_size,
            tokenizer,
            "cpu",
            allowed_max_length=1024,
            num_workers=num_workers,
        )
    )

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}, Validation size: {len(val_data)}")

    for dset, name in zip(
        [train_data, test_data, val_data], ["Train", "Test", "Validation"]
    ):
        print(f"\n{name} Dataset Example Entry:\n", dset[0])

    print("\nTrain Loader:")
    for inputs, targets in train_loader:
        # Should print (batch_size, sequence_length) for both inputs and targets. Sequence length may vary due to padding.
        print(inputs.shape, targets.shape)  

    input_text = format_input_to_alpaca_prompt(val_data[0])  
    
    # Format the first entry of validation data into prompt style format.
    print("Input text:\n", input_text)
