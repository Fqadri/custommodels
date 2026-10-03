from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..model._12_gpt2model import (
    GPT2Model,
    generate_text_from_inputsample_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
)
from ..save_load_model._3_loadopenaiweights import load_weights_into_gpt
from ..save_load_model.gpt_download import DEFAULT_MODELS_DIR, download_and_load_gpt2
from ._1_rawdataandpromptformatconcept import DEFAULT_DATA_PATH, format_input, load_data

MODEL_CONFIGS = {
    "124M": {"dim_model": 768, "num_heads": 12, "num_layers": 12},
    "355M": {"dim_model": 1024, "num_heads": 16, "num_layers": 24},
    "774M": {"dim_model": 1280, "num_heads": 20, "num_layers": 36},
    "1558M": {"dim_model": 1600, "num_heads": 25, "num_layers": 48},
}

# Here we will go over preparing training dataset for supervised instruction finetuning and generate response using pretrained model.


# This class takes in the raw data entries, formats them into prompt style format for the model and pretokenizes them.
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        # for each raw data entry, format it into prompty style format (here Alpaca) and then tokenize it.
        for entry in data:
            instruction_plus_input = format_input(entry)
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


def get_pretrained_model(
    model_size: str, models_dir: str | Path = DEFAULT_MODELS_DIR
) -> GPT2Model:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size {model_size!r}; choose from {tuple(MODEL_CONFIGS)}."
        )
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    BASE_CONFIG.update(MODEL_CONFIGS[model_size])
    print(f"Loading GPT-2 model weights for {model_size}...")
    settings, params = download_and_load_gpt2(model_size, models_dir=models_dir)
    expected = {
        "n_vocab": BASE_CONFIG["vocab_size"],
        "n_ctx": BASE_CONFIG["context_length"],
        "n_embd": BASE_CONFIG["dim_model"],
        "n_head": BASE_CONFIG["num_heads"],
        "n_layer": BASE_CONFIG["num_layers"],
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Checkpoint settings do not match GPT-2 {model_size}: {settings}"
        )

    model = GPT2Model(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    return model.eval()


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
        pad_token_id=tokenizer.eot_token,
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
    tokenizer = tiktoken.get_encoding("gpt2")
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


def main() -> None:

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
    tokenizer = tiktoken.get_encoding("gpt2")
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

    print(
        f"Train size: {len(train_data)}, Test size: {len(test_data)}, Validation size: {len(val_data)}"
    )

    for dset, name in zip(
        [train_data, test_data, val_data], ["Train", "Test", "Validation"]
    ):
        print(
            f"\n{name} Dataset Example Entry:\n", dset[0]
        )  # Print the first entry of each dataset split.

    print("\nTrain Loader:")
    for inputs, targets in train_loader:
        print(
            inputs.shape, targets.shape
        )  # Should print (batch_size, sequence_length) for both inputs and targets. Sequence length may vary due to padding.

    # Example of loading pretrained model and generating text.
    print("\nGetting Pretrained Model:")
    CHOOSE_MODEL = "355M"
    model = get_pretrained_model(CHOOSE_MODEL)
    model.to(device).eval()

    input_text = format_input(
        val_data[0]
    )  # Format the first entry of validation data into prompt style format.
    print("Input text:\n", input_text)
    token_ids = generate_text_from_inputsample_with_sampling(
        model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=35,
        context_size=1024,
        eos_id=50256,  # EOS token id for GPT-2. Same as pad_token_id.
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("------------Generated text--------------:\n", generated_text)

    # Extract the response part from the generated text.
    # When evaluating model's performance on specific task, we want to focus solely on the model's generated response and not the entire input prompt.
    # Notice without instruction fine-tuning, model often just parrots back the input instruction instead of generating a meaningful response.
    response_text = generated_text[len(input_text) :]
    print("-----------Response text---------------:\n", response_text)


if __name__ == "__main__":
    main()
