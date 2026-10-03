import json
import time
from pathlib import Path

import tiktoken
import torch
from tqdm import tqdm

from ..model._12_gpt2model import (
    generate_text_from_inputsample_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
)
from ..pretraining_eval._2_trainingandevaluation import (
    evaluate_model,
    plot_losses,
    train_model_simple,
)
from ..save_load_model.gpt_download import DEFAULT_MODELS_DIR
from ._1_rawdataandpromptformatconcept import (
    DEFAULT_DATA_PATH,
    InstructionEntry,
    format_input,
    load_data,
)
from ._2_preparedatasetandloadmodel import (
    create_train_val_test_dataloaders,
    get_pretrained_model,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "gpt2" / "instruction"
)


def print_initial_loss(train_loader, val_loader, model, device, num_batches=5):
    train_loss, val_loss = evaluate_model(
        model, train_loader, val_loader, device, eval_iter=num_batches
    )
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


def finetune_model_for_instruction_following(
    model,
    device,
    train_loader,
    val_loader,
    tokenizer,
    start_context="",
    num_epochs=2,
):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    start_time = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    losses = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
    )
    elapsed_minutes = (time.perf_counter() - start_time) / 60
    print(f"Fine-tuning completed in {elapsed_minutes:.2f} minutes.")
    return losses


def save_test_responses(
    model,
    test_data: list[InstructionEntry],
    tokenizer,
    device,
    context_length: int,
    output_path: Path,
    max_new_tokens: int = 256,
) -> None:
    if not test_data or max_new_tokens <= 0 or context_length <= 0:
        raise ValueError(
            "Test data must be non-empty and token limits must be positive."
        )
    responses = []
    model.eval()
    for index, entry in enumerate(tqdm(test_data, desc="Generating test responses")):
        input_text = format_input(entry)
        input_ids = text_to_token_ids(input_text, tokenizer).to(device)
        token_ids = generate_text_from_inputsample_with_sampling(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_length,
            eos_id=tokenizer.eot_token,
        )
        response_text = (
            token_ids_to_text(token_ids[:, input_ids.shape[1] :], tokenizer)
            .strip()
            .removeprefix("### Response:")
            .strip()
        )
        if index < 3:
            print(input_text)
            print("\nCorrect Response:\n", entry["output"])
            print("\nModel Response:\n", response_text)
            print("-" * 60)
        responses.append({**entry, "model_response": response_text})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(responses, file, indent=4, ensure_ascii=False)
    print(f"Model responses saved to {output_path}")


def run_finetuning(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_size: str = "355M",
    batch_size: int = 8,
    num_epochs: int = 2,
    context_length: int = 1024,
    num_workers: int = 0,
    device_name: str = "auto",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_new_tokens: int = 256,
) -> Path:
    if num_epochs <= 0 or max_new_tokens <= 0:
        raise ValueError("Epoch and generated-token counts must be positive.")
    if not 1 <= context_length <= 1024:
        raise ValueError("GPT-2 context_length must be between 1 and 1024.")
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device_name must be auto, cpu, or cuda.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available in this PyTorch environment."
        )
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_data, val_data, test_data, train_loader, val_loader, _ = (
        create_train_val_test_dataloaders(
            load_data(data_path),
            batch_size=batch_size,
            tokenizer=tokenizer,
            device="cpu",
            allowed_max_length=context_length,
            num_workers=num_workers,
        )
    )
    print(f"Device: {device}")
    print(
        f"Train: {len(train_data)}, validation: {len(val_data)}, test: {len(test_data)}"
    )
    model = get_pretrained_model(model_size, models_dir=models_dir)
    model.to(device)
    print("\nInitial losses before instruction fine-tuning:")
    print_initial_loss(train_loader, val_loader, model, device)

    print("\nStarting instruction fine-tuning:")
    train_losses, val_losses, tokens_seen = finetune_model_for_instruction_following(
        model,
        device,
        train_loader,
        val_loader,
        tokenizer,
        start_context=format_input(val_data[0]),
        num_epochs=num_epochs,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"gpt2-{model_size}-sft.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Fine-tuned model saved to {checkpoint_path}")

    epochs_seen = (torch.arange(len(train_losses)) * 5 + 1) / len(train_loader)
    plot_losses(
        epochs_seen,
        tokens_seen,
        train_losses,
        val_losses,
        output_path=output_dir / "losses.png",
    )
    save_test_responses(
        model,
        test_data,
        tokenizer,
        device,
        context_length,
        output_dir / "instruction-data-with-response.json",
        max_new_tokens=max_new_tokens,
    )
    return checkpoint_path


if __name__ == "__main__":
    import sys

    from .main import main

    main(["train", *sys.argv[1:]])
