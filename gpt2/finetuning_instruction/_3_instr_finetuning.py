import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common.generation import generate_text_from_inputsample_with_sampling
from common.pretraining import (
    evaluate_model,
    plot_losses,
    train_model_simple,
)
from common.tokenization import text_to_token_ids, token_ids_to_text

from ..model._12_gpt2model import GPT2Model
from ..model.gpt2_tokenizer import GPT2Tokenizer
from ..save_load_model._2_gpt2_download_modeldata import (
    DEFAULT_MODELS_DIR,
    get_gpt2_pretrained_modeldata,
)
from ..save_load_model._3_load_gpt2_modelweights import load_weights_into_gpt
from ._1_prompt_data_format import (
    DEFAULT_DATA_PATH,
    InstructionEntry,
    format_input_to_alpaca_prompt,
    load_data,
)
from ._2_prepare_instr_ft_dataset import (
    create_train_val_test_dataloaders,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "gpt2" / "instruction"
)

MODEL_CONFIGS = {
    "124M": {"dim_model": 768, "num_heads": 12, "num_layers": 12},
    "355M": {"dim_model": 1024, "num_heads": 16, "num_layers": 24},
    "774M": {"dim_model": 1280, "num_heads": 20, "num_layers": 36},
    "1558M": {"dim_model": 1600, "num_heads": 25, "num_layers": 48},
}

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
    *,
    context_size,
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
        context_size=context_size,
    )
    elapsed_minutes = (time.perf_counter() - start_time) / 60
    print(f"Fine-tuning completed in {elapsed_minutes:.2f} minutes.")
    return losses

def _generate_instruction_response(
    model,
    input_text: str,
    tokenizer,
    device,
    context_length: int,
    max_new_tokens: int,
) -> str:
    model.eval()
    input_ids = text_to_token_ids(input_text, tokenizer).to(device)
    token_ids = generate_text_from_inputsample_with_sampling(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        eos_id=tokenizer.eot_token,
    )
    return (
        token_ids_to_text(token_ids[:, input_ids.shape[1] :], tokenizer)
        .strip()
        .removeprefix("### Response:")
        .strip()
    )

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
    for index, entry in enumerate(tqdm(test_data, desc="Generating test responses")):
        input_text = format_input_to_alpaca_prompt(entry)
        response_text = _generate_instruction_response(
            model, input_text, tokenizer, device, context_length, max_new_tokens
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
    settings, params = get_gpt2_pretrained_modeldata(model_size, models_dir=models_dir)
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
    load_weights_into_gpt(model, params, hyperparameters=settings)

    return model.eval()

def run_instruction_finetuning(
    prompt: str = "Every effort moves you",
    max_new_tokens: int = 256,
    data_path: str | Path = DEFAULT_DATA_PATH, # path to instruction dataset
    model_size: str = "124M",
    run_test_data: bool = False,
) -> Path:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer.")

    batch_size = 8
    num_epochs = 2
    context_length = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(123)

    # Get GPT2 Custom model
    tokenizer = GPT2Tokenizer()
    model = get_pretrained_model(model_size, models_dir=DEFAULT_MODELS_DIR)
    model.to(device)
    
    input_text = format_input_to_alpaca_prompt(InstructionEntry(instruction=prompt, input="", output="")
                                               )
    print(f"\nDevice: {device}")
    print("\nPrompt:", prompt)

    print("\n-------------Before instruction fine-tuning:----------------")
    print("\nModel Response Before Fine-Tuning::")
    print(
        _generate_instruction_response(
            model, input_text, tokenizer, device,
            model.position_embedding.num_embeddings, max_new_tokens,
        )
    )

    train_data, val_data, test_data, train_loader, val_loader, _ = (
        create_train_val_test_dataloaders(
            load_data(data_path),
            batch_size=batch_size,
            tokenizer=tokenizer,
            device="cpu",
            allowed_max_length=context_length,
            num_workers=0,
        )
    )

    print("\n---------------------------------------------------------")
    
    print(f"\n\nTrain: {len(train_data)}, validation: {len(val_data)}, test: {len(test_data)}")
    print("\nInitial losses before instruction fine-tuning:")
    print_initial_loss(train_loader, val_loader, model, device)

    print("starttime:", time.strftime("%Y-%m-%d %H:%M:%S"), "Starting instruction fine-tuning:")
    
    train_losses, val_losses, tokens_seen = finetune_model_for_instruction_following(
        model,
        device,
        train_loader,
        val_loader,
        tokenizer,
        start_context=format_input_to_alpaca_prompt(val_data[0]),
        num_epochs=num_epochs,
        # Epoch samples retain the model's full context, independent of training truncation.
        context_size=model.position_embedding.num_embeddings,
    )

    print("\n\nInstruction fine-tuning completed at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n-------------After instruction fine-tuning:----------------")
    print("\nModel Response After Fine-Tuning::")
    print(
        _generate_instruction_response(
            model, input_text, tokenizer, device,
            model.position_embedding.num_embeddings, max_new_tokens,
        )
    )

    print("\n---------------------------------------------------------")

    # Save the fine-tuned model checkpoint
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"gpt2-{model_size}-sft.pth"
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f"\n\nFine-tuned model saved to {checkpoint_path}")

    epochs_seen = (torch.arange(len(train_losses)) * 5 + 1) / len(train_loader)

    print(f"\n\nPlotting training and validation losses")

    plot_losses(
        epochs_seen,
        tokens_seen,
        train_losses,
        val_losses,
        output_path=output_dir / "losses.png",
    )

    if run_test_data:
        print(f"\n\n------Saving test responses in {output_dir / 'instruction-data-with-response.json'}------")
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
