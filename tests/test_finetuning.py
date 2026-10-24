import inspect
import json
import os
import pickle
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from common import pretraining as training
from gpt2.finetuning_instruction import _1_prompt_data_format as data
from gpt2.finetuning_instruction import _2_prepare_instr_ft_dataset as preparation
from gpt2.finetuning_instruction import _3_instr_finetuning as finetuning
from gpt2.model._12_gpt2model import GPT2Model
from gpt2.pretraining_eval import _2_pretraining_eval as pretraining_demo


class TinyTokenizer:
    eot_token = 31

    def encode(self, text):
        return [1, 2, 3, 4]

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def test_common_pretraining_import_does_not_load_model_packages():
    script = """
import sys
import common.pretraining
assert not any(name.split(".")[0] in {"gpt2", "llama2"} for name in sys.modules)
assert not common.pretraining.plt.get_fignums()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=60,
    )
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.fixture
def tiny_model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = GPT2Model(
        {
            "vocab_size": 32,
            "dim_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "context_length": 8,
            "drop_rate": 0.0,
            "qkv_bias": True,
        }
    )
    yield model
    torch.set_num_threads(previous_threads)


@pytest.fixture
def dataset_path(tmp_path):
    entries = [
        {
            "instruction": f"Instruction {index}",
            "input": "",
            "output": f"Answer {index}",
        }
        for index in range(10)
    ]
    path = tmp_path / "instructions.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_imports_do_not_execute_demos():
    script = """
import importlib
import pkgutil
from unittest.mock import patch
import matplotlib.pyplot as plt
import torch
from gpt2.model._12_gpt2model import GPT2Model

def forbidden(*args, **kwargs):
    raise AssertionError("Imports must not run demos or download data")

with patch("requests.get", forbidden), patch("urllib.request.urlopen", forbidden), \
     patch.object(GPT2Model, "__init__", forbidden), patch.object(torch, "manual_seed", forbidden), \
     patch.object(plt, "show", forbidden):
    for name in ("finetuning_instruction", "pretraining_eval", "save_load_model"):
        package = importlib.import_module("gpt2." + name)
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)
assert not plt.get_fignums()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=60,
    )
    assert result.stdout == ""


@pytest.mark.parametrize(
    "value", [{}, [], [{}], [{"instruction": 1, "input": "", "output": ""}]]
)
def test_rejects_invalid_instruction_data(tmp_path, value):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_data(path)


def test_default_data_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(data.load_data()) == 1100
    assert not (tmp_path / "instruction-data.json").exists()
    with pytest.raises(FileNotFoundError):
        data.load_data(tmp_path / "missing.json")


def test_preview_accepts_a_single_entry(tmp_path, capsys):
    path = tmp_path / "single.json"
    path.write_text(
        '[{"instruction": "Say hello", "input": "", "output": "Hello"}]',
        encoding="utf-8",
    )
    data.preview_formatted_data(path)
    assert "### Response:\nHello" in capsys.readouterr().out


def test_collate_preserves_eos_and_masks_padding():
    batch = [[1, 2, 3], [4]]
    inputs, targets = preparation.custom_collate(batch, pad_token_id=31)
    assert inputs.tolist() == [[1, 2, 3], [4, 31, 31]]
    assert targets.tolist() == [[2, 3, 31], [31, -100, -100]]
    cropped_inputs, cropped_targets = preparation.custom_collate(
        batch, pad_token_id=31, allowed_max_length=2
    )
    assert torch.equal(cropped_inputs, inputs[:, :2])
    assert torch.equal(cropped_targets, targets[:, :2])
    assert batch == [[1, 2, 3], [4]]


@pytest.mark.parametrize("batch", [[], [[]]])
def test_collate_rejects_empty_sequences(batch):
    with pytest.raises(ValueError, match="non-empty"):
        preparation.custom_collate(batch)


def test_loaders_work_with_spawned_workers(dataset_path):
    parts = preparation.create_train_val_test_dataloaders(
        data.load_data(dataset_path),
        2,
        TinyTokenizer(),
        "cpu",
        allowed_max_length=8,
        num_workers=1,
    )
    assert [len(part) for part in parts[:3]] == [8, 1, 1]
    train_loader = parts[3]
    pickle.dumps(train_loader.collate_fn)
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == targets.shape == (2, 4)
    assert inputs.device.type == "cpu"
    assert torch.all(targets[:, -1] == 31)


def test_loaders_reject_empty_splits_and_dropped_training_batch(dataset_path):
    entries = data.load_data(dataset_path)
    with pytest.raises(ValueError, match="at least 10"):
        preparation.create_train_val_test_dataloaders(
            entries[:2], 1, TinyTokenizer(), "cpu"
        )
    with pytest.raises(ValueError, match="reduce batch_size"):
        preparation.create_train_val_test_dataloaders(
            entries, 32, TinyTokenizer(), "cpu"
        )


def test_pretraining_windows_have_shifted_targets(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(10))
    monkeypatch.setattr(
        pretraining_demo, "GPT2Tokenizer",
        Mock(side_effect=AssertionError("Use the supplied tokenizer")),
    )
    loader = training.create_dataloader_v1(
        "example", batch_size=3, context_size=3, stride=3, shuffle=False,
        tokenizer=tokenizer,
    )
    inputs, targets = next(iter(loader))
    assert inputs.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert targets.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tokenizer.encode.assert_called_once_with("example")


def test_pretraining_splits_reuse_supplied_tokenizer(monkeypatch, tmp_path):
    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(10))
    monkeypatch.setattr(
        pretraining_demo, "GPT2Tokenizer",
        Mock(side_effect=AssertionError("Use the supplied tokenizer")),
    )
    data_path = tmp_path / "custom-corpus.txt"
    text = "Custom training text. " * 10
    data_path.write_text(text, encoding="utf-8")
    train_loader, val_loader, returned = training.create_train_val_dataloaders(
        3, 2, tokenizer=tokenizer, data_path=data_path
    )
    assert returned is tokenizer
    assert tokenizer.encode.call_count == 3
    full_text, train_text, val_text = [
        call.args[0] for call in tokenizer.encode.call_args_list
    ]
    assert full_text == text
    assert train_text + val_text == full_text
    assert len(train_text) == int(0.9 * len(full_text))
    assert len(train_loader) == 1
    assert len(val_loader) == 2
    for loader in (train_loader, val_loader):
        inputs, targets = next(iter(loader))
        assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_pretraining_rejects_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        training.create_train_val_dataloaders(
            3, 2, tokenizer=TinyTokenizer(), data_path=tmp_path / "missing.txt"
        )


def test_gpt2_default_corpus_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert pretraining_demo.DEFAULT_DATA_PATH.is_absolute()
    assert pretraining_demo.DEFAULT_DATA_PATH.is_file()
    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(10))
    train_loader, val_loader, returned = training.create_train_val_dataloaders(
        3, 2, tokenizer=tokenizer, data_path=str(pretraining_demo.DEFAULT_DATA_PATH)
    )
    assert returned is tokenizer
    assert len(train_loader) == 1
    assert len(val_loader) == 2


def test_pretraining_orchestrator_passes_tokenizer_through_training(
    tiny_model, monkeypatch, tmp_path
):
    tokenizer = TinyTokenizer()
    loader = [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))]
    loaders = Mock(return_value=(loader, loader, tokenizer))
    monkeypatch.setattr(training, "create_train_val_dataloaders", loaders)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(
        pretraining_demo, "GPT2Tokenizer",
        Mock(side_effect=AssertionError("Use the supplied tokenizer")),
    )
    before = tiny_model.token_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=0.01)
    sample = Mock(wraps=training.generate_and_print_sample)
    monkeypatch.setattr(training, "generate_and_print_sample", sample)
    data_path = tmp_path / "corpus.txt"
    train_losses, val_losses, tokens_seen, returned = training.train_model(
        tiny_model, 8, optimizer, 1, "prompt", tokenizer=tokenizer, data_path=data_path
    )
    loaders.assert_called_once_with(
        context_length=8, batch_size=2, tokenizer=tokenizer, data_path=data_path
    )
    assert returned is tokenizer
    assert len(train_losses) == len(val_losses) == 1
    assert tokens_seen == [2]
    assert not torch.equal(before, tiny_model.token_embedding.weight)
    sample.assert_called_once_with(
        tiny_model, tokenizer, "prompt", torch.device("cpu"), context_size=8
    )


def test_validation_concept_demo_reports_losses_without_training(
    tiny_model, monkeypatch, capsys
):
    loader = [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))]
    tokenizer = TinyTokenizer()
    tokenizer_factory = Mock(return_value=tokenizer)
    loaders = Mock(return_value=(loader, loader, tokenizer))
    monkeypatch.setattr(pretraining_demo, "GPT2Tokenizer", tokenizer_factory)
    monkeypatch.setattr(pretraining_demo, "create_train_val_dataloaders", loaders)
    monkeypatch.setattr(pretraining_demo, "GPT2Model", Mock(return_value=tiny_model))
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    before = {
        name: value.clone() for name, value in tiny_model.state_dict().items()
    }

    pretraining_demo.demonstrate_training_validation_loss()

    tokenizer_factory.assert_called_once_with()
    loaders.assert_called_once_with(
        256, 2, tokenizer=tokenizer, data_path=pretraining_demo.DEFAULT_DATA_PATH
    )
    output = capsys.readouterr().out
    assert "Train Loader:" in output
    assert "Validation Loader:" in output
    assert "Training Loss:" in output
    assert "Validation Loss:" in output
    for name, value in tiny_model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in tiny_model.parameters())


@pytest.mark.parametrize("is_training", [True, False])
def test_evaluation_restores_model_mode(tiny_model, is_training):
    tiny_model.train(is_training)
    loader = [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))]
    train_loss, val_loss = training.evaluate_model(tiny_model, loader, loader, "cpu", 1)
    assert train_loss == pytest.approx(val_loss)
    assert tiny_model.training == is_training
    with pytest.raises(ValueError, match="empty"):
        training.evaluate_model(tiny_model, [], loader, "cpu", 1)
    assert tiny_model.training == is_training


@pytest.mark.parametrize("is_training", [True, False])
def test_sample_generation_uses_requested_device_and_context(
    tiny_model, monkeypatch, is_training
):
    tiny_model.train(is_training)

    def generate(model, encoded, **kwargs):
        assert encoded.device.type == "meta"
        assert kwargs["context_size"] == 3
        assert kwargs["max_new_tokens"] == 50
        assert not model.training
        assert not torch.is_grad_enabled()
        return torch.tensor([[1, 2]])

    monkeypatch.setattr(training, "generate_text_from_inputsample", generate)
    training.generate_and_print_sample(
        tiny_model, TinyTokenizer(), "prompt", "meta", context_size=3
    )
    assert tiny_model.training == is_training


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("failure_stage", ["encoding", "generation"])
def test_sample_generation_restores_mode_on_failure(
    tiny_model, monkeypatch, is_training, failure_stage
):
    tiny_model.train(is_training)
    tokenizer = TinyTokenizer()
    failure = Mock(side_effect=RuntimeError("sample failed"))
    if failure_stage == "encoding":
        monkeypatch.setattr(tokenizer, "encode", failure)
    else:
        monkeypatch.setattr(training, "generate_text_from_inputsample", failure)
    with pytest.raises(RuntimeError, match="sample failed"):
        training.generate_and_print_sample(
            tiny_model, tokenizer, "prompt", "cpu", context_size=3
        )
    assert tiny_model.training == is_training


@pytest.mark.parametrize(
    "trainer", [training.train_model_simple, training.train_model_enhanced]
)
@pytest.mark.parametrize("context_size", [0, -1])
def test_training_rejects_invalid_sample_context_before_updates(
    tiny_model, trainer, context_size
):
    optimizer = Mock()
    loader = [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))]
    with pytest.raises(ValueError, match="context_size must be positive"):
        trainer(
            tiny_model, loader, loader, optimizer, "cpu", 1, 1, 1,
            "prompt", TinyTokenizer(), context_size=context_size,
        )
    optimizer.step.assert_not_called()


def test_post_training_demo_disables_dropout_for_both_generators(
    tiny_model, monkeypatch, tmp_path
):
    tiny_model.train()
    monkeypatch.setattr(pretraining_demo, "GPT2Model", Mock(return_value=tiny_model))
    tokenizer = TinyTokenizer()
    encode = Mock(wraps=tokenizer.encode)
    monkeypatch.setattr(tokenizer, "encode", encode)
    tokenizer_factory = Mock(return_value=tokenizer)
    train = Mock(return_value=([1.0], [1.0], [8], tokenizer))
    monkeypatch.setattr(pretraining_demo, "GPT2Tokenizer", tokenizer_factory)
    monkeypatch.setattr(pretraining_demo, "train_model", train)
    monkeypatch.setattr(pretraining_demo, "plot_losses", Mock())

    def generate(model, encoded, **kwargs):
        assert model is tiny_model
        assert all(not module.training for module in model.modules())
        return encoded

    greedy = Mock(side_effect=generate)
    sampling = Mock(side_effect=generate)
    monkeypatch.setattr(pretraining_demo, "generate_text_from_inputsample", greedy)
    monkeypatch.setattr(
        pretraining_demo, "generate_text_from_inputsample_with_sampling", sampling
    )

    pretraining_demo.run_pretraining_eval_gpt2()

    tokenizer_factory.assert_called_once_with()
    assert train.call_args.args[4] == "Every effort moves you"
    assert train.call_args.kwargs == {
        "tokenizer": tokenizer,
        "data_path": pretraining_demo.DEFAULT_DATA_PATH,
    }
    assert [call.args for call in encode.call_args_list] == [
        ("Every effort moves you",),
        ("Every effort moves you",),
    ]
    for generator in (greedy, sampling):
        generator.assert_called_once()
        assert generator.call_args.kwargs["max_new_tokens"] == 15

    train.reset_mock()
    encode.reset_mock()
    greedy.reset_mock()
    sampling.reset_mock()
    data_path = str(tmp_path / "custom-corpus.txt")
    pretraining_demo.run_pretraining_eval_gpt2(
        data_path=data_path,
        max_new_tokens=3,
        prompt="A custom training prompt",
    )

    assert train.call_args.args[4] == "A custom training prompt"
    assert train.call_args.kwargs == {
        "tokenizer": tokenizer,
        "data_path": data_path,
    }
    assert [call.args for call in encode.call_args_list] == [
        ("A custom training prompt",),
        ("A custom training prompt",),
    ]
    for generator in (greedy, sampling):
        generator.assert_called_once()
        assert generator.call_args.kwargs["max_new_tokens"] == 3


def test_instruction_workflow_compares_responses_and_saves_outputs(
    tiny_model, dataset_path, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "outputs"
    initial_weights = {
        name: value.clone() for name, value in tiny_model.state_dict().items()
    }
    get_model = Mock(return_value=tiny_model)
    generate = Mock(wraps=finetuning._generate_instruction_response)
    prepare = Mock(wraps=finetuning.create_train_val_test_dataloaders)
    train = Mock(wraps=finetuning.finetune_model_for_instruction_following)
    workflow = Mock()
    for name, spy in (
        ("get_pretrained_model", get_model),
        ("_generate_instruction_response", generate),
        ("create_train_val_test_dataloaders", prepare),
        ("finetune_model_for_instruction_following", train),
    ):
        monkeypatch.setattr(finetuning, name, spy)
        workflow.attach_mock(spy, name)
    monkeypatch.setattr(finetuning, "DEFAULT_MODELS_DIR", tmp_path / "cache")
    monkeypatch.setattr(finetuning, "DEFAULT_OUTPUT_DIR", output)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(
        finetuning, "GPT2Tokenizer", Mock(return_value=TinyTokenizer())
    )
    monkeypatch.setattr(
        training.plt, "show", Mock(side_effect=AssertionError("Interactive plot"))
    )
    sample = Mock(wraps=training.generate_and_print_sample)
    monkeypatch.setattr(training, "generate_and_print_sample", sample)
    sampling = Mock(wraps=finetuning.generate_text_from_inputsample_with_sampling)
    monkeypatch.setattr(
        finetuning, "generate_text_from_inputsample_with_sampling", sampling
    )
    checkpoint_path = finetuning.run_instruction_finetuning(
        prompt="Say hello",
        max_new_tokens=2,
        data_path=dataset_path,
        model_size="124M",
    )
    assert [call[0] for call in workflow.mock_calls] == [
        "get_pretrained_model",
        "_generate_instruction_response",
        "create_train_val_test_dataloaders",
        "finetune_model_for_instruction_following",
        "_generate_instruction_response",
        "_generate_instruction_response",
    ]
    expected_prompt = data.format_input_to_alpaca_prompt(
        data.InstructionEntry(instruction="Say hello", input="", output="")
    )
    assert generate.call_args_list[0].args[1] == expected_prompt
    assert generate.call_args_list[0].args == generate.call_args_list[1].args
    assert generate.call_args_list[0].args[-1] == 2
    assert [call.kwargs["max_new_tokens"] for call in sampling.call_args_list] == [
        2, 2, 2,
    ]
    get_model.assert_called_once_with("124M", models_dir=tmp_path / "cache")
    assert prepare.call_args.kwargs["batch_size"] == 8
    assert prepare.call_args.kwargs["allowed_max_length"] == 1024
    assert sample.call_count == 2
    assert sample.call_args.kwargs["context_size"] == 8
    assert checkpoint_path == output / "gpt2-124M-sft.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert set(checkpoint) == set(initial_weights)
    assert all(torch.isfinite(value).all() for value in checkpoint.values())
    assert any(
        not torch.equal(value, initial_weights[name])
        for name, value in checkpoint.items()
    )
    tiny_model.load_state_dict(checkpoint, strict=True)
    responses = json.loads(
        (output / "instruction-data-with-response.json").read_text(encoding="utf-8")
    )
    assert len(responses) == 1
    assert set(responses[0]) == {"instruction", "input", "output", "model_response"}
    assert responses[0]["instruction"] == "Instruction 8"
    assert isinstance(responses[0]["model_response"], str)
    assert (output / "losses.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not training.plt.get_fignums()
    console_output = capsys.readouterr().out
    assert (
        console_output.index("Before instruction fine-tuning:")
        < console_output.index("Fine-tuning completed")
        < console_output.index("After instruction fine-tuning:")
    )


def test_response_extraction_handles_eos_without_mutating_data(
    tiny_model, tmp_path, monkeypatch
):
    entries = [data.InstructionEntry(instruction="Say hi", input="", output="Hi")]

    def generate(model, idx, context_size, eos_id, **kwargs):
        assert context_size == 8
        assert eos_id == 31
        assert not model.training
        return idx

    monkeypatch.setattr(
        finetuning, "generate_text_from_inputsample_with_sampling", generate
    )
    path = tmp_path / "responses.json"
    finetuning.save_test_responses(tiny_model, entries, TinyTokenizer(), "cpu", 8, path)
    assert json.loads(path.read_text())[0]["model_response"] == ""
    assert "model_response" not in entries[0]


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens", "message"),
    [
        ("", 1, "prompt"),
        (" \n", 1, "prompt"),
        ([], 1, "prompt"),
        ("Say hello", 0, "max_new_tokens"),
        ("Say hello", -1, "max_new_tokens"),
        ("Say hello", 1.5, "max_new_tokens"),
        ("Say hello", True, "max_new_tokens"),
    ],
)
def test_instruction_runner_rejects_invalid_inputs_before_loading_model(
    monkeypatch, prompt, max_new_tokens, message
):
    get_model = Mock()
    monkeypatch.setattr(finetuning, "get_pretrained_model", get_model)
    with pytest.raises(ValueError, match=message):
        finetuning.run_instruction_finetuning(
            prompt=prompt, max_new_tokens=max_new_tokens
        )
    get_model.assert_not_called()


@pytest.mark.parametrize(
    ("generated_text", "expected"),
    [
        ("\n\n### Response:\nHello", "Hello"),
        (" Hello ", "Hello"),
        (
            "Hello ### Response: stays in the answer",
            "Hello ### Response: stays in the answer",
        ),
    ],
)
def test_response_extraction_strips_only_the_leading_header(
    tiny_model, tmp_path, monkeypatch, generated_text, expected
):
    tokenizer = Mock(eot_token=31)
    tokenizer.encode.return_value = [1, 2]
    tokenizer.decode.return_value = generated_text
    monkeypatch.setattr(
        finetuning,
        "generate_text_from_inputsample_with_sampling",
        Mock(return_value=torch.tensor([[1, 2, 3]])),
    )
    path = tmp_path / "responses.json"
    finetuning.save_test_responses(
        tiny_model,
        [data.InstructionEntry(instruction="Say hello", input="", output="Hello")],
        tokenizer,
        "cpu",
        8,
        path,
    )
    tokenizer.decode.assert_called_once_with([3])
    assert json.loads(path.read_text())[0]["model_response"] == expected


def test_instruction_runner_has_four_defaulted_parameters():
    parameters = inspect.signature(finetuning.run_instruction_finetuning).parameters
    assert list(parameters) == ["prompt", "max_new_tokens", "data_path", "model_size"]
    assert parameters["prompt"].default == "Every effort moves you"
    assert parameters["max_new_tokens"].default == 256
    assert parameters["data_path"].default == data.DEFAULT_DATA_PATH
    assert parameters["model_size"].default == "355M"
