import json
import os
import pickle
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from gpt2.finetuning_instruction import _1_rawdataandpromptformatconcept as data
from gpt2.finetuning_instruction import _2_preparedatasetandloadmodel as preparation
from gpt2.finetuning_instruction import _3_instructionfinetuningandeval as finetuning
from gpt2.finetuning_instruction.main import main
from gpt2.model._12_gpt2model import GPT2Model
from gpt2.pretraining_eval import _1_trainingvalidationlossconcept as losses
from gpt2.pretraining_eval import _2_trainingandevaluation as training


class TinyTokenizer:
    eot_token = 31

    def encode(self, text, **kwargs):
        return [1, 2, 3, 4]

    def decode(self, tokens):
        return " ".join(map(str, tokens))


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
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=60,
    )
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    "value", [{}, [], [{}], [{"instruction": 1, "input": "", "output": ""}]]
)
def test_rejects_invalid_instruction_data(tmp_path, value):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_instruction_data(path)


def test_default_data_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        Mock(side_effect=AssertionError("Unexpected download")),
    )
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
    main(["preview", "--data", str(path)])
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
    monkeypatch.setattr(losses.tiktoken, "get_encoding", Mock(return_value=tokenizer))
    loader = losses.create_dataloader_v1(
        "example", batch_size=3, context_size=3, stride=3, shuffle=False
    )
    inputs, targets = next(iter(loader))
    assert inputs.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert targets.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


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


def test_sample_generation_uses_requested_device(tiny_model, monkeypatch):
    tiny_model.eval()

    def generate(model, encoded, **kwargs):
        assert encoded.device.type == "meta"
        return torch.tensor([[1, 2]])

    monkeypatch.setattr(training, "generate_text_from_inputsample", generate)
    training.generate_and_print_sample(tiny_model, TinyTokenizer(), "prompt", "meta")
    assert not tiny_model.training


def test_train_cli_runs_real_optimization_and_saves_outputs(
    tiny_model, dataset_path, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "outputs"
    initial_weights = {
        name: value.clone() for name, value in tiny_model.state_dict().items()
    }
    get_model = Mock(return_value=tiny_model)
    monkeypatch.setattr(finetuning, "get_pretrained_model", get_model)
    monkeypatch.setattr(
        finetuning.tiktoken, "get_encoding", Mock(return_value=TinyTokenizer())
    )
    monkeypatch.setattr(
        training.plt, "show", Mock(side_effect=AssertionError("Interactive plot"))
    )
    main(
        [
            "train",
            "--data",
            str(dataset_path),
            "--model",
            "124M",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--context-length",
            "8",
            "--device",
            "cpu",
            "--max-new-tokens",
            "2",
            "--models-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(output),
        ]
    )
    get_model.assert_called_once_with("124M", models_dir=tmp_path / "cache")
    checkpoint = torch.load(output / "gpt2-124M-sft.pth", weights_only=True)
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
    assert "Fine-tuning completed" in capsys.readouterr().out


def test_response_extraction_handles_eos_without_mutating_data(
    tiny_model, tmp_path, monkeypatch
):
    entries = [data.InstructionEntry(instruction="Say hi", input="", output="Hi")]

    def generate(model, idx, context_size, eos_id, **kwargs):
        assert context_size == 8
        assert eos_id == 31
        return idx

    monkeypatch.setattr(
        finetuning, "generate_text_from_inputsample_with_sampling", generate
    )
    path = tmp_path / "responses.json"
    finetuning.save_test_responses(tiny_model, entries, TinyTokenizer(), "cpu", 8, path)
    assert json.loads(path.read_text())[0]["model_response"] == ""
    assert "model_response" not in entries[0]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["train", "--epochs", "0"],
        ["train", "--context-length", "1025"],
        ["prepare", "--batch-size", "-1"],
        ["prepare", "--num-workers", "-1"],
        ["train", "--model", "unknown"],
        ["score"],
    ],
)
def test_cli_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_help_does_not_import_training_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from gpt2.finetuning_instruction.main import main\n"
                "try:\n    main(['train', '--help'])\n"
                "except SystemExit as error:\n    assert error.code == 0\n"
                "assert 'torch' not in sys.modules\n"
                "assert 'tensorflow' not in sys.modules\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--model" in result.stdout


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


def test_numbered_training_module_forwards_help():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gpt2.finetuning_instruction._3_instructionfinetuningandeval",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "--epochs" in result.stdout
    assert "Starting instruction fine-tuning" not in result.stdout
