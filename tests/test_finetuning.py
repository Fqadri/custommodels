import json
import os
import pickle
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from common import pretraining as training
from gpt2.finetuning_instruction import _1_rawdataandpromptformatconcept as data
from gpt2.finetuning_instruction import _2_preparedatasetandloadmodel as preparation
from gpt2.finetuning_instruction import _3_instructionfinetuningandeval as finetuning
from gpt2.finetuning_instruction.main import main
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
    tiny_model, monkeypatch
):
    tiny_model.train()
    monkeypatch.setattr(pretraining_demo, "GPT2Model", Mock(return_value=tiny_model))
    tokenizer = TinyTokenizer()
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

    pretraining_demo.main()

    tokenizer_factory.assert_called_once_with()
    assert train.call_args.kwargs["tokenizer"] is tokenizer
    assert train.call_args.kwargs["data_path"] == pretraining_demo.DEFAULT_DATA_PATH
    greedy.assert_called_once()
    sampling.assert_called_once()


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
        finetuning, "GPT2Tokenizer", Mock(return_value=TinyTokenizer())
    )
    monkeypatch.setattr(
        training.plt, "show", Mock(side_effect=AssertionError("Interactive plot"))
    )
    sample = Mock(wraps=training.generate_and_print_sample)
    monkeypatch.setattr(training, "generate_and_print_sample", sample)
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
            "4",
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
    assert sample.call_count == 1
    assert sample.call_args.kwargs["context_size"] == 8
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
