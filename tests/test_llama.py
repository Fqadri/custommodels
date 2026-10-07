import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import sentencepiece as spm
import torch
from torch import nn

from common import pretraining as training
from common.generation import (
    generate_text_from_inputsample,
    generate_text_from_inputsample_with_sampling,
)
from common.multi_head_attention import MultiHeadAttention
from common.tokenization import text_to_token_ids, token_ids_to_text
from common.transformer_block import TransformerBlock as SharedTransformerBlock
from llama2 import llamatokenizer
from llama2 import main as demo
from llama2.feedforward_swiglu import FeedForward
from llama2.llama2model import Llama2Model
from llama2.llamatokenizer import LlamaTokenizer
from llama2.pretraining_eval import _pretraining_eval as pretraining_demo
from llama2.rmsnormalization import RMSNorm
from llama2.rope import RotaryEmbedding
from llama2.transformer_block import TransformerBlock


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter([
            "Every effort moves you forward.",
            "Hello world",
            "Tiny models can generate text.",
            "This tokenizer is only for testing.",
        ]),
        model_writer=model,
        model_type="bpe",
        vocab_size=64,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    path = tmp_path_factory.mktemp("llama-tokenizer") / "tokenizer.model"
    path.write_bytes(model.getvalue())
    return path


@pytest.fixture
def config(tokenizer_file):
    return {
        "vocab_size": LlamaTokenizer(tokenizer_file).vocab_size,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 32,
        "dtype": torch.float32,
    }


@pytest.fixture(autouse=True)
def use_one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("text", ["", "Hello world", "Every effort moves you"])
def test_tokenizer_adds_bos_without_eos_and_works_with_common_helpers(
    tokenizer_file, text
):
    tokenizer = LlamaTokenizer(tokenizer_file)
    raw = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    expected = [raw.bos_id(), *raw.encode(text, out_type=int)]
    tokens = text_to_token_ids(text, tokenizer)

    assert tokens.tolist() == [expected]
    assert tokens.dtype == torch.long
    assert tokenizer.bos_id == raw.bos_id()
    assert tokenizer.eos_id == raw.eos_id()
    assert tokenizer.eos_id not in expected
    assert tokenizer.vocab_size == raw.get_piece_size()
    assert token_ids_to_text(tokens, tokenizer) == text


def test_tokenizer_rejects_missing_special_tokens(monkeypatch):
    processor = Mock()
    processor.bos_id.return_value = -1
    processor.eos_id.return_value = 2
    monkeypatch.setattr(
        llamatokenizer.spm, "SentencePieceProcessor", Mock(return_value=processor)
    )
    with pytest.raises(ValueError, match="BOS and EOS"):
        LlamaTokenizer("tokenizer.model")


def test_pretraining_windows_accept_llama_tokenizer(tokenizer_file):
    tokenizer = LlamaTokenizer(tokenizer_file)
    text = "Every effort moves you forward. Hello world. " * 3
    expected = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    loader = training.create_dataloader_v1(
        text, batch_size=2, context_size=4, stride=4, shuffle=False,
        tokenizer=tokenizer,
    )
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape == (2, 4)
    assert inputs[0, 0].item() == tokenizer.bos_id
    torch.testing.assert_close(inputs, torch.stack([expected[:4], expected[4:8]]))
    torch.testing.assert_close(targets, torch.stack([expected[1:5], expected[5:9]]))


@pytest.mark.parametrize(
    "trainer", [training.train_model_simple, training.train_model_enhanced]
)
def test_llama_training_generates_epoch_end_samples(
    config, tokenizer_file, monkeypatch, capsys, trainer
):
    model = Llama2Model(config)
    tokenizer = LlamaTokenizer(tokenizer_file)
    loader = training.create_dataloader_v1(
        "Every effort moves you forward. Hello world.",
        batch_size=1, context_size=4, stride=4, shuffle=False, tokenizer=tokenizer,
    )
    batches = [next(iter(loader))]
    before = model.tok_emb.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generate = Mock(wraps=generate_text_from_inputsample)
    monkeypatch.setattr(training, "generate_text_from_inputsample", generate)

    results = trainer(
        model, batches, batches, optimizer, "cpu", 1, 1, 1,
        "Hello world", tokenizer, context_size=3,
    )

    assert len(results[0]) == len(results[1]) == 1
    assert results[2] == [4]
    assert not torch.equal(before, model.tok_emb.weight)
    assert generate.call_count == 1
    assert generate.call_args.kwargs["context_size"] == 3
    assert generate.call_args.kwargs["max_new_tokens"] == 50
    assert "Generated text:" in capsys.readouterr().out
    assert model.training


def test_shared_pretraining_trains_llama_from_supplied_corpus(
    config, tokenizer_file, tmp_path, monkeypatch, capsys
):
    model = Llama2Model(config)
    tokenizer = LlamaTokenizer(tokenizer_file)
    data_path = tmp_path / "llama-corpus.txt"
    data_path.write_text("Hello world. " * 20, encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.tok_emb.weight.detach().clone()

    train_losses, val_losses, tokens_seen, returned = training.train_model(
        model, 4, optimizer, 1, "Hello world",
        tokenizer=tokenizer, data_path=data_path,
    )

    assert returned is tokenizer
    assert len(train_losses) == len(val_losses) == len(tokens_seen) > 0
    assert torch.isfinite(torch.tensor(train_losses + val_losses)).all()
    assert not torch.equal(before, model.tok_emb.weight)
    assert "Generated text:" in capsys.readouterr().out
    assert model.training


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_model_wires_llama_components_and_backpropagates(config, dtype):
    model = Llama2Model({**config, "dtype": dtype})
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    assert isinstance(model.final_norm, RMSNorm)
    assert model.out_head.weight is not model.tok_emb.weight
    for block in model.trf_blocks:
        assert isinstance(block, SharedTransformerBlock)
        assert type(block).forward is SharedTransformerBlock.forward
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.attention.rope, RotaryEmbedding)
        assert isinstance(block.ff, FeedForward)
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)
        assert block.norm1 is not block.norm2
    assert model.trf_blocks[0].norm1.weight is not model.trf_blocks[1].norm1.weight
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None
        if isinstance(module, nn.Dropout):
            assert module.p == 0.0

    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 5]])
    logits = model(inputs)
    assert logits.shape == (2, 4, config["vocab_size"])
    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()
    loss = nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rmsnorm_avoids_fp16_square_overflow():
    norm = RMSNorm(8, dtype=torch.float16)
    x = torch.full((1, 1, 8), 1000.0, dtype=torch.float16, requires_grad=True)
    output = norm(x)
    torch.testing.assert_close(output, torch.ones_like(x))
    assert "weight" in dict(norm.named_parameters())
    assert norm.weight.dtype == torch.float16
    output.sum().backward()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_model_is_causal_and_rejects_excess_context(config):
    model = Llama2Model(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 6, 7]])
    torch.testing.assert_close(
        model(inputs)[:, :2], model(changed)[:, :2], rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="context length"):
        model(torch.ones(1, 9, dtype=torch.long))


def test_optimizer_updates_and_checkpoint_roundtrip(config, tmp_path):
    model = Llama2Model(config)
    before = model.tok_emb.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    loss = nn.functional.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    loss.backward()
    optimizer.step()
    assert not torch.equal(model.tok_emb.weight, before)

    checkpoint = tmp_path / "llama.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = Llama2Model(config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    assert not any(".rope." in key for key in model.state_dict())


def test_shared_generation_runs_with_real_tokenizer(config, tokenizer_file):
    model = Llama2Model(config).eval()
    tokenizer = LlamaTokenizer(tokenizer_file)
    inputs = text_to_token_ids("Hello world", tokenizer)
    first = generate_text_from_inputsample(model, inputs, 3, model.context_length)
    second = generate_text_from_inputsample(model, inputs, 3, model.context_length)
    assert torch.equal(first, second)
    sampled = generate_text_from_inputsample_with_sampling(
        model, inputs, 3, model.context_length, temperature=0.7, top_k=5
    )
    for result in (first, sampled):
        assert result.shape == (1, inputs.shape[1] + 3)
        assert torch.equal(result[:, :inputs.shape[1]], inputs)
        assert torch.all((result >= 0) & (result < tokenizer.vocab_size))
        assert isinstance(token_ids_to_text(result, tokenizer), str)


def test_demo_runs_reproducibly_with_real_tiny_model(
    config, tokenizer_file, monkeypatch, capsys
):
    monkeypatch.setattr(demo, "LLAMA2_CONFIG_7B", config)
    demo.main(["--tokenizer", str(tokenizer_file)])
    first = capsys.readouterr().out
    assert "Total number of parameters:" in first
    assert first.count("Generated text:\n") == 2
    assert first.count("Generated text with sampling:\n") == 1
    demo.main(["--tokenizer", str(tokenizer_file)])
    assert capsys.readouterr().out == first


def test_demo_rejects_missing_tokenizer_before_allocating_model(monkeypatch, tmp_path):
    model = Mock()
    monkeypatch.setattr(demo, "Llama2Model", model)
    with pytest.raises(SystemExit) as error:
        demo.main(["--tokenizer", str(tmp_path / "missing.model")])
    assert error.value.code == 2
    model.assert_not_called()


def test_demo_rejects_mismatched_vocabulary_before_allocating_model(
    config, tokenizer_file, monkeypatch
):
    monkeypatch.setattr(
        demo, "LLAMA2_CONFIG_7B", {**config, "vocab_size": config["vocab_size"] + 1}
    )
    model = Mock()
    monkeypatch.setattr(demo, "Llama2Model", model)
    with pytest.raises(SystemExit) as error:
        demo.main(["--tokenizer", str(tokenizer_file)])
    assert error.value.code == 2
    model.assert_not_called()


def test_7b_configuration_constructs_without_allocating_weights():
    with torch.device("meta"):
        model = Llama2Model(demo.LLAMA2_CONFIG_7B)
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_738_415_616
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert all(buffer.device.type == "meta" for buffer in model.buffers())


def test_block_rejects_incompatible_head_count(config):
    with pytest.raises(ValueError, match="divisible"):
        TransformerBlock({**config, "n_heads": 3})


def test_pretraining_corpus_is_an_identical_copy():
    source = (
        Path(__file__).resolve().parents[1]
        / "gpt2" / "pretraining_eval" / "the-verdict.txt"
    )
    copied = pretraining_demo.DEFAULT_DATA_PATH
    assert source.resolve() != copied.resolve()
    assert source.read_bytes() == copied.read_bytes()


def test_small_pretraining_configuration_matches_gpt2_demo_scale():
    cfg = pretraining_demo.LLAMA2_CONFIG_SMALL
    assert cfg["emb_dim"] == 768
    assert cfg["n_heads"] == cfg["n_layers"] == 12
    assert cfg["context_length"] == 256
    with torch.device("meta"):
        model = Llama2Model(cfg)
    assert sum(parameter.numel() for parameter in model.parameters()) == 134_105_856
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_pretraining_runner_trains_plots_and_generates(
    config, tokenizer_file, tmp_path, monkeypatch, capsys
):
    data_path = tmp_path / "pretraining.txt"
    data_path.write_text("Hello world. " * 40, encoding="utf-8")
    model = Llama2Model(config)
    before = model.tok_emb.weight.detach().clone()
    factory = Mock(return_value=model)
    train = Mock(wraps=training.train_model)
    greedy = Mock(wraps=generate_text_from_inputsample)
    sampled = Mock(wraps=generate_text_from_inputsample_with_sampling)
    show = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(pretraining_demo, "LLAMA2_CONFIG_SMALL", config)
    monkeypatch.setattr(pretraining_demo, "Llama2Model", factory)
    monkeypatch.setattr(pretraining_demo, "train_model", train)
    monkeypatch.setattr(pretraining_demo, "generate_text_from_inputsample", greedy)
    monkeypatch.setattr(
        pretraining_demo, "generate_text_from_inputsample_with_sampling", sampled
    )
    monkeypatch.setattr(training.plt, "show", show)

    pretraining_demo.main([
        "--tokenizer", str(tokenizer_file),
        "--data", str(data_path), "--epochs", "1",
    ])

    factory.assert_called_once_with(config)
    assert train.call_count == 1
    assert train.call_args.args[3] == 1
    assert train.call_args.kwargs["data_path"] == data_path
    assert isinstance(train.call_args.kwargs["tokenizer"], LlamaTokenizer)
    assert not torch.equal(before, model.tok_emb.weight)
    assert not model.training
    show.assert_called_once()
    assert not training.plt.get_fignums()
    greedy.assert_called_once()
    sampled.assert_called_once()
    assert greedy.call_args.args[0] is sampled.call_args.args[0] is model
    assert greedy.call_args.kwargs["max_new_tokens"] == 15
    assert sampled.call_args.kwargs["context_size"] == config["context_length"]
    assert sampled.call_args.kwargs["eos_id"] == LlamaTokenizer(tokenizer_file).eos_id
    output = capsys.readouterr().out
    assert "Generated text:\n" in output
    assert "Generated text with greedy decoding:" in output
    assert "Generated text with sampling:" in output


def test_pretraining_defaults_to_ten_epochs_and_its_local_corpus(
    config, tokenizer_file, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model = Llama2Model(config)
    tokenizer = LlamaTokenizer(tokenizer_file)
    train = Mock(return_value=([1.0], [1.0], [8], tokenizer))
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(pretraining_demo, "LLAMA2_CONFIG_SMALL", config)
    monkeypatch.setattr(pretraining_demo, "Llama2Model", Mock(return_value=model))
    monkeypatch.setattr(pretraining_demo, "train_model", train)
    monkeypatch.setattr(pretraining_demo, "plot_losses", Mock())

    def generate(model, encoded, **kwargs):
        assert not model.training
        return encoded

    monkeypatch.setattr(pretraining_demo, "generate_text_from_inputsample", generate)
    monkeypatch.setattr(
        pretraining_demo, "generate_text_from_inputsample_with_sampling", generate
    )
    pretraining_demo.main(["--tokenizer", str(tokenizer_file)])
    assert train.call_args.args[3] == 10
    assert train.call_args.kwargs["data_path"] == pretraining_demo.DEFAULT_DATA_PATH
    assert pretraining_demo.DEFAULT_DATA_PATH.is_absolute()


def test_pretraining_loss_demo_only_evaluates(
    config, tokenizer_file, tmp_path, monkeypatch, capsys
):
    data_path = tmp_path / "validation.txt"
    data_path.write_text("Hello world. " * 40, encoding="utf-8")
    model = Llama2Model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(pretraining_demo, "LLAMA2_CONFIG_SMALL", config)
    monkeypatch.setattr(pretraining_demo, "Llama2Model", Mock(return_value=model))

    pretraining_demo.demonstrate_training_validation_loss(tokenizer_file, data_path)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    output = capsys.readouterr().out
    assert "Training Loss:" in output
    assert "Validation Loss:" in output


@pytest.mark.parametrize(
    "failure", ["missing-tokenizer", "missing-data", "invalid-epochs", "vocabulary"]
)
def test_pretraining_rejects_invalid_inputs_before_allocating_model(
    config, tokenizer_file, tmp_path, monkeypatch, failure
):
    model = Mock()
    monkeypatch.setattr(pretraining_demo, "Llama2Model", model)
    cfg = dict(config)
    args = [
        "--tokenizer", str(tokenizer_file),
        "--data", str(pretraining_demo.DEFAULT_DATA_PATH),
    ]
    if failure == "missing-tokenizer":
        args[1] = str(tmp_path / "missing.model")
    elif failure == "missing-data":
        args[3] = str(tmp_path / "missing.txt")
    elif failure == "invalid-epochs":
        args.extend(["--epochs", "0"])
    else:
        cfg["vocab_size"] += 1
    monkeypatch.setattr(pretraining_demo, "LLAMA2_CONFIG_SMALL", cfg)
    with pytest.raises(SystemExit) as error:
        pretraining_demo.main(args)
    assert error.value.code == 2
    model.assert_not_called()


@pytest.mark.parametrize(
    "module_name", ["llama2.main", "llama2.pretraining_eval._pretraining_eval"]
)
def test_import_is_safe_and_cli_help_does_not_allocate_model(module_name):
    imported = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert imported.stdout == ""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert "--tokenizer" in result.stdout
    assert "Total number of parameters:" not in result.stdout
