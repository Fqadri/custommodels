import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from common.generation import generate_text_from_inputsample_with_sampling
from gpt2.model import gpt2_tokenizer
from gpt2.model import main as demo
from gpt2.model._11_transformerblock import TransformerBlock
from gpt2.model._12_gpt2model import GPT2Model


def test_import_does_not_run_demo() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import gpt2.model.main"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float64])
def test_main_generates_reproducible_examples(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dtype: torch.dtype | None,
) -> None:
    monkeypatch.setattr(
        demo,
        "GPT2_CONFIG_124M",
        {
            **demo.GPT2_CONFIG_124M,
            "vocab_size": 32,
            "dim_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "context_length": 8,
        },
    )
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4]
    tokenizer.decode.side_effect = lambda tokens: " ".join(map(str, tokens))
    get_encoding = Mock(return_value=tokenizer)
    monkeypatch.setattr(gpt2_tokenizer.tiktoken, "get_encoding", get_encoding)
    model_factory = Mock(wraps=demo.GPT2Model)
    monkeypatch.setattr(demo, "GPT2Model", model_factory)

    main_kwargs = {} if dtype is None else {"dtype": dtype}
    demo.main(**main_kwargs)
    output = capsys.readouterr().out

    model_factory.assert_called_once_with(
        demo.GPT2_CONFIG_124M,
        debug=False,
        dtype=torch.float32 if dtype is None else dtype,
    )
    get_encoding.assert_called_once_with("gpt2")
    assert tokenizer.encode.call_count == 3
    for call in tokenizer.encode.call_args_list:
        assert call.args == ("Every effort moves you",)
        assert call.kwargs == {"allowed_special": {"<|endoftext|>"}}

    sequences = [call.args[0] for call in tokenizer.decode.call_args_list]
    assert [len(tokens) for tokens in sequences] == [14, 14, 19]
    assert sequences[0] == sequences[1]
    for tokens in sequences:
        assert tokens[:4] == [1, 2, 3, 4]
        assert all(0 <= token < 32 for token in tokens)

    assert output.count("Generated text:\n") == 2
    assert output.count("Generated text with sampling:\n") == 1

    demo.main(**main_kwargs)
    assert capsys.readouterr().out == output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_model_and_block_preserve_dtype_through_backward(dtype: torch.dtype) -> None:
    torch.manual_seed(123)
    config = {
        "vocab_size": 32,
        "dim_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "context_length": 8,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    for module, inputs, expected_shape in (
        (
            GPT2Model(config, dtype=dtype),
            torch.randint(0, 32, (2, 4)),
            (2, 4, 32),
        ),
        (
            TransformerBlock(config, dtype=dtype),
            torch.randn(2, 4, 8, dtype=dtype),
            (2, 4, 8),
        ),
    ):
        assert all(parameter.dtype == dtype for parameter in module.parameters())
        output = module(inputs)
        assert output.shape == expected_shape
        assert output.dtype == dtype
        assert torch.isfinite(output).all()

        output.square().mean().backward()
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert parameter.grad.dtype == dtype
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize(
    ("steps", "eos_id", "max_new_tokens", "expected", "expected_calls"),
    [
        pytest.param(
            [[9, 3], [4, 5], [6, 9], [7, 8]],
            9, 4, [[1, 9, 9], [2, 3, 5]], 3,
            id="staggered-eos",
        ),
        pytest.param(
            [[3, 9], [5, 4], [9, 6]],
            9, 3, [[1, 3, 5], [2, 9, 9]], 3,
            id="reversed-finish-order",
        ),
        pytest.param(
            [[9, 9]], 9, 3, [[1], [2]], 1, id="simultaneous-eos",
        ),
        pytest.param(
            [[3], [9], [4]], 9, 3, [[1, 3]], 2, id="single-prompt",
        ),
        pytest.param(
            [[3, 4], [5, 6]],
            9, 2, [[1, 3, 5], [2, 4, 6]], 2,
            id="token-limit-without-eos",
        ),
        pytest.param(
            [[9, 3], [4, 5]],
            9, 2, [[1, 9, 9], [2, 3, 5]], 2,
            id="token-limit-with-unfinished-prompt",
        ),
        pytest.param(
            [[0, 3], [4, 0]], 0, 3, [[1, 0], [2, 3]], 2, id="zero-eos-id",
        ),
        pytest.param(
            [[9, 3], [4, 9]],
            None, 2, [[1, 9, 4], [2, 3, 9]], 2,
            id="eos-disabled",
        ),
        pytest.param(
            [], 9, 0, [[1], [2]], 0, id="zero-token-budget",
        ),
    ],
)
def test_sampling_tracks_eos_per_prompt(
    steps: list[list[int]],
    eos_id: int | None,
    max_new_tokens: int,
    expected: list[list[int]],
    expected_calls: int,
    temperature: float,
) -> None:
    batch_size = len(expected)
    model = Mock(
        side_effect=[
            torch.full((batch_size, 1, 10), -10.0).scatter_(
                2, torch.tensor(step).reshape(batch_size, 1, 1), 10.0
            )
            for step in steps
        ]
    )
    inputs = torch.arange(1, batch_size + 1).unsqueeze(1)
    result = generate_text_from_inputsample_with_sampling(
        model,
        inputs,
        max_new_tokens=max_new_tokens,
        context_size=8,
        temperature=temperature,
        top_k=1,
        eos_id=eos_id,
    )
    assert result.tolist() == expected
    assert result.dtype == inputs.dtype
    assert result.device == inputs.device
    assert model.call_count == expected_calls
