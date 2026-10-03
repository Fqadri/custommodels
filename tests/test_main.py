import subprocess
import sys
from unittest.mock import Mock

import pytest

from gpt2 import main as demo


def test_import_does_not_run_demo() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import gpt2.main"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_main_generates_reproducible_examples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    monkeypatch.setattr(demo.tiktoken, "get_encoding", get_encoding)

    demo.main()
    output = capsys.readouterr().out

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

    demo.main()
    assert capsys.readouterr().out == output
