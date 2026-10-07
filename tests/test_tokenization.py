from unittest.mock import Mock

import pytest
import torch

from common.tokenization import text_to_token_ids, token_ids_to_text
from gpt2.model import gpt2_tokenizer
from gpt2.model.gpt2_tokenizer import GPT2Tokenizer


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("Hello world", [15496, 995]),
        ("<|endoftext|>", [50256]),
        ("Hello<|endoftext|>", [15496, 50256]),
    ],
)
def test_gpt2_wrapper_preserves_special_tokens_and_eot_id(
    monkeypatch, text, ids
):
    encoding = Mock(eot_token=50256)
    encoding.encode.return_value = ids
    encoding.decode.return_value = text
    get_encoding = Mock(return_value=encoding)
    monkeypatch.setattr(gpt2_tokenizer.tiktoken, "get_encoding", get_encoding)

    tokenizer = GPT2Tokenizer()
    tokens = text_to_token_ids(text, tokenizer)

    get_encoding.assert_called_once_with("gpt2")
    encoding.encode.assert_called_once_with(
        text, allowed_special={"<|endoftext|>"}
    )
    assert tokens.tolist() == [ids]
    assert tokens.dtype == torch.long
    assert token_ids_to_text(tokens, tokenizer) == text
    encoding.decode.assert_called_once_with(ids)
    assert tokenizer.eot_token == 50256


def test_common_helpers_accept_tokenizer_without_gpt2_options():
    class PlainTokenizer:
        def encode(self, text):
            assert text == "Hello"
            return [1, 2]

        def decode(self, ids):
            assert ids == [1, 2]
            return "Hello"

    tokenizer = PlainTokenizer()
    tokens = text_to_token_ids("Hello", tokenizer)
    assert tokens.shape == (1, 2)
    assert tokens.dtype == torch.long
    assert token_ids_to_text(tokens, tokenizer) == "Hello"
