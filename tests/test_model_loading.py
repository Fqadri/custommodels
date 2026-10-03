from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import requests
import torch

from gpt2.model._12_gpt2model import GPT2Model
from gpt2.save_load_model import _3_loadopenaiweights as weights
from gpt2.save_load_model import gpt_download as downloads


def test_checkpoint_weights_map_to_custom_model():
    config = {
        "vocab_size": 8,
        "dim_model": 4,
        "num_layers": 1,
        "num_heads": 2,
        "context_length": 8,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model = GPT2Model(config)
    rng = np.random.default_rng(123)

    def array(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    params = {
        "wte": array(8, 4),
        "wpe": array(8, 4),
        "g": array(4),
        "b": array(4),
        "blocks": [
            {
                "attn": {
                    "c_attn": {"w": array(4, 12), "b": array(12)},
                    "c_proj": {"w": array(4, 4), "b": array(4)},
                },
                "mlp": {
                    "c_fc": {"w": array(4, 16), "b": array(16)},
                    "c_proj": {"w": array(16, 4), "b": array(4)},
                },
                "ln_1": {"g": array(4), "b": array(4)},
                "ln_2": {"g": array(4), "b": array(4)},
            }
        ],
    }
    weights.load_weights_into_gpt(model, params)
    block = model.transformer_blocks[0]
    checkpoint_block = params["blocks"][0]
    for index, layer in enumerate(
        (block.attention.W_query, block.attention.W_key, block.attention.W_value)
    ):
        np.testing.assert_array_equal(
            layer.weight.detach().numpy(),
            checkpoint_block["attn"]["c_attn"]["w"][:, index * 4 : (index + 1) * 4].T,
        )
        np.testing.assert_array_equal(
            layer.bias.detach().numpy(),
            checkpoint_block["attn"]["c_attn"]["b"][index * 4 : (index + 1) * 4],
        )
    for layer, expected in (
        (block.attention.out_projection, checkpoint_block["attn"]["c_proj"]),
        (block.ff.layers[0], checkpoint_block["mlp"]["c_fc"]),
        (block.ff.layers[2], checkpoint_block["mlp"]["c_proj"]),
    ):
        np.testing.assert_array_equal(layer.weight.detach().numpy(), expected["w"].T)
        np.testing.assert_array_equal(layer.bias.detach().numpy(), expected["b"])
    for layer, expected in (
        (block.norm1, checkpoint_block["ln_1"]),
        (block.norm2, checkpoint_block["ln_2"]),
        (model.final_norm, params),
    ):
        np.testing.assert_array_equal(layer.scale.detach().numpy(), expected["g"])
        np.testing.assert_array_equal(layer.shift.detach().numpy(), expected["b"])
    np.testing.assert_array_equal(
        model.token_embedding.weight.detach().numpy(), params["wte"]
    )
    np.testing.assert_array_equal(
        model.position_embedding.weight.detach().numpy(), params["wpe"]
    )
    assert model.output_layer.weight is model.token_embedding.weight
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]]))).all()


def test_assign_rejects_incompatible_parameters():
    with pytest.raises(ValueError, match="qkv_bias"):
        weights.assign(None, np.zeros(4))
    with pytest.raises(ValueError, match="Shape mismatch"):
        weights.assign(torch.zeros(4), np.zeros(5))
    parameter = torch.nn.Parameter(
        torch.zeros(4, dtype=torch.float64), requires_grad=False
    )
    assigned = weights.assign(parameter, np.ones(4, dtype=np.float32))
    assert assigned.dtype == torch.float64
    assert not assigned.requires_grad


def test_cached_weights_do_not_contact_the_network(tmp_path, monkeypatch):
    destination = tmp_path / "weights"
    destination.write_bytes(b"cached")
    get = Mock(side_effect=AssertionError("Unexpected network request"))
    monkeypatch.setattr(downloads.requests, "get", get)
    downloads.download_file("https://example.test/weights", destination)
    get.assert_not_called()


def test_download_urls_and_local_checkpoint_paths(tmp_path, monkeypatch):
    seen_urls = []

    def download(url, destination, backup_url):
        seen_urls.extend([url, backup_url])
        if destination.name == "hparams.json":
            destination.write_text('{"n_layer": 1}', encoding="utf-8")

    load = Mock(return_value={"blocks": []})
    monkeypatch.setattr(downloads, "download_file", download)
    monkeypatch.setattr(downloads, "load_gpt2_params_from_tf_ckpt", load)
    settings, params = downloads.download_and_load_gpt2("124M", tmp_path)
    assert settings == {"n_layer": 1}
    assert params == {"blocks": []}
    assert len(seen_urls) == 14
    assert all("\\" not in url and "/124M/" in url for url in seen_urls)
    load.assert_called_once_with(str(tmp_path / "124M" / "model.ckpt"), settings)


def response_with_chunks(chunks, size):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Length": str(size)}
    response.iter_content.return_value = chunks
    return response


def test_download_uses_backup_and_publishes_complete_file(
    tmp_path, monkeypatch, capsys
):
    response = response_with_chunks([b"abc"], 3)
    get = Mock(side_effect=[requests.ConnectionError("primary unavailable"), response])
    monkeypatch.setattr(downloads.requests, "get", get)
    destination = tmp_path / "weights"
    downloads.download_file(
        "https://primary.test/file", destination, "https://backup.test/file"
    )
    assert destination.read_bytes() == b"abc"
    assert get.call_args_list[1].args[0] == "https://backup.test/file"
    assert not list(tmp_path.glob("*.part"))
    assert "Primary download failed" in capsys.readouterr().out


def test_failed_download_raises_and_removes_partial_file(tmp_path, monkeypatch):
    response = response_with_chunks([b"abc"], 10)
    monkeypatch.setattr(downloads.requests, "get", Mock(return_value=response))
    destination = tmp_path / "weights"
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        downloads.download_file(
            "https://primary.test/file", destination, "https://backup.test/file"
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))
