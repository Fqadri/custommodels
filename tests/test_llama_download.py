from unittest.mock import Mock

import pytest

from llama2 import download_load_weights as downloader


@pytest.mark.parametrize("instructionfinetuned", [False, True])
def test_download_uses_configured_hf_cache_without_loading_model(
    tmp_path, monkeypatch, capsys, instructionfinetuned
):
    cache_dir = tmp_path / "HF_Cache" / "hub"
    model_name = "Llama-2-7b-chat" if instructionfinetuned else "Llama-2-7b"
    snapshot = cache_dir / f"models--meta-llama--{model_name}" / "snapshots" / "revision"
    download = Mock(return_value=str(snapshot))
    model = Mock(side_effect=AssertionError("Downloading must not allocate a model"))
    tokenizer = Mock(side_effect=AssertionError("Downloading must not load a tokenizer"))
    monkeypatch.setattr(downloader.constants, "HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setattr(downloader, "snapshot_download", download)
    monkeypatch.setattr(downloader, "Llama2Model", model)
    monkeypatch.setattr(downloader, "LlamaTokenizer", tokenizer)

    if instructionfinetuned:
        tokenizer_path, checkpoint_path = downloader.download_llama2(
            instructionfinetuned=True
        )
    else:
        tokenizer_path, checkpoint_path = downloader.download_llama2()
    assert tokenizer_path == snapshot / "tokenizer.model"
    assert checkpoint_path == snapshot / "consolidated.00.pth"

    download.assert_called_once_with(
        repo_id=f"meta-llama/{model_name}",
        allow_patterns=["consolidated.00.pth", "params.json", "tokenizer.model"],
        cache_dir=cache_dir,
        token=True,
    )
    model.assert_not_called()
    tokenizer.assert_not_called()
    output = capsys.readouterr().out
    assert str(cache_dir) in output
    assert str(snapshot) in output


def test_download_failure_propagates_without_reporting_success(monkeypatch, capsys):
    monkeypatch.setattr(
        downloader, "snapshot_download", Mock(side_effect=RuntimeError("Access denied"))
    )

    with pytest.raises(RuntimeError, match="Access denied"):
        downloader.download_llama2()

    assert "Downloaded LLaMA 2 files:" not in capsys.readouterr().out
