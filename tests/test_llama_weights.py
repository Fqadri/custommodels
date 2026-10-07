import pytest
import torch

from llama2.download_load_weights import assign, load_weights_into_llama
from llama2.llama2model import Llama2Model


@pytest.fixture
def config():
    return {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": 2,
        "hidden_dim": 16,
        "dtype": torch.float32,
    }


@pytest.mark.parametrize("numpy_source", [False, True])
def test_assign_copies_in_place_and_returns_none(numpy_source):
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    storage_pointer = parameter.data_ptr()
    source = torch.arange(4, dtype=torch.float32)

    result = assign(parameter, source.numpy() if numpy_source else source)

    assert result is None
    assert parameter.data_ptr() == storage_pointer
    assert parameter.dtype == torch.float64
    assert parameter.requires_grad
    assert parameter.grad is None
    torch.testing.assert_close(parameter, source.double())


def test_assign_rejects_shape_mismatch_without_mutating_destination():
    parameter = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="Shape mismatch in tensor 'query'"):
        assign(parameter, torch.ones(5), tensor_name="query")
    torch.testing.assert_close(parameter, torch.zeros(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loading_keeps_all_parameters_registered_and_copies_weights(
    tmp_path, config, dtype
):
    cfg = {**config, "dtype": dtype}
    model = Llama2Model(cfg)
    parameters_before = dict(model.named_parameters())
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    mapping = {
        "tok_embeddings.weight": "tok_emb.weight",
        "norm.weight": "final_norm.weight",
        "output.weight": "out_head.weight",
    }
    for index in range(cfg["n_layers"]):
        for original, destination in (
            ("attention.wq", "attention.W_query"),
            ("attention.wk", "attention.W_key"),
            ("attention.wv", "attention.W_value"),
            ("attention.wo", "attention.out_projection"),
            ("attention_norm", "norm1"),
            ("feed_forward.w1", "ff.fc1"),
            ("feed_forward.w3", "ff.fc2"),
            ("feed_forward.w2", "ff.fc3"),
            ("ffn_norm", "norm2"),
        ):
            mapping[f"layers.{index}.{original}.weight"] = (
                f"trf_blocks.{index}.{destination}.weight"
            )

    assert set(mapping.values()) == set(parameters_before)
    weights = {
        original: torch.full(parameters_before[destination].shape, (index + 1) / 100)
        for index, (original, destination) in enumerate(mapping.items())
    }
    checkpoint = tmp_path / "consolidated.00.pth"
    torch.save(weights, checkpoint)

    load_weights_into_llama(model, cfg, checkpoint)

    parameters_after = dict(model.named_parameters())
    assert set(parameters_after) == set(parameters_before)
    for original, destination in mapping.items():
        assert parameters_after[destination] is parameters_before[destination]
        torch.testing.assert_close(
            parameters_after[destination], weights[original].to(dtype), rtol=0, atol=0
        )
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffers_before[name], rtol=0, atol=0)
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("configured_layers", [0, 1, 3])
def test_layer_count_mismatch_fails_before_reading_or_mutating(
    tmp_path, config, configured_layers
):
    model = Llama2Model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    with pytest.raises(ValueError, match="Layer count mismatch"):
        load_weights_into_llama(
            model, {**config, "n_layers": configured_layers}, tmp_path / "missing.pth"
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "checkpoint_indices",
    [
        pytest.param([], id="no-layers"),
        pytest.param([0], id="missing-layer"),
        pytest.param([0, 1, 2], id="extra-layer"),
        pytest.param([0, 2], id="skipped-index"),
        pytest.param([1, 2], id="shifted-indices"),
        pytest.param([-1, 0], id="negative-index"),
    ],
)
def test_checkpoint_layer_mismatch_fails_before_mutating(
    tmp_path, config, checkpoint_indices
):
    model = Llama2Model(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    weights = {"tok_embeddings.weight": torch.ones_like(model.tok_emb.weight)}
    for index in checkpoint_indices:
        weights[f"layers.{index}.attention.wq.weight"] = torch.ones(
            config["emb_dim"], config["emb_dim"]
        )
    checkpoint = tmp_path / "invalid-layers.pth"
    torch.save(weights, checkpoint)

    with pytest.raises(ValueError, match="Checkpoint layers do not match"):
        load_weights_into_llama(model, config, checkpoint)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
