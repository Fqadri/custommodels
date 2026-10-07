from unittest.mock import Mock

import pytest
import torch

from llama2 import rope as rope_module
from llama2.rope import RotaryEmbedding, compute_rope, precompute_rope_params


def test_frequencies_and_angles_repeat_for_adjacent_pairs():
    cos, sin = precompute_rope_params(head_dim=8, context_length=3)
    frequencies = torch.tensor([1.0, 0.1, 0.01, 0.001])
    angles = torch.arange(3).unsqueeze(1) * frequencies.unsqueeze(0)
    expected_angles = angles.repeat_interleave(2, dim=1)

    assert cos.shape == sin.shape == (3, 8)
    torch.testing.assert_close(cos, expected_angles.cos())
    torch.testing.assert_close(sin, expected_angles.sin())
    torch.testing.assert_close(cos[:, 0::2], cos[:, 1::2])
    torch.testing.assert_close(sin[:, 0::2], sin[:, 1::2])


@pytest.mark.parametrize(
    ("cos", "sin", "expected"),
    [
        ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [-2.0, 1.0, -4.0, 3.0]),
        ([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [-2.0, 1.0, 3.0, 4.0]),
    ],
)
def test_rotation_uses_adjacent_pairs(cos, sin, expected):
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    output = compute_rope(x, torch.tensor([cos]), torch.tensor([sin]))
    torch.testing.assert_close(output, torch.tensor(expected).reshape_as(x))


@pytest.mark.parametrize("head_dim", [2, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_matches_complex_rotation(head_dim, dtype):
    torch.manual_seed(123)
    x = torch.randn(2, 5, 3, head_dim, dtype=dtype).transpose(1, 2)
    cos, sin = precompute_rope_params(
        head_dim, theta_base=1000, context_length=7
    )
    output = compute_rope(x, cos, sin)

    frequencies = 1000.0 ** (
        -2 * torch.arange(head_dim // 2, dtype=torch.float32) / head_dim
    )
    angles = torch.outer(torch.arange(5, dtype=torch.float32), frequencies)
    rotations = torch.polar(torch.ones_like(angles), angles)
    pairs = x.float().reshape(2, 3, 5, head_dim // 2, 2).contiguous()
    expected = torch.view_as_real(
        torch.view_as_complex(pairs) * rotations
    ).flatten(-2).to(dtype)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[:, :, 0], x[:, :, 0], rtol=0, atol=0)


def test_rope_preserves_pair_norms_and_gradients():
    torch.manual_seed(123)
    x = torch.randn(2, 3, 5, 8, requires_grad=True)
    cos, sin = precompute_rope_params(8, context_length=5)
    output = compute_rope(x, cos, sin)

    original_norms = x.reshape(2, 3, 5, 4, 2).square().sum(dim=-1)
    rotated_norms = output.reshape(2, 3, 5, 4, 2).square().sum(dim=-1)
    torch.testing.assert_close(rotated_norms, original_norms)

    output.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_rope_rejects_odd_head_dimensions():
    with pytest.raises(AssertionError, match="must be even"):
        precompute_rope_params(3)
    with pytest.raises(AssertionError, match="must be even"):
        compute_rope(
            torch.ones(1, 1, 1, 3), torch.ones(1, 3), torch.zeros(1, 3)
        )


def test_rotary_embedding_precomputes_once_and_reuses_buffers(monkeypatch):
    precompute = Mock(wraps=precompute_rope_params)
    monkeypatch.setattr(rope_module, "precompute_rope_params", precompute)
    rotary = RotaryEmbedding(4, context_length=6)
    cos = rotary.cos
    sin = rotary.sin
    x = torch.randn(2, 3, 5, 4)

    for inputs in (x, x * 2):
        torch.testing.assert_close(rotary(inputs), compute_rope(inputs, cos, sin))
    precompute.assert_called_once_with(
        head_dim=4, theta_base=10_000, context_length=6
    )
    assert rotary.cos is cos
    assert rotary.sin is sin
    assert set(dict(rotary.named_buffers())) == {"cos", "sin"}
    assert not list(rotary.parameters())
    assert not rotary.state_dict()
