import pytest
import torch
from torch.nn import functional as F

from common.multi_head_attention import MultiHeadAttention
from gpt2.model._6_multiHeadAttention import MultiHeadAttention as DemoAttention
from llama2.rope import RotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("qkv_bias", [False, True])
@pytest.mark.parametrize("out_bias", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_attention_matches_pytorch_causal_attention(
    dtype, qkv_bias, out_bias, use_rope
):
    torch.manual_seed(123)
    attention = MultiHeadAttention(
        dim_in=6,
        dim_out=8,
        context_length=8,
        dropout=0.0,
        num_heads=2,
        qkv_bias=qkv_bias,
        dtype=dtype,
        out_bias=out_bias,
        rope=RotaryEmbedding(4, context_length=8) if use_rope else None,
    )
    x = torch.randn(2, 5, 6, dtype=dtype, requires_grad=True)
    output = attention(x)

    queries = attention.W_query(x).reshape(2, 5, 2, 4).transpose(1, 2)
    keys = attention.W_key(x).reshape(2, 5, 2, 4).transpose(1, 2)
    values = attention.W_value(x).reshape(2, 5, 2, 4).transpose(1, 2)
    if use_rope:
        angles = torch.outer(torch.arange(5), torch.tensor([1.0, 0.01]))
        rotations = torch.polar(torch.ones_like(angles), angles)

        def rotate(tensor):
            pairs = tensor.reshape(2, 2, 5, 2, 2).contiguous()
            rotated = torch.view_as_complex(pairs) * rotations
            return torch.view_as_real(rotated).flatten(-2)

        queries = rotate(queries)
        keys = rotate(keys)

    expected = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
    expected = expected.transpose(1, 2).reshape(2, 5, 8)
    expected = attention.out_projection(expected)

    assert output.shape == (2, 5, 8)
    assert output.dtype == dtype
    assert (attention.out_projection.bias is not None) == out_bias
    for projection in (attention.W_query, attention.W_key, attention.W_value):
        assert (projection.bias is not None) == qkv_bias
    torch.testing.assert_close(output, expected)
    actual_gradient = torch.autograd.grad(output.square().sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("use_rope", [False, True])
def test_attention_prevents_future_tokens_from_affecting_prefix(use_rope):
    torch.manual_seed(123)
    attention = MultiHeadAttention(
        8, 8, 8, dropout=0.0, num_heads=2,
        rope=RotaryEmbedding(4, context_length=8) if use_rope else None,
    ).eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:])
    torch.testing.assert_close(
        attention(x)[:, :2], attention(changed)[:, :2], rtol=0, atol=0
    )


def test_numbered_demo_exposes_the_shared_attention_class():
    assert DemoAttention is MultiHeadAttention


def test_attention_defaults_preserve_gpt2_checkpoint_layout():
    attention = MultiHeadAttention(8, 8, 8, dropout=0.0, num_heads=2)
    assert attention.rope is None
    assert attention.out_projection.bias is not None
    assert set(attention.state_dict()) == {
        "mask", "W_query.weight", "W_key.weight", "W_value.weight",
        "out_projection.weight", "out_projection.bias",
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_llama_attention_supports_lower_precision_backward(dtype):
    torch.manual_seed(123)
    attention = MultiHeadAttention(
        8, 8, 8, dropout=0.0, num_heads=2,
        qkv_bias=False, out_bias=False, dtype=dtype,
        rope=RotaryEmbedding(4, context_length=8),
    )
    x = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    output = attention(x)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for parameter in attention.parameters():
        assert parameter.dtype == dtype
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rope_buffers_follow_parent_attention_device():
    attention = MultiHeadAttention(
        8, 8, 8, dropout=0.0, num_heads=2,
        rope=RotaryEmbedding(4, context_length=8),
    )
    assert "rope.cos" in dict(attention.named_buffers())
    assert "rope.sin" in dict(attention.named_buffers())
    assert not any(key.startswith("rope.") for key in attention.state_dict())

    attention.to("meta")
    assert all(buffer.device.type == "meta" for buffer in attention.buffers())
    assert all(parameter.device.type == "meta" for parameter in attention.parameters())
