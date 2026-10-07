import torch
from torch import nn

from common.transformer_block import TransformerBlock
from gpt2.model._11_transformerblock import TransformerBlock as GPT2TransformerBlock


def test_shared_block_applies_components_in_pre_norm_order():
    calls = []

    class RecordingScale(nn.Module):
        def __init__(self, name, factor):
            super().__init__()
            self.name = name
            self.factor = factor

        def forward(self, x):
            calls.append((self.name, x.detach().clone()))
            return x * self.factor

    block = TransformerBlock(
        attention=RecordingScale("attention", 3),
        ff=RecordingScale("ff", 7),
        norm1=RecordingScale("norm1", 2),
        norm2=RecordingScale("norm2", 5),
    )
    x = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    output = block(x)

    assert [name for name, _ in calls] == ["norm1", "attention", "norm2", "ff"]
    for (_, actual), multiplier in zip(calls, [1, 2, 7, 35], strict=True):
        torch.testing.assert_close(actual, multiplier * x.detach())
    torch.testing.assert_close(output, 252 * x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 252))


def test_shared_block_dropout_preserves_residuals_and_respects_eval_mode():
    block = TransformerBlock(
        attention=nn.Identity(),
        ff=nn.Identity(),
        norm1=nn.Identity(),
        norm2=nn.Identity(),
        drop_rate=1.0,
    )
    x = torch.tensor([[[1.0, 2.0]]])
    torch.testing.assert_close(block(x), x)

    block.eval()
    torch.testing.assert_close(block(x), 4 * x)


def test_gpt2_inherits_shared_forward_without_changing_checkpoint_paths():
    config = {
        "dim_model": 8,
        "num_heads": 2,
        "context_length": 8,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }
    block = GPT2TransformerBlock(config, debug=True, dtype=torch.float64)
    assert GPT2TransformerBlock.forward is TransformerBlock.forward
    assert list(dict(block.named_children())) == [
        "attention", "ff", "norm1", "norm2", "dropout"
    ]
    assert block.attention.debug
    assert block.dropout.p == config["drop_rate"]
    assert block.norm1 is not block.norm2
    assert all(parameter.dtype == torch.float64 for parameter in block.parameters())
    assert {
        "attention.W_query.weight",
        "attention.out_projection.weight",
        "ff.layers.0.weight",
        "ff.layers.2.weight",
        "norm1.scale",
        "norm2.shift",
    } <= set(block.state_dict())
