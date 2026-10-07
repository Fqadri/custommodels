from torch import nn

# Shared transformer block used by various models (e.g., GPT-2, Llama).
class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        ff: nn.Module,
        norm1: nn.Module,
        norm2: nn.Module,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.ff = ff
        self.norm1 = norm1
        self.norm2 = norm2
        self.dropout = nn.Dropout(drop_rate)

    # Normalize before attention and feed-forward (pre-norm); x has shape (B, T, d_model).
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        x = self.attention(x)  # Shape (B, T, d_model)
        x = self.dropout(x)

        x = x + shortcut  # Add the original input (shortcut connection).

        shortcut = x  # Shortcut connection to the feed-forward network.
        x = self.norm2(x)

        x = self.ff(x)  # Also referred to as MLP.
        x = self.dropout(x)

        x = x + shortcut  # Add the original input (shortcut connection).

        return x
