
import torch
import torch.nn as nn

# Llama uses the SiLU activation function (instead of GELU), which is also known as the Swish function
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
# Llama uses a "Gates Linear Unit" (GLU) variant of SiLU called SwiGLU, which essentially results in a slightly differently structured FeedForward module
# https://arxiv.org/abs/2002.05202

# We also added a dtype=cfg["dtype"] setting above, which will allow us to load the model directly in lower precision formats later to reduce memory usage (versus instantiating it in the original 32-bit precision format and then converting it)
# Llama doesn't use any bias units
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(123)  # For reproducibility
    batch_example = torch.randn(1, 2, 768) # Example batch of input sequences. Shape 1 X 2 X 768 (B X T X D). 1 input sequences each of length 2 tokens and 768 dimensions.

    cfg = {
        "emb_dim": 768,    # Dimension of the model (d_model).
        "hidden_dim": 3072, # Hidden dimension in the feed-forward network (usually 4 * emb_dim).
        "dtype": torch.float32, # Data type for the linear layers.
    }

    feedforward = FeedForward(cfg) # d_model = 768 as in GPT-2 small
    out = feedforward(batch_example)

    print("Output shape after FeedForward:", out.shape)  #Should print (1, 2, 768)
    print("Output after FeedForward:", out)
