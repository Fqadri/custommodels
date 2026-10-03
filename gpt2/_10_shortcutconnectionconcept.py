
import torch
import torch.nn as nn

# Here we will go over the concept of shortcut connections (also known as residual connections) in neural networks.

class ExampleShortcutConnection(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            # Forward method is automatically called since we are using nn.Sequential
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()), # First layer. Forward pass will call these in order.
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU())
        ])

    # x = input tensor of shape (B, T, d_model)
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape: # Ensure shapes match for addition
                x = x + layer_output  # Add shortcut connection. x came from previous layer.
            else:
                x = layer_output # No shortcut connection

        return x
    
def print_gradients(model, x):
    output = model(x) # Forward pass through model.
    target = torch.tensor([[0.]]) # Dummy target, for simplicity, for loss computation

    loss_fn = nn.MSELoss() # Mean Squared Error loss function
    loss = loss_fn(output, target)

    loss.backward() # Backpropagation to compute gradients. PyTorch computes gradients for each layer automatically.

    for name, param in model.named_parameters():
        if 'weight' in name: # Only print gradients for weight parameters
            print(f"{name} has a gradient mean of {param.grad.abs().mean().item()}")

# Example usage
torch.manual_seed(123)  # For reproducibility
layer_sizes = [3, 3, 3, 3, 3, 1] # Example layer sizes
sample_input = torch.tensor([[[0.1, 0.0, -1.0]]]) # Example input tensor of shape (1, 1, 3)

print("Without Shortcut Connections:\n")
model_without_shortcut = ExampleShortcutConnection(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)

print("\nWith Shortcut Connections:\n")
model_with_shortcut = ExampleShortcutConnection(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)