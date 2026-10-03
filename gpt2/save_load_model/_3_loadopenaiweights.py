import numpy as np
import tiktoken
import torch

from ..model._12_gpt2model import (
    GPT2Model,
    generate_text_from_inputsample_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
)
from .gpt_download import DEFAULT_MODELS_DIR, download_and_load_gpt2

# This file demostrates downloading weights from OpenAI's GPT-2 and loading them into a custom GPT-2 implementation.


def assign(left, right):
    if left is None:
        raise ValueError("Pretrained GPT-2 weights require a model with qkv_bias=True.")
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")

    return torch.nn.Parameter(
        torch.tensor(right, dtype=left.dtype, device=left.device),
        requires_grad=left.requires_grad,
    )


def load_weights_into_gpt(gpt, params):
    if len(params["blocks"]) != len(gpt.transformer_blocks):
        raise ValueError(
            "Checkpoint and model have different numbers of transformer blocks."
        )
    # Load positional and input token embedding weights.
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params["wpe"])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )  # Split concatenated weights into Q, K, V weight matrices.
        # Load Q, K, V weights into our custom GPT-2 model's multi-head attention layers.
        # Note the transpose operation. This is because in our custom implementation we have defined the weight matrices to be of shape (dim_in, dim_out) whereas in OpenAI GPT-2 they are stored as (dim_out, dim_in).
        gpt.transformer_blocks[b].attention.W_query.weight = assign(
            gpt.transformer_blocks[b].attention.W_query.weight, q_w.T
        )
        gpt.transformer_blocks[b].attention.W_key.weight = assign(
            gpt.transformer_blocks[b].attention.W_key.weight, k_w.T
        )
        gpt.transformer_blocks[b].attention.W_value.weight = assign(
            gpt.transformer_blocks[b].attention.W_value.weight, v_w.T
        )

        # Load Q, K, V biases into our custom GPT-2 model's multi-head attention layers.
        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )  # Split concatenated biases into Q, K, V biases.
        gpt.transformer_blocks[b].attention.W_query.bias = assign(
            gpt.transformer_blocks[b].attention.W_query.bias, q_b
        )
        gpt.transformer_blocks[b].attention.W_key.bias = assign(
            gpt.transformer_blocks[b].attention.W_key.bias, k_b
        )
        gpt.transformer_blocks[b].attention.W_value.bias = assign(
            gpt.transformer_blocks[b].attention.W_value.bias, v_b
        )

        # Load output projection weights and bias of multi-head attention layer.
        gpt.transformer_blocks[b].attention.out_projection.weight = assign(
            gpt.transformer_blocks[b].attention.out_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].attention.out_projection.bias = assign(
            gpt.transformer_blocks[b].attention.out_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # Load feed-forward network weights and biases.
        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )  # First linear layer weights
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )  # First linear layer bias
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )  # Second linear layer weights
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )  # Second linear layer bias

        # Load layer normalization weights and biases.
        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )  # Layer norm before attention scale (gain)
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )  # Layer norm before attention shift (bias)
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )  # Layer norm before feed-forward scale (gain)
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )  # Layer norm before feed-forward shift (bias)

    # Load final layer normalization weights and biases.
    gpt.final_norm.scale = assign(
        gpt.final_norm.scale, params["g"]
    )  # Final layer norm scale (gain)
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift, params["b"]
    )  # Final layer norm shift (bias)
    # Load output layer weights. Note that output layer weights are tied to the input token embedding weights (weight tying).
    gpt.output_layer.weight = gpt.token_embedding.weight


def main() -> None:
    GPT2_CONFIG_124M = {
        "vocab_size": 50257,  # Size of the vocabulary.
        "dim_model": 768,  # Dimension of the model (d_model).
        "num_layers": 12,  # Number of transformer layers or blocks.
        "num_heads": 12,  # Number of attention heads per layer.
        "context_length": 1024,  # Maximum context length (number of tokens in input sequence).
        "drop_rate": 0.1,  # Dropout rate to prevent overfitting.
        "qkv_bias": False,  # Whether to include bias terms in Q, K, V projections.
    }

    model_configs = {
        "gpt-small (124M)": {"dim_model": 768, "num_heads": 12, "num_layers": 12},
        "gpt-medium (355M)": {"dim_model": 1024, "num_heads": 16, "num_layers": 24},
        "gpt-large (774M)": {"dim_model": 1280, "num_heads": 20, "num_layers": 36},
        "gpt-xl (1558M)": {"dim_model": 1600, "num_heads": 25, "num_layers": 48},
    }

    # Update model hyperparameters based on the selected model size
    model_name = "gpt-small (124M)"
    NEW_CONFIG = GPT2_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update(
        {"context_length": 1024}
    )  # Open AI GPT-2 uses 1024 context length for all model sizes.
    NEW_CONFIG.update(
        {"qkv_bias": True}
    )  # Open AI GPT-2 uses qkv_bias=True for all model sizes.

    gpt = GPT2Model(NEW_CONFIG)
    gpt.eval()

    _settings, params = download_and_load_gpt2("124M", models_dir=DEFAULT_MODELS_DIR)
    load_weights_into_gpt(gpt, params)

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_from_inputsample_with_sampling(
        gpt,
        text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT2_CONFIG_124M["context_length"],
        temperature=1.5,  # Higher the temperature, more random the output. Lower the temperature, more deterministic the output.
        top_k=50,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text with sampling:\n", generated_text)


if __name__ == "__main__":
    main()
