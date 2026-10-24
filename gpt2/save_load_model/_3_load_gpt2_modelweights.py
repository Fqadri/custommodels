import numpy as np
import torch

def assign(left, right):
    if left is None:
        raise ValueError("Pretrained GPT-2 weights require a model with qkv_bias=True.")
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")

    return torch.nn.Parameter(
        torch.tensor(right, dtype=left.dtype, device=left.device),
        requires_grad=left.requires_grad,
    )

def validate_model_architecture_settings(gpt, hyperparameters) -> None:
    architecture = {
        "n_vocab": gpt.token_embedding.num_embeddings,
        "n_ctx": gpt.position_embedding.num_embeddings,
        "n_embd": gpt.token_embedding.embedding_dim,
        "n_layer": len(gpt.transformer_blocks),
    }

    for name, actual in architecture.items():
        expected = hyperparameters.get(name)
        if expected != actual:
            raise ValueError(
                f"{name} mismatch: checkpoint={expected}, model={actual}"
            )

    expected_heads = hyperparameters["n_head"]
    for index, block in enumerate(gpt.transformer_blocks):
        actual_heads = block.attention.num_heads
        if expected_heads != actual_heads:
            raise ValueError(
                f"Layer {index} n_head mismatch: "
                f"checkpoint={expected_heads}, model={actual_heads}"
            )

def load_weights_into_gpt(gpt, params, hyperparameters):
    
    validate_model_architecture_settings(gpt, hyperparameters)
    
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