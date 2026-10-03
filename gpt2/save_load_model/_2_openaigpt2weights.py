from .gpt_download import DEFAULT_MODELS_DIR, download_and_load_gpt2

# This file demostrates downloading weights from OpenAI's GPT-2 and inspecting them.


# GPT2 comes in different sizes. Hyperparameters for different sizes can be found in hparams.json file in the downloaded model folder.
# 124M in gpt2-small which has 12 layers, 768 hidden size, 12 attention heads
# 355M in gpt2-medium which has 24 layers, 1024 hidden size, 16 attention heads
# 774M in gpt2-large which has 36 layers, 1280 hidden size, 20 attention heads
# 1558M in gpt2-xl which has 48 layers, 1600 hidden size, 25 attention heads
# Files downloaded include:
#   - checkpoint - TensorFlow checkpoint file prefix
#   - encoder.json - tokenizer encoder file. Contains the mapping of tokens to their IDs using BPE.
#   - hparams.json - model hyperparameters
#   - model.ckpt.data-00000-of-00001 - Contains the actual tensor values — all the learned weight matrices and biases. Think of it as the raw numeric data blob.
#   - model.ckpt.index - TensorFlow checkpoint index file
#   - model.ckpt.meta - TensorFlow checkpoint meta graph file
#   - vocab.bpe - Defines how those tokens were created using Byte Pair Encoding (BPE) merge rules. Basically merge rules (how to build tokens from characters) and encoder.json final vocab mapping.
def load_openai_weights_gpt2(model_size, models_dir):
    # Download and load the GPT-2 model weights and configuration
    settings, params = download_and_load_gpt2(model_size, models_dir=models_dir)

    # Settings - stores the model hyperparameters similar to GPT2_CONFIG_124M dictionary
    # params - dictionary with model weights loaded from the TensorFlow checkpoint.
    # wpe - positional embedding weights. Shape [vocab_size, d_model]
    # wte - token embedding weights. Shape [vocab_size, d_model]. Output layer weights are tied to these weights (weight tying).
    # blocks - list of transformer block weights.
    # b - final layer norm weights bias i.e. shift. Shape [d_model]
    # g - final layer norm weights gain i.e. scale. Shape [d_model]
    return settings, params


def main() -> None:
    # Example usage
    settings, params = load_openai_weights_gpt2("124M", models_dir=DEFAULT_MODELS_DIR)
    print("Model settings:", settings)
    print("Model parameter dictionary keys:", params.keys())

    print(
        "Weights involved in first transformer block:", params["blocks"][0].keys()
    )  # attn - attention, mlp - feed-forward network, ln_1 - layer norm before attention, ln_2 - layer norm before feed-forward network
    print(
        "Weights involved in first transformer block attention component:",
        params["blocks"][0]["attn"].keys(),
    )  # c_attn - weights for Q, K, V. c_proj - output projection weights.
    print(
        "Weights (and bias) involved in first transformer block attention component involved in Q, K, V:",
        params["blocks"][0]["attn"]["c_attn"].keys(),
    )  # w - weights, b - bias
    print(
        "Weights of the first transformer block's attention component involved in Q, K, V:",
        params["blocks"][0]["attn"]["c_attn"]["w"],
    )  # Outputs concatenated weights for Q, K, and V

    # Extract and print the Q, K, V weights from the concatenated weights
    attn_w = params["blocks"][0]["attn"]["c_attn"]["w"]
    third = attn_w.shape[1] // 3

    # Slice into Q, K, V
    Wq = attn_w[:, :third]
    Wk = attn_w[:, third : 2 * third]
    Wv = attn_w[:, 2 * third :]

    # Output shapes and sample values
    print("Wq shape:", Wq.shape)
    print("Wk shape:", Wk.shape)
    print("Wv shape:", Wv.shape)

    print(
        "Output layer weights of first transformer block:",
        params["blocks"][0]["attn"]["c_proj"].keys(),
    )  # w - weights, b - bias
    print(
        "Weights involved in FF layer in first transformer block:",
        params["blocks"][0]["mlp"].keys(),
    )  # c_fc - first hidden layer, c_proj - output layer


if __name__ == "__main__":
    main()
