# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

# Default model data download cache directory
DEFAULT_MODELS_DIR = Path(__file__).with_name("gpt2_models")

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
def download_gpt2_modeldata(url, destination, backup_url=None):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    def attempt_download(download_url):
        temporary_path = None
        try:
            with requests.get(download_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                file_size = int(response.headers.get("Content-Length", 0))
                written = 0
                with NamedTemporaryFile(
                    dir=destination.parent, suffix=".part", delete=False
                ) as file:
                    temporary_path = Path(file.name)
                    with tqdm(
                        total=file_size or None,
                        unit="iB",
                        unit_scale=True,
                        desc=destination.name,
                    ) as progress:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                                written += len(chunk)
                                progress.update(len(chunk))
                if written == 0 or (file_size and written != file_size):
                    raise requests.exceptions.ChunkedEncodingError(
                        f"Incomplete download for {destination.name}: "
                        f"expected {file_size} bytes, received {written}."
                    )
                temporary_path.replace(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    try:
        attempt_download(url)
    except requests.exceptions.RequestException as error:
        if backup_url is None:
            raise
        print(f"Primary download failed: {error}. Trying backup: {backup_url}")
        attempt_download(backup_url)


# Downloads GPT2 model data and returns the model hyperparameters and weights.
def get_gpt2_pretrained_modeldata(model_size, models_dir=DEFAULT_MODELS_DIR):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}.")

    model_dir = Path(models_dir) / model_size
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = (
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        download_gpt2_modeldata(
            f"{base_url}/{model_size}/{filename}",
            model_dir / filename,
            f"{backup_base_url}/{model_size}/{filename}",
        )

    # load model hyperparameters from hparams.json
    # Settings - stores the model hyperparameters similar to GPT2_CONFIG_124M dictionary
    # params - dictionary with model weights loaded from the TensorFlow checkpoint.
    # wpe - positional embedding weights. Shape [vocab_size, d_model]
    # wte - token embedding weights. Shape [vocab_size, d_model]. Output layer weights are tied to these weights (weight tying).
    # blocks - list of transformer block weights.
    # b - final layer norm weights bias i.e. shift. Shape [d_model]
    # g - final layer norm weights gain i.e. scale. Shape [d_model]
    with (model_dir / "hparams.json").open(encoding="utf-8") as file:
        settings = json.load(file)
    
    # load GPT model parameters
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    reader = tf.train.load_checkpoint(str(model_dir / "model.ckpt"))
    for name in sorted(reader.get_variable_to_shape_map()):
        variable_array = np.squeeze(reader.get_tensor(name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        target_dict[variable_name_parts[-1]] = variable_array
    
    return settings, params

def print_gpt2_modeldata() -> None:
    # Example usage
    settings, params = get_gpt2_pretrained_modeldata("124M", models_dir=DEFAULT_MODELS_DIR)
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
