# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
from tqdm import tqdm

DEFAULT_MODELS_DIR = Path(__file__).with_name("gpt2_models")


def download_and_load_gpt2(model_size, models_dir=DEFAULT_MODELS_DIR):
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
        download_file(
            f"{base_url}/{model_size}/{filename}",
            model_dir / filename,
            f"{backup_base_url}/{model_size}/{filename}",
        )

    with (model_dir / "hparams.json").open(encoding="utf-8") as file:
        settings = json.load(file)
    params = load_gpt2_params_from_tf_ckpt(str(model_dir / "model.ckpt"), settings)
    return settings, params


def download_file(url, destination, backup_url=None):
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


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    reader = tf.train.load_checkpoint(ckpt_path)
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

    return params
