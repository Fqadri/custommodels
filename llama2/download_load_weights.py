
from pathlib import Path
import torch
from huggingface_hub import constants, snapshot_download

from llama2.llama2model import Llama2Model
from llama2.llamatokenizer import LlamaTokenizer

# returns path to the tokenizer and checkpoint files
def download_llama2(instructionfinetuned: bool = False) -> tuple[Path, Path]:
    """Download files and return (tokenizer_path, checkpoint_path)."""
    # Honor HF_HOME/HF_HUB_CACHE so the checkpoint stays on the configured cache drive.
    cache_dir = Path(constants.HF_HUB_CACHE)
    print(f"Hugging Face model cache: {cache_dir}")

    model_name = "meta-llama/Llama-2-7b"
    if instructionfinetuned:
        model_name += "-chat"

    model_directory = snapshot_download(
        repo_id=model_name,
        allow_patterns=[ # allowed_patterns basically means which files to download from the repository
            "consolidated.00.pth",
            "params.json",
            "tokenizer.model",
        ],
        cache_dir=cache_dir,
        token=True,  # Use credentials saved by `hf auth login`; never hardcode a token.
    )
    
    print(f"Downloaded LLaMA 2 files: {model_directory}")
    
    snapshot_path = Path(model_directory)
    tokenizer_path = snapshot_path / "tokenizer.model"
    model_weights_checkpoint_path = snapshot_path / "consolidated.00.pth"
    
    return tokenizer_path, model_weights_checkpoint_path

# Copy values into the existing left tensor in place from right
def assign(left, right, tensor_name="unknown") -> None:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
    
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

# Similar to how we loaded weights with GPT2
def load_weights_into_llama(model, param_config, weights_file) :
    # Reject a configuration mismatch before reading the checkpoint or changing any weights.
    num_layers = len(model.trf_blocks)
    if param_config["n_layers"] != num_layers:
        raise ValueError(
            f"Layer count mismatch: config specifies {param_config['n_layers']}, "
            f"but the model has {num_layers} transformer blocks."
        )

    weights = torch.load(
        weights_file,
        map_location="cpu",  # Ensure weights are loaded on the CPU first
        weights_only=True)

    # Require every expected checkpoint layer, with no missing, extra, or skipped indices.
    checkpoint_layers = {
        name.split(".")[1] for name in weights if name.startswith("layers.")
    }
    expected_layers = {str(index) for index in range(num_layers)}
    if checkpoint_layers != expected_layers:
        raise ValueError(
            f"Checkpoint layers do not match the model's {num_layers} transformer blocks. "
            f"Missing indices: {sorted(expected_layers - checkpoint_layers)}; "
            f"unexpected indices: {sorted(checkpoint_layers - expected_layers)}."
        )

    # Load the token embedding weights into the model.
    assign(model.tok_emb.weight, weights["tok_embeddings.weight"])

    # Load the weights for each transformer block in the model.
    for l in range(num_layers):

        q_raw = weights[f"layers.{l}.attention.wq.weight"]
        assign(
            model.trf_blocks[l].attention.W_query.weight,
            q_raw
        )
        k_raw = weights[f"layers.{l}.attention.wk.weight"]
        assign(
            model.trf_blocks[l].attention.W_key.weight,
            k_raw
        )
        assign(
            model.trf_blocks[l].attention.W_value.weight,
            weights[f"layers.{l}.attention.wv.weight"]
        )
        assign(
            model.trf_blocks[l].attention.out_projection.weight,
            weights[f"layers.{l}.attention.wo.weight"]
        )
        assign(
            model.trf_blocks[l].norm1.weight,
            weights[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        assign(
            model.trf_blocks[l].ff.fc1.weight,
            weights[f"layers.{l}.feed_forward.w1.weight"]
        )
        # Meta names the gate/up/down projections w1/w3/w2; ours are fc1/fc2/fc3.
        assign(
            model.trf_blocks[l].ff.fc2.weight,
            weights[f"layers.{l}.feed_forward.w3.weight"]
        )
        assign(
            model.trf_blocks[l].ff.fc3.weight,
            weights[f"layers.{l}.feed_forward.w2.weight"]
        )
        assign(
            model.trf_blocks[l].norm2.weight,
            weights[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    assign(model.final_norm.weight, weights["norm.weight"])
    assign(model.out_head.weight, weights["output.weight"])

    return

def get_model_with_preexisting_weights() -> tuple[Llama2Model, LlamaTokenizer]:
    LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
    }

    model = Llama2Model(LLAMA2_CONFIG_7B)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device);

    tokenizer_file, weight_file = download_llama2()
    
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    if not weight_file.is_file():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    
    tokenizer = LlamaTokenizer(str(tokenizer_file))

    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weight_file)

    return model, tokenizer
