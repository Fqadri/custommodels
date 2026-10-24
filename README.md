# Custom Models

## Description

Custom implementations of transformer models for learning how their building
blocks and text generation work.

### GPT-2

The repository starts with a GPT-2 implementation covering attention, layer
normalization, feed-forward networks, transformer blocks, greedy and sampled
generation, pretrained-weight loading, and supervised instruction fine-tuning.

### LLaMA 2

LLaMA 2 is assembled from the same shared attention and transformer-block
components, with these changes from GPT-2:

- **RoPE** replaces learned absolute positional embeddings.
- **RMSNorm** replaces LayerNorm.
- **SwiGLU** replaces the GELU feed-forward network.
- Attention and feed-forward projections are **bias-free**.

## Structure

- [common/](common/) - Shared causal attention, pre-norm transformer blocks,
  generation, tokenizer helpers, and [training/evaluation](common/pretraining.py).
  Callers supply the model, tokenizer, context size, and corpus path.
- [gpt2/](gpt2/) - Learned positional embeddings, LayerNorm, GELU, and a
  tiktoken byte-level BPE wrapper.
- [llama2/](llama2/) - LLaMA 2 entry points and training workflows.
  [model/](llama2/model/) contains the model, SentencePiece tokenizer, RoPE,
  RMSNorm, SwiGLU, and bias-free transformer blocks.
  [save_load_model/](llama2/save_load_model/) handles checkpoint downloads and loading.
- [tests/](tests/) - Component, generation, training, and checkpoint-loading tests.
- [qwen3/](qwen3/) - Placeholder for a future model.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from
the repository root:

```powershell
uv sync
```

The project uses Python 3.11. Run the commands below from the repository root
so package imports resolve.

The [notebooks/](notebooks/) examples use Transformers 4.x, Accelerate, and
BertViz with a compatible Hugging Face Hub version. Select the project's
`.venv` kernel and restart it after dependency updates. vLLM is not included;
its notebooks need a separate compatible Linux/WSL environment. CUDA-only
cells also require a CUDA-enabled environment or CPU-safe changes.

## GPT-2

[run_gpt2_model_raw](gpt2/model/_12_gpt2model.py) accepts these optional arguments:

- `prompt`: defaults to `"Every effort moves you"`.
- `max_new_tokens`: defaults to `15` additional tokens for each of the two
  greedy generations and the sampled generation, excluding the prompt tokens.
- `dtype`: defaults to `torch.float32`.

For example:

```powershell
uv run python -c "from gpt2.model._12_gpt2model import run_gpt2_model_raw; run_gpt2_model_raw(prompt='Every effort moves you', max_new_tokens=15)"

uv run python -c "import torch; from gpt2.model._12_gpt2model import run_gpt2_model_raw; run_gpt2_model_raw(dtype=torch.bfloat16)"
```

### Pre-training

[run_pretraining_eval_gpt2](gpt2/pretraining_eval/_2_pretraining_eval.py) trains
for 10 epochs and accepts these optional arguments:

- `data_path`: a string or `Path`, defaulting to the bundled
  [the-verdict.txt](gpt2/pretraining_eval/the-verdict.txt).
- `max_new_tokens`: defaults to `15` additional tokens for each final
  greedy/sampled output. Epoch-end previews retain their 50-token limit.
- `prompt`: defaults to `"Every effort moves you"` and is used for both
  epoch-end previews and final generation.

```powershell
uv run python -c "from gpt2.pretraining_eval._2_pretraining_eval import run_pretraining_eval_gpt2; run_pretraining_eval_gpt2(data_path=r'gpt2\pretraining_eval\the-verdict.txt', max_new_tokens=15, prompt='Every effort moves you')"
```

### Instruction fine-tuning

From the repository root, run this in CMD or PowerShell. `uv run` uses the
project environment without requiring activation:

[run_instruction_finetuning](gpt2/finetuning_instruction/_3_instr_finetuning.py)
is a plain Python function with five optional parameters:

- `prompt`: defaults to `"Translate to Spanish - Hello World"`.
- `max_new_tokens`: defaults to `256`; limits the before/after responses and
  saved test responses. Epoch previews retain their existing 50-token limit.
- `data_path`: defaults to the bundled
  [instruction-data.json](gpt2/finetuning_instruction/instruction-data.json),
  containing 1,100 instruction examples to split into training, validation,
  and test sets.
- `model_size`: defaults to `"124M"`; supports `"124M"`, `"355M"`, `"774M"`,
  and `"1558M"`. Selects the matching model configuration and pretrained weights.
- `run_test_data`: defaults to `False`. Set to `True` to generate and save
  responses for the held-out test dataset after fine-tuning. Training,
  before/after prompt responses, checkpoint saving, and loss plotting still
  run when it is `False`.

Instruction data is local-only. Missing files raise `FileNotFoundError`;
no dataset download is attempted. OpenAI model weights are downloaded on
first use and cached for later runs.

```powershell
uv run python -c "from gpt2.finetuning_instruction._3_instr_finetuning import run_instruction_finetuning; run_instruction_finetuning(prompt='Translate to Spanish - Hello World', max_new_tokens=100, data_path=r'gpt2\finetuning_instruction\instruction-data.json', model_size='124M', run_test_data=False)"
```

The workflow:

- Loads OpenAI's pretrained weights into the custom GPT-2 model.
- Wraps the supplied prompt in the training dataset's Alpaca instruction format
  and prints the pretrained model's response.
- Prepares instruction datasets and batches, then evaluates initial losses.
- Instruction-fine-tunes the same model.
- Runs the same formatted prompt again with the same greedy decoding settings
  and token limit, then prints the fine-tuned response.
- Saves the checkpoint and loss plot; generates and saves test responses only
  when `run_test_data=True`.

Other settings are fixed in the function: batch size **8**,
**2 epochs**, context length **1024**, and zero data-loader workers. CUDA is
used when available, otherwise CPU. This runs fine-tuning, not pretraining
from scratch.

There is no argument parser, subcommand, or command-line flag interface.
Call the function directly, as above; importing the module does not start it.
Weights are cached under `gpt2\save_load_model\gpt2_models`; saved outputs go to
`outputs\gpt2\instruction`:

- `gpt2-<model_size>-sft.pth`: fine-tuned model weights (`gpt2-355M-sft.pth` by default).
- `losses.png`: training and validation loss plot.
- `instruction-data-with-response.json`: generated test responses, written only
  when `run_test_data=True`.

The function returns the saved checkpoint path.
Automatic response scoring is not implemented.

## LLaMA 2

[run_llama2_model](llama2/model/llama2_model.py) runs generation using the pretrained base LLaMA 2 7B
checkpoint. The demo currently uses these fixed settings:

- Prompt: `"Every effort moves you"`.
- Two greedy generations of up to `10` new tokens each.
- One sampled generation of up to `15` new tokens, stopping early on EOS.

Unlike the GPT-2 generation function, this function does not currently
accept prompt or token-limit arguments. Importing the module does not run
the demo; call the function explicitly.

Obtain access to [Meta's LLaMA 2 repository](https://huggingface.co/meta-llama/Llama-2-7b),
then authenticate and run:

```powershell
$env:HF_HOME = "D:\hf_cache"
uv run hf auth login
uv run python -X utf8 -u -c "from llama2.model.llama2_model import run_llama2_model; run_llama2_model()"
```

The entry point loads the **base LLaMA 2 7B checkpoint** into the custom model,
then performs greedy and sampled generation. It currently runs on CPU with
bfloat16 weights. Allow roughly **27+ GiB of RAM**, plus runtime headroom:
the current loader holds both the model and checkpoint in memory.

[load_llama2_modelweights.py](llama2/save_load_model/load_llama2_modelweights.py) downloads and maps
Meta's original `consolidated.00.pth` checkpoint layer by layer. It validates
layer counts and tensor shapes, copies parameters in place, and preserves
the adjacent-pair Q/K layout used by our RoPE.

For **download only**, without allocating a model:

```powershell
uv run python -c "from llama2.save_load_model.load_llama2_modelweights import download_llama2; print(download_llama2())"
```

The helper returns `(tokenizer_path, checkpoint_path)` as `Path` objects.
It honors `HF_HOME` / `HF_HUB_CACHE`; the example above uses `D:\hf_cache\hub`.
The weights are approximately 13.5 GB, and cached files are reused.

### Pre-training

[The pretraining runner](llama2/pretraining_eval/_pretraining_eval.py) accepts:

- `--tokenizer`: required path to a 32,000-token SentencePiece tokenizer model.
- `--data`: defaults to the bundled
  [the-verdict.txt](llama2/pretraining_eval/the-verdict.txt).
- `--epochs`: defaults to `10`.

```powershell
uv run -m llama2.pretraining_eval._pretraining_eval --tokenizer C:\models\llama\tokenizer.model --epochs 1
```

This trains a **134M-parameter float32 learning model**, not the official 7B
model, using the shared training code. It evaluates losses, generates epoch-end
samples, plots losses, and runs final inference. CUDA is used when available.

### Instruction fine-tuning

A LLaMA instruction-fine-tuning workflow is **not yet implemented** in this
repository.

For inference with Meta's already fine-tuned chat checkpoint, pass
`instructionfinetuned=True` to
[get_llama_model_using_weights](llama2/model/llama2_model.py) to select `Llama-2-7b-chat`.
This loads existing weights; it does not run fine-tuning. The caller decides
when to call `model.eval()`.

Chat weights do **not** automatically apply a chat template. Format a single
user turn as `[INST] Your instruction [/INST]`. The tokenizer already prepends
BOS (`<s>`) without appending EOS; do not add BOS twice. Plain prompts such as
`Every effort moves you` are suitable for base-model text completion.

## Checks

```powershell
$env:MPLBACKEND = "Agg"  # Headless plotting; no Tk installation required.
uv run python -m pytest -q
uv pip check
```

Weight-loader tests use synthetic checkpoints and mocked downloads. Full
pretrained 7B generation has not yet been verified end to end in this project.

## Start JupyterLab

From PowerShell:

```powershell
Set-Location D:\GitHub\custom-models
uv run jupyter lab
```

JupyterLab uses the project's environment and normally opens your browser
automatically. If it does not, open the URL printed in the terminal
(`http://localhost:8888/lab?token=...`). Keep the token private.
