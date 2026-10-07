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
- [llama2/](llama2/) - Interleaved RoPE on Q/K, RMSNorm, SwiGLU, bias-free
  attention, and a SentencePiece wrapper.
- [tests/](tests/) - Component, generation, training, and checkpoint-loading tests.
- [qwen3/](qwen3/) - Placeholder for a future model.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from
the repository root:

```powershell
uv sync
```

The project uses Python 3.11. Run entry points with `-m` so package imports resolve.

## GPT-2

```powershell
# Randomly initialized generation demo; float32 by default.
uv run -m gpt2.model.main

# Pretrain from scratch on the bundled corpus, plot losses, then generate.
uv run -m gpt2.pretraining_eval._2_pretraining_eval
```

Random weights produce incoherent text. The small pretraining corpus is for
learning, not reproducing OpenAI GPT-2's capabilities.

### Instruction fine-tuning

```powershell
uv run -m gpt2.finetuning_instruction.main preview
uv run -m gpt2.finetuning_instruction.main prepare
uv run -m gpt2.finetuning_instruction.main train --model 124M --batch-size 2 --epochs 1 --context-length 256
```

Fine-tuning loads OpenAI's pretrained weights and uses the bundled 1,100-example
instruction dataset. Weights are cached under `gpt2\save_load_model\gpt2_models`;
checkpoints, loss plots, and test responses go to `outputs\gpt2\instruction`.
Use `train --help` for data, device, model-size, and output options.
Automatic response scoring is not implemented.

## LLaMA 2 pretrained inference

Obtain access to [Meta's LLaMA 2 repository](https://huggingface.co/meta-llama/Llama-2-7b),
then authenticate and run:

```powershell
$env:HF_HOME = "D:\hf_cache"
uv run hf auth login
uv run python -X utf8 -u -m llama2.main
```

The entry point loads the **base LLaMA 2 7B checkpoint** into the custom model,
then performs greedy and sampled generation. It currently runs on CPU with
bfloat16 weights. Allow roughly **27+ GiB of RAM**, plus runtime headroom:
the current loader holds both the model and checkpoint in memory.

[download_load_weights.py](llama2/download_load_weights.py) downloads and maps
Meta's original `consolidated.00.pth` checkpoint layer by layer. It validates
layer counts and tensor shapes, copies parameters in place, and preserves
the adjacent-pair Q/K layout used by our RoPE.

For **download only**, without allocating a model:

```powershell
uv run python -c "from llama2.download_load_weights import download_llama2; print(download_llama2())"
```

The helper returns `(tokenizer_path, checkpoint_path)` as `Path` objects.
It honors `HF_HOME` / `HF_HUB_CACHE`; the example above uses `D:\hf_cache\hub`.
The weights are approximately 13.5 GB, and cached files are reused.

### Base versus chat weights

Pass `instructionfinetuned=True` to
[get_llama_model_using_weights](llama2/main.py) to select `Llama-2-7b-chat`.
The caller decides when to call `model.eval()`.

Chat weights do **not** automatically apply a chat template. Format a single
user turn as `[INST] Your instruction [/INST]`. The tokenizer already prepends
BOS (`<s>`) without appending EOS; do not add BOS twice. Plain prompts such as
`Every effort moves you` are suitable for base-model text completion.

## LLaMA-style pretraining

```powershell
uv run -m llama2.pretraining_eval._pretraining_eval --tokenizer C:\models\llama\tokenizer.model --epochs 1
```

This trains a **134M-parameter float32 learning model**, not the official 7B
model, using the shared training code. It evaluates losses, generates epoch-end
samples, plots losses, and runs final inference. CUDA is used when available.

The runner defaults to 10 epochs and its own copy of
[the-verdict.txt](llama2/pretraining_eval/the-verdict.txt); GPT-2's original is
retained. Use `--data` for another corpus. A compatible 32,000-token SentencePiece
model file is required. LLaMA instruction fine-tuning is not yet implemented.

## Checks

```powershell
uv run python -m pytest -q
uv pip check
```

Weight-loader tests use synthetic checkpoints and mocked downloads. Full
pretrained 7B generation has not yet been verified end to end in this project.
