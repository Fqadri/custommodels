# Custom Models

## Description

Custom implementations of transformer models for learning how their building
blocks and text generation work. The repository currently includes a PyTorch
GPT-2 implementation covering attention, layer normalization, feed-forward
networks, transformer blocks, greedy and sampling-based generation, pretrained
weight loading, and supervised instruction fine-tuning.
A separate folder is reserved for Qwen3, with more models to be added over time.

## Folder organization

Each model lives in its own top-level folder. All models share one
[uv](https://docs.astral.sh/uv/) project and dependency environment; there is no
shared Python package that model folders must be placed inside.

- [gpt2/](gpt2/) - GPT-2 implementation and demos.
  - [model/](gpt2/model/) contains the architecture and numbered building-block
    examples. [_12_gpt2model.py](gpt2/model/_12_gpt2model.py) defines the model
    configuration, model class, and text-generation helpers;
    [main.py](gpt2/model/main.py) runs the generation demo.
  - [finetuning_instruction/](gpt2/finetuning_instruction/) contains instruction
    data, prompt formatting, dataset preparation, and the fine-tuning workflow.
    Its [main.py](gpt2/finetuning_instruction/main.py) selects which step to run.
  - [pretraining_eval/](gpt2/pretraining_eval/) contains reusable loss, training,
    evaluation, and plotting helpers, plus guarded learning demos.
  - [save_load_model/](gpt2/save_load_model/) contains checkpoint download,
    pretrained-weight mapping, and save/load helpers.
- [qwen3/](qwen3/) - Placeholder for a future Qwen3 implementation;
  [main.py](qwen3/main.py) is currently empty.
- [tests/](tests/) - Automated tests for generation, fine-tuning, and weight loading.
- [pyproject.toml](pyproject.toml) - Project metadata and shared dependencies.
- [uv.lock](uv.lock) - Resolved dependency versions.
- [.python-version](.python-version) - Python version used by uv (3.11).

## Run locally

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Open a terminal in the repository root.
3. Install the dependencies and run the GPT-2 demo:

   ```powershell
   uv sync
   uv run -m gpt2.model.main
   ```

uv manages Python 3.11 and the project's virtual environment. Run model entry
points with `-m` from the repository root so package imports resolve correctly,
rather than executing `gpt2\model\main.py` directly.

The demo starts with `"Every effort moves you"` and generates two identical
greedy continuations of 10 tokens, followed by a 15-token sampled continuation.
The model is randomly initialized, not pretrained, so incoherent output is
expected. The first run requires internet access to download GPT-2 tokenizer
data, which tiktoken caches locally.

### Fine-tune GPT-2 on instructions

Run these commands from the repository root. First, inspect the prompts and
dataset batches without loading a model or starting training:

```powershell
uv run -m gpt2.finetuning_instruction.main preview
uv run -m gpt2.finetuning_instruction.main prepare
```

The bundled [instruction-data.json](gpt2/finetuning_instruction/instruction-data.json)
contains 1,100 examples, split into 935 training, 55 validation, and 110 test
entries. The default data path is relative to the module, not the terminal's
working directory.

For a lower-memory starting run, use the 124M model, a smaller batch, and shorter
training sequences:

```powershell
uv run -m gpt2.finetuning_instruction.main train --model 124M --batch-size 2 --epochs 1 --context-length 256
```

Unlike the random-initialization demo, this loads pretrained OpenAI GPT-2
weights before fine-tuning. Missing weights are downloaded on the first run
and cached under `gpt2\save_load_model\gpt2_models`; existing cached files are
reused without a network request. TensorFlow reads the original checkpoint;
training itself uses PyTorch.

CUDA is used when available; otherwise training runs on CPU and may be slow.
Larger models need substantially more memory. Sequences longer than
`--context-length` are truncated, which can remove part of the prompt or answer.

The original larger configuration is also available:

```powershell
uv run -m gpt2.finetuning_instruction.main train --model 355M --batch-size 8 --epochs 2
```

Useful training options:

| Option | Default / purpose |
|---|---|
| `--model` | `355M`; accepts `124M`, `355M`, `774M`, or `1558M`. |
| `--epochs`, `--batch-size` | `2` epochs and batches of `8`. |
| `--context-length` | `1024`; must be between `1` and `1024`. |
| `--device` | `auto`; accepts `cpu` or `cuda` to choose explicitly. |
| `--data` | Use a different instruction JSON file. |
| `--models-dir` | Override the pretrained-weight cache directory. |
| `--output-dir` | Override the output directory, for example `outputs\my-run`. |
| `--max-new-tokens` | `256` generated tokens per test response. |
| `--num-workers` | `0`; data-loader workers, with CPU collation for Windows compatibility. |

A custom dataset must be a non-empty JSON list whose entries have string
`instruction`, `input`, and `output` fields. Use an empty string for absent input.
Training requires at least 10 entries and a batch size no larger than the
training split. The final incomplete training batch is dropped; validation and
test batches are retained.

Training prints initial losses, evaluates periodically, and generates a sample
after each epoch. It then saves the model weights, a loss plot, and responses
for the test split under `outputs\gpt2\instruction` by default:

```text
outputs/
  gpt2/
    instruction/
      gpt2-124M-sft.pth
      losses.png
      instruction-data-with-response.json
```

The checkpoint filename reflects the selected model size. It contains the
model's PyTorch state dictionary, not optimizer state for resuming training.
The loss plot is saved without opening an interactive window. Downloaded
weights and default outputs are excluded from Git. Reruns overwrite output
files; use a different `--output-dir` to retain previous runs.

Automatic response scoring is not implemented yet;
[_4_evaluationandscoring.py](gpt2/finetuning_instruction/_4_evaluationandscoring.py)
remains a placeholder. To see all available training arguments:

```powershell
uv run -m gpt2.finetuning_instruction.main train --help
```

### Run tests and checks

```powershell
uv run python -m pytest -q
uv run ruff check gpt2\model\main.py gpt2\finetuning_instruction gpt2\pretraining_eval gpt2\save_load_model tests
uv pip check
```

The tests use small models, mock tokenizers, and synthetic checkpoint weights.
They cover import safety, data preparation, command routing, real optimizer
updates, saved outputs, and download failures without downloading model weights
or tokenizer data.