# Custom Models

## Description

Custom implementations of transformer models for learning how their building
blocks and text generation work. The repository currently includes a PyTorch
GPT-2 implementation covering attention, layer normalization, feed-forward
networks, transformer blocks, and greedy and sampling-based generation.
A separate folder is reserved for Qwen3, with more models to be added over time.

## Folder organization

Each model lives in its own top-level folder. All models share one
[uv](https://docs.astral.sh/uv/) project and dependency environment; there is no
shared Python package that model folders must be placed inside.

- [gpt2/](gpt2/) - GPT-2 implementation and demos.
  - Numbered Python files introduce concepts and implement the model components.
  - [_12_gpt2model.py](gpt2/_12_gpt2model.py) contains the model configuration,
    model class, and text-generation helpers.
  - [main.py](gpt2/main.py) runs the text-generation demo.
- [qwen3/](qwen3/) - Placeholder for a future Qwen3 implementation;
  [main.py](qwen3/main.py) is currently empty.
- [tests/](tests/) - Automated tests for the GPT-2 demo.
- [pyproject.toml](pyproject.toml) - Project metadata and shared dependencies.
- [uv.lock](uv.lock) - Resolved dependency versions.
- [.python-version](.python-version) - Python version used by uv (3.11).

## Run locally

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Open a terminal in the repository root.
3. Install the dependencies and run the GPT-2 demo:

   ```powershell
   uv sync
   uv run -m gpt2.main
   ```

uv manages Python 3.11 and the project's virtual environment. Run model entry
points with `-m` from the repository root so package imports resolve correctly,
rather than executing `gpt2\main.py` directly.

The demo starts with `"Every effort moves you"` and generates two identical
greedy continuations of 10 tokens, followed by a 15-token sampled continuation.
The model is randomly initialized, not pretrained, so incoherent output is
expected. The first run requires internet access to download GPT-2 tokenizer
data, which tiktoken caches locally.

To run the tests and checks:

```powershell
uv run python -m pytest -q
uv run ruff check gpt2\main.py tests\test_main.py
uv pip check
```

The tests use a small model and a mock tokenizer, so they do not download
tokenizer data or instantiate the full-size model.