# Localizer

This is the actual localizer, a usage example can be seen in [here](src/main.py)(but it likely wont work due to missing secrets) or in the [CLI](../cli/README.md)

## Install
```bash
uv sync
```

This default install is CPU-compatible.

## Optional: CUDA PyTorch
If you want CUDA-enabled PyTorch, install the matching CUDA wheel after syncing dependencies.

Example for CUDA 12.8:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchvision
```

If you use another CUDA version, swap `cu128` for the matching index from the PyTorch install matrix.
