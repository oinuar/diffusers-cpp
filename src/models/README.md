# Model Implementations (`models/`)

This directory contains model pipeline implementations and shared neural network utilities. Models are organized into two categories: pipeline ports from Hugging Face repositories, and reusable utility modules used by those pipelines.

## Pipeline Index

| Pipeline | Directory | Status | Docs |
|----------|-----------|--------|------|
| Flux2 (Flux.2 / Klein 9B) | `diffusers/transformers/flux2/` | Transformer ported; text encoder and VAE missing | [flux2/](diffusers/transformers/flux2/) |
| Qwen3 (Causal LM) | `transformers/qwen3/` | Base model and causal LM head implemented | [qwen3/](transformers/qwen3/) |

## Shared Utilities

These modules provide reusable neural network components used across multiple pipelines:

| Directory | Description | Docs |
|-----------|-------------|------|
| `attention/` | Pluggable attention operation backends (FlashAttention, Softmax) with compile-time template dispatch | [attention/](attention/) |
| `embeddings/` | Timestep encoding utilities and position embedding functions for diffusion models | [embeddings/](embeddings/) |
| `normalization/` | Normalization layer implementations (LayerNorm, RMSNorm, AdaLayerNormContinuous) in inference-only mode | [normalization/](normalization/) |

## Adding a New Model Pipeline

To add a new model port from Python diffusers:

1. **Create the model class** inheriting `nn::Module` in an appropriate subdirectory (e.g., `models/diffusers/transformers/<model>/`).
2. **Add sub-modules via the children map** in the constructor using `modules["name"] = std::make_shared<...>(args...)`.
3. **Implement `forward(ggml_context* ctx, ...)`** following the convention of explicit context and Tensor return type. See [src/nn/README.md](../nn/) for detailed porting steps.
4. **Use shared utilities** from `attention/`, `embeddings/`, and `normalization/` where applicable — these are model-agnostic and reusable across pipelines.
5. **Weight loading works automatically** — the model's `accept()` method is called by the GGUF loader visitor, which traverses all Parameters. See [src/ggml/README.md](../ggml/) for tensor and backend details.
