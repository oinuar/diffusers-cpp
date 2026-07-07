# Embedding Utilities (`embeddings/`)

Timestep encoding utilities and position embedding functions for diffusion models.

## Overview

This module provides building blocks for converting scalar timestep values and position indices into learned or sinusoidal embeddings that condition neural network layers. These are core components of diffusion model architectures where timestep information modulates the denoising process.

## Components

| File | Class/Function | Purpose |
|------|---------------|---------|
| `Timesteps.hpp` | Timestep functions | Convert scalar noise scheduler timesteps into embedding-compatible tensor representations |
| `TimestepEmbedding.hpp` | `TimestepEmbedding` | Project timestep embeddings through a learned linear transformation with bias, producing conditioned embedding vectors for modulation layers |
| `funcs.hpp` | Helper functions | Utility functions for embedding computation used across the module |

## C++ Design Notes

- **No training mode**: All implementations are inference-only — no gradient tracking or parameter updates. Embedding weights are loaded from GGUF files and used as static lookups or projections during inference.
- **Explicit tensor types**: Timestep and position embeddings operate on `Tensor` objects from the ggml layer, returning computed embedding tensors ready to be consumed by normalization or modulation layers.
