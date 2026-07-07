# Normalization Layers (`normalization/`)

Normalization layer implementations for neural network inference.

## Overview

This module provides different normalization techniques used to stabilize and accelerate training and inference in deep neural networks. All implementations are inference-only — no training mode, dropout, or gradient tracking is included.

## Components

| File | Class | Purpose |
|------|-------|---------|
| `LayerNorm.hpp` | `LayerNorm` | Standard layer normalization that normalizes across the feature dimension using computed mean and variance, with optional learnable scale and shift parameters |
| `RMSNorm.hpp` | `RMSNorm` | Root Mean Square Layer Normalization — a simplified variant that normalizes by RMS only, without mean centering. Faster than standard LayerNorm with comparable results in many architectures |
| `AdaLayerNormContinuous.hpp` | `AdaLayerNormContinuous` | Adaptive Layer Normalization that conditions its scale and shift parameters on an external embedding vector (e.g., timestep embedding), enabling dynamic normalization behavior per input |

## C++ Design Notes

- **Inference-only**: No training mode or dropout handling — all layers compute their normalized output directly. This matches the project's focus on inference performance and eliminates conditional branches for training/inference mode.
- **Parameterized scale/shift**: Standard LayerNorm and RMSNorm use learnable parameters loaded from GGUF files; AdaLayerNormContinuous computes its modulation parameters dynamically from an input embedding tensor at forward time.
