# Attention Operations (`attention/`)

Pluggable attention operation backends for use with transformer-based models.

## Overview

This module provides different attention computation strategies implemented as pluggable operation types. They are designed to be used as template parameters on attention processors, enabling compile-time selection of the attention algorithm without runtime dispatch overhead.

## Components

| File | Class | Purpose |
|------|-------|---------|
| `FlashAttnOp.hpp/cpp` | `FlashAttnOp` | FlashAttention-based operation for memory-efficient attention computation with reduced KV cache bandwidth usage |
| `SoftmaxAttnOp.hpp/cpp` | `SoftmaxAttnOp` | Standard softmax-based attention operation using pairwise dot-product similarity |

## C++ Design Notes

- **Template-based dispatch**: Attention operations are selected at compile time via template instantiation (e.g., `AttnProcessor<FlashAttnOp>`) rather than runtime polymorphism. This eliminates virtual call overhead and enables compiler optimization across the attention kernel boundary.
- **Uniform interface**: Both implementations expose the same interface signature, allowing them to be swapped transparently by changing a single template parameter without modifying the processor that uses them.
- **No training mode**: These operations are inference-only implementations — no gradient tracking or dropout handling is included, matching the project's focus on inference performance.
