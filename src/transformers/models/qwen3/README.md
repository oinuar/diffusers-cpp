# Qwen3 Causal Language Model (`transformers/qwen3/`)

C++ port of the Hugging Face diffusers Qwen3 causal language model (text encoder).

## Overview

- **Source**: Ported from [Hugging Face diffusers](https://github.com/huggingface/diffusers) — the Python implementation of the Qwen3 decoder-only transformer architecture for causal language modeling.
- **Status**: Base model and causal LM head implemented.

## C++ Design Decisions vs. Python Reference

### 1. Explicit computation graph context

Python diffusers uses implicit PyTorch autograd context; C++ requires passing `ggml_context* ctx` explicitly to every `forward()`. This is because ggml's graph nodes are non-owning and tied to a context lifetime — the caller must manage arena allocation manually.

### 2. KV cache as a dedicated class

Python diffusers represents KV cache as `List[Tuple[Tensor, Tensor]]` (a list of tuples). C++ wraps this in a `Qwen3Cache` class with `update()` and `get()` methods. This gives type safety, centralizes sequence-length tracking (`get_seq_length()`), and makes the cache object's lifetime explicit in the forward() signature rather than hiding it as a mutable kwarg.

### 3. Config struct instead of kwargs

Python uses keyword arguments on model initialization; C++ has a flat `Qwen3Config` struct passed to constructors. This provides compile-time validation of required parameters, enables default values with documentation, and makes the config serializable/deserializable without relying on Python's dynamic dispatch.

### 4. Rotary embeddings as a separate module

In some Python implementations rotary embeddings are computed inline inside attention; C++ extracts them into `Qwen3RotaryEmbedding` as a standalone `Module`. This mirrors the nn.Module philosophy (everything is a composable component) and allows the rotary embedding weights to be loaded via the visitor pattern like any other parameter.

### 5. Position embeddings passed as pre-computed pair

The attention forward() takes `position_embeddings` as `std::pair<Tensor, Tensor>` (cos/sin) rather than raw position_ids. The caller computes them via `Qwen3RotaryEmbedding` before calling attention. This separates the embedding computation from the attention kernel and keeps attention focused on its core responsibility.
