# Flux2 Transformer Pipeline (`diffusers/transformers/flux2/`)

C++ port of the Hugging Face diffusers Flux.2 (Klein 9B) diffusion transformer pipeline.

## Overview

- **Source**: Ported from [Hugging Face diffusers](https://github.com/huggingface/diffusers) — the Python implementation of the Flux.2 (Klein 9B) transformer architecture for image generation.
- **Status**: Transformer block ported; text encoder and VAE not yet implemented.

## C++ Design Decisions vs. Python Reference

### 1. Explicit computation graph context

Python diffusers uses implicit PyTorch autograd context; C++ requires passing `ggml_context* ctx` explicitly to every `forward()`. This is because ggml's graph nodes are non-owning and tied to a context lifetime — the caller must manage arena allocation manually.

### 2. Compile-time attention dispatch

Python uses runtime method dispatch or function arguments for attention backends; C++ uses template instantiation (`Flux2AttnProcessor<SoftmaxAttnOp>`). This eliminates virtual call overhead and lets the compiler inline attention kernels, at the cost of longer compile times. The attention operation is a compile-time type parameter, not a runtime configuration.

### 3. Non-owning tensor abstraction

Python Tensors own their data; C++ `Tensor` is a thin wrapper around `ggml_tensor*` with no ownership. Tensors are valid only while the parent `ggml_context` lives. This mirrors ggml's arena allocation model but means lifetime management is the caller's responsibility, not the tensor's.

### 4. Modular modulation pattern

Python diffusers applies timestep modulation inline; C++ extracts it into a dedicated `Flux2Modulation` utility with explicit shift/scale/gate splitting. This keeps TransformerBlock forward() readable and reuses the same pattern in SingleTransformerBlock, following the modular design philosophy of the nn/ layer.

### 5. Manual f16 clipping

Python relies on PyTorch's internal overflow handling; C++ applies explicit `clip(-65504, 65504)` on f16 outputs because ggml's quantized backends can produce subnormal values that need capping before storage. This is a defensive measure specific to the ggml execution model.
