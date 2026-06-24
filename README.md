# Diffusers Pipelines in C++ / GGML

A standalone C++ implementation of [Hugging Face diffusers](https://github.com/huggingface/diffusers) pipelines, powered by [GGML](https://github.com/ggerganov/ggml) for efficient inference on CPU and GPU.

This project port selected Python diffusers pipelines to C++, providing a drop-in replacement that runs locally without Python dependencies. Models are stored in GGUF format for efficient loading and minimal memory footprint.

## Goals

- **Standalone**: No Python, no parent project dependency — only GGML as a git submodule.
- **Fidelity**: Match the Python diffusers pipeline API and behavior as closely as possible.
- **Performance**: Leverage GGML's optimized backends (CPU, CUDA, ROCm, Metal, Vulkan) for fast inference.
- **Modular**: Mirror the diffusers architecture — `Module` hierarchy with visitor-based weight loading.

## Implemented Pipelines

| Pipeline | Status | 
|----------|--------|
| Flux2 (Flux.2 / Klein 9B) | Transformer ported, text encoder and vae missing |
| Ideogram 4.0 | Planned |

## Architecture Overview

### Module System

The project uses a hierarchical module system that mirrors PyTorch's `nn.Module`:

```
Module (base class)
├── children: unordered_map<string, shared_ptr<Module>>
└── accept(Visitor&) → traverses tree in depth-first order

Parameter (leaf module)
├── Holds a Tensor::Shape (declared shape before loading)
├── set(Tensor) → assigns loaded weights
└── forward() → returns the tensor for computation

Linear, SiLU, Dropout, Identity, ... (derived modules)
└── Direct C++ conversions of their Python diffusers counterparts
```

Each `Module` exposes an `accept(Visitor&)` method that enables the **visitor pattern** for traversing the entire module tree. This is used primarily for GGUF weight loading: a `GGUFLoaderVisitor` visits every `Parameter`, reads tensor data from the GGUF file, validates shapes, and assigns weights.

### Tensor Abstraction

`Tensor` is a C++ wrapper around `ggml_tensor*`. It provides a PyTorch-like API (`reshape`, `permute`, operations like `+`, `-`, `*`, `exp`, etc.) that constructs ggml graph nodes without executing computation. Actual computation happens when the caller builds and executes a ggml graph via the backend scheduler.

Key design points:
- **Non-owning**: Tensor does not own its underlying `ggml_context` or `ggml_tensor`. It is valid only while the context remains alive.
- **Graph-building**: Operations return new Tensor objects wrapping graph nodes; no computation occurs at construction time.
- **Shape class**: A fixed-size 4-dimension shape (`Tensor::Shape`) with numpy-like indexing and string formatting.

### GGML Backend Integration

`GGMLBackend` wraps `ggml_backend_t` for device initialization (CPU, CUDA, Metal, etc.). It manages backend lifecycle and provides the execution context for building and running computation graphs.

### GGUF Loading

Weights are loaded from GGUF files via `GGUFLoaderVisitor`. The loader:
1. Opens the GGUF file and builds a name→tensor-index lookup table.
2. Allocates all tensors into a ggml context bound to the target backend.
3. Visits every `Parameter` in the module tree, validates dimension count and shape against the GGUF tensor, reads raw data from disk, and assigns it via `ggml_backend_tensor_set`.

### Porting Strategy

Each diffusers model is ported as a C++ class hierarchy that mirrors the Python structure:
- Model classes (e.g., `Flux2Transformer2DModel`) expose static factory methods like `from_pretrained(backend, path)`.
- Sub-modules (attention blocks, MLPs, normalization layers) are separate classes in organized directories (`models/transformers/flux2/`, `models/embeddings/`, `models/normalization/`, etc.).
- The model's `accept()` method is called by the GGUF loader to traverse and populate all parameters.

Here is a table of selected examples how Python syntax maps to C++:

| Python | C++ |
|--------|-----|
| `torch.cat([encoder_query, query], dim=1)` | `Tensor::cat({encoder_query, query}, 1);` |
| `.shape[1]` | `.shape()[1]` |
| `.dtype` | `.dtype()` |
| `self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)` | `modules["linear_out"] = std::make_shared<Linear>(inner_dim, dim_out, bias);` |
| `x = self.linear_out(x)` | `auto linear_out = std::static_pointer_cast<Linear>(modules["linear_out"]); x = linear_out->forward(ctx, x);` |
| `self.to_out[0](x)` | `auto to_out0 = std::static_pointer_cast<Linear>(modules["to_out.0"]); to_out0->forward(ctx, x);` |
| `qkv, mlp_hidden_states = torch.split(hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)` | `auto parts = hidden_states.split_with_sizes({3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor}, -1); auto qkv = parts.at(0); auto mlp_hidden_states = parts.at(1);` |
| `query.unflatten(-1, (attn.heads, -1))` | `query.unflatten(-1, {attn.heads, -1});` |
| `hidden_states.flatten(2, 3)` | `hidden_states.flatten(2, 3);` |
| `hidden_states.to(query.dtype)` | `hidden_states.to(query.dtype());` |
| `x[:, None, :]` | `x[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}];` |
| `x[:, 2, :]` | `x[{Tensor::Slice::all(), Tensor::Slice::index(2), Tensor::Slice::all()}];` |
| `x[..., 0:10:2]` | `x[{Tensor::Slice::ellipsis(), Tensor::Slice::range(0, 10, 2)}];` |
| `x[None, ...]` | `x[{Tensor::Slice::none(), Tensor::Slice::ellipsis()}];` |
| `hidden_states[:, :text_seq_len]` | `hidden_states[{Tensor::Slice::all(), Tensor::Slice::range(std::nullopt, text_seq_len)}];` |
| `hidden_states[:, text_seq_len:]` | `hidden_states[{Tensor::Slice::all(), Tensor::Slice::range(text_seq_len, std::nullopt)}];` |
| Arithmetic operations | Operator overloads (use as-is) |


## Build Instructions

### Prerequisites

- CMake 3.12+
- C++17 compiler (GCC, Clang, MSVC)
- [GGML](https://github.com/ggerganov/ggml) — cloned as a git submodule

### Clone & Build

```bash
git clone <repository-url>
cd diffusers-cpp
git submodule update --init   # initializes GGML submodule

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

### GPU Backend Options

Enable GPU backends via CMake flags (passed to the cmake configure step):

| Flag | Backend |
|------|---------|
| `-DSD_CUDA=ON` | NVIDIA CUDA |
| `-DSD_METAL=ON` | Apple Metal (macOS/iOS) |
| `-DSD_HIPBLAS=ON` | AMD ROCm/HIP |
| `-DSD_VULKAN=ON` | Vulkan (cross-platform GPU) |
| `-DSD_OPENCL=ON` | OpenCL |
| `-DSD_SYCL=ON` | Intel SYCL |

### Standalone Build

This project is fully standalone. The only external dependency is GGML (via submodule).

## Usage

Once built, run inference with the compiled CLI tool:

```bash
./flux2-cli --model path/to/model.gguf [options]
```

### Model Format

Models must be in GGUF format, converted from diffusers-style safetensors checkpoints using the provided conversion utility in `utils/convert-model/`.

## Project Structure

```
flux2-cli/
├── src/
│   ├── Tensor.hpp/cpp          # ggml_tensor wrapper (PyTorch-like API)
│   ├── modules/                # Neural network module classes
│   │   ├── Module.hpp/cpp      # Base class with visitor pattern support
│   │   ├── Parameter.hpp       # Leaf module for trainable weights
│   │   ├── Visitor.hpp         # Visitor interface for tree traversal
│   │   ├── Linear.hpp/cpp      # Linear/dense layer
│   │   ├── SiLU.hpp            # Sigmoid-weighted linear unit (SwiGLU)
│   │   ├── Dropout.hpp         # Dropout regularization
│   │   └── Identity.hpp        # Identity activation passthrough
│   ├── models/                 # Model implementations (diffusers ports)
│   │   ├── transformers/flux2/  # Flux2 Transformer 2D model
│   │   ├── attention/            # Attention mechanisms (FlashAttention, etc.)
│   │   ├── embeddings/           # Positional, patch, text embeddings
│   │   └── normalization/        # Layer norm, RMS norm, etc.
│   ├── GGMLBackend.hpp         # ggml_backend_t wrapper for device management
│   ├── GGMLScheduler.hpp       # Computation graph scheduling (planned)
│   ├── GGUFLoaderVisitor.cpp/hpp  # Visitor for loading GGUF weights
│   └── main.cpp                # CLI entry point
├── utils/convert-model/        # Conversion utility: diffusers safetensors → GGUF
└── CMakeLists.txt              # Build configuration
```

## Status

This project is **work in progress**. Core infrastructure (Tensor abstraction, Module system, GGUF loading) and the draft of Flux2 transformer pipeline are implemented.
