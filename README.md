# Diffusers Pipelines in C++ / GGML

A standalone C++ implementation of [Hugging Face diffusers](https://github.com/huggingface/diffusers) pipelines, powered by [ggml](https://github.com/ggerganov/ggml) for efficient inference on CPU and GPU.

This project ports selected Python diffusers pipelines to C++, providing a drop-in replacement that runs locally without Python dependencies. Models are stored in GGUF format for efficient loading and minimal memory footprint.

## Goals

- **Standalone**: No Python, no parent project dependency — only GGML as a git submodule.
- **Fidelity**: Match the Python diffusers pipeline API and behavior as closely as possible.
- **Performance**: Leverage ggml's optimized backends (CPU, CUDA, ROCm, Metal, Vulkan) for fast inference.
- **Modular**: Mirror the diffusers architecture — `Module` hierarchy with visitor-based weight loading.

## Implemented Pipelines

| Pipeline | Status | Docs |
|----------|--------|------|
| Flux2 (Flux.2 / Klein 9B) | Transformer ported; text encoder and VAE missing | [models/diffusers/transformers/flux2/](src/models/diffusers/transformers/flux2/) |
| Qwen3 (Causal LM) | Base model and causal LM head implemented | [models/transformers/qwen3/](src/models/transformers/qwen3/) |
| Ideogram 4.0 | Planned | — |

## Build Instructions

### Prerequisites

- CMake 3.12+
- C++17 compiler (GCC, Clang, MSVC)
- [ggml](https://github.com/ggerganov/ggml) — cloned as a git submodule

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

Models must be in GGUF format, converted from diffusers-style safetensors checkpoints using the conversion utility in `utils/convert-model/`.

## Project Structure

```
diffusers-cpp/
├── src/
│   ├── nn/                     # Neural network primitives (Module system, Layer classes)
│   ├── models/                 # Model implementations and shared utilities
│   │   ├── diffusers/          # Hugging Face diffusers pipeline ports (Flux2, etc.)
│   │   ├── transformers/       # Transformer model ports (Qwen3, etc.)
│   │   ├── attention/          # Pluggable attention operation backends
│   │   ├── embeddings/         # Timestep and position embedding utilities
│   │   └── normalization/      # Normalization layer implementations
│   ├── ggml/                   # GGML C library wrapper (Tensor, Backend, Scheduler)
│   └── main.cpp                # CLI entry point
├── utils/convert-model/        # Conversion utility: diffusers safetensors → GGUF
├── diffusers/                  # Reference: Hugging Face diffusers Python repo (for porting)
├── transformers/               # Reference: Hugging Face transformers Python repo (for porting)
└── CMakeLists.txt              # Build configuration
```

## Status

This project is **work in progress**. Core infrastructure (Tensor abstraction, Module system, GGUF loading) and draft pipeline implementations are available. See the pipeline table above for individual component status.
