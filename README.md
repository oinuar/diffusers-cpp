# Diffusers Pipelines in C++

A standalone C++ implementation of [Hugging Face diffusers](https://github.com/huggingface/diffusers) pipelines, powered by [ggml](https://github.com/ggerganov/ggml) for efficient inference on CPU and GPU.

This project ports selected Python diffusers pipelines to C++, providing a drop-in replacement that runs locally without Python dependencies. Models are stored in GGUF format for efficient loading and minimal memory footprint.

## Goals

- **Standalone**: No Python, no parent project dependency — only GGML as a git submodule.
- **Fidelity**: Match the Python diffusers pipeline API and behavior as closely as possible.
- **Performance**: Leverage ggml's optimized backends (CPU, CUDA, ROCm, Metal, Vulkan) for fast inference.
- **Modular**: Mirror the diffusers architecture — `Module` hierarchy with visitor-based weight loading.

## Implemented Pipelines

| Pipeline | Status | Components |
|----------|--------|------|
| [Flux2KleinPipeline](src/diffusers/pipelines/flux2/Flux2KleinPipeline.hpp) | Ported | [Flux2Transformer2DModel](src/diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp), [Qwen3ForCausalLM](src/transformers/models/qwen3/Qwen3ForCausalLM.hpp), [AutoencoderKLFlux2](src/diffusers/models/autoencoders/AutoencoderKLFlux2.hpp), [Qwen2TokenizerFast](src/transformers/models/qwen2/Qwen2TokenizerFast.hpp) |
| Ideogram 4.0 | Planned | — |
| Wan 2.2 | Planned | — |

## Build Instructions

### Prerequisites

- CMake 3.23+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 19.20+)
- [docker](https://www.docker.com/get-started/) (for build sandbox)
- [uv](https://docs.astral.sh/uv/) (for running tests)

### Clone & Build

```bash
git clone https://github.com/oinuar/diffusers-cpp
cd diffusers-cpp
git submodule update --init --recursive

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCARGO_EXECUTABLE="$PWD/../utils/cargo.sh" -DDIFFUSERS_CUDA=ON   # or OFF for CPU-only
cmake --build . --config Release -j$(nproc)
```

### GPU Backend Options

Enable GPU backends via `DIFFUSERS_*` CMake flags at configure time.

| Flag | Backend | Platform |
|------|---------|----------|
| `-DDIFFUSERS_CUDA=ON` | NVIDIA CUDA (cuBLAS + mmq) | Linux, Windows |
| `-DDIFFUSERS_METAL=ON` | Apple Metal (BLAS + compute) | macOS, iOS *(default on Apple)* |
| `-DDIFFUSERS_HIP=ON` | AMD ROCm/HIP (rocBLAS) | Linux, Windows *(via WSL not supported)* |
| `-DDIFFUSERS_VULKAN=ON` | Vulkan (cross-platform GPU) | Linux, Windows, macOS |
| `-DDIFFUSERS_OPENCL=ON` | OpenCL 1.2/2.x fallback | Cross-platform |
| `-DDIFFUSERS_SYCL=ON` | Intel oneAPI SYCL *(experimental)* | Linux, Windows *(requires Intel compiler or DPC++)* |
| `-DDIFFUSERS_MUSA=ON` | MOFUSE MUSA (Moore Threads GPU) | Linux *(requires MUSA toolkit)* |

### Running Tests

The project ships with tests:

```bash
# After building, run tests via CTest:
ctest --progress --output-on-failure
```

## Status

This project is **work in progress**. Core infrastructure (Tensor abstraction, Module system, GGUF loading) and draft pipeline implementations are available. See the pipeline table above for individual component status.
