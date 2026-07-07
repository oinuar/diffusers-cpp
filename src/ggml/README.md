# GGML Integration Layer (`ggml/`)

C++ wrapper around the [ggml](https://github.com/ggerganov/ggml) C library, providing a PyTorch-like tensor API with arena allocation and graph-based execution.

## Why This Layer Exists

ggml is a C library for tensor computation with arena allocation and graph-based execution. This C++ layer wraps it to provide:
- A PyTorch-like tensor API (reshape, permute, arithmetic ops) without Python or PyTorch dependencies
- Type-safe memory management through RAII wrappers around ggml contexts and backends
- A computation graph workflow that mirrors PyTorch's lazy evaluation model

## Core Abstractions — What They Are and Why

| Abstraction | Purpose | Why it exists |
|-------------|---------|---------------|
| **Context** | Arena for all tensors and graph nodes | Scope-based lifetime management — everything allocated in a Context is freed when the Context is destroyed |
| **Backend** | Device abstraction (CPU, CUDA, Metal) via `ggml_backend_t` | Abstracts device initialization and tensor transfer without exposing ggml's C API directly |
| **Scheduler** | Builds execution plans from operation graphs; coordinates multiple backends | Decouples graph building from execution order; enables multi-device dispatch without the caller managing dependencies |
| **Graph** | Scheduled computation plan ready for execution (returned by `scheduler.plan()`) | Separates planning from running; the same graph can be executed with different inputs |

## Operation Chains and Execution Flow

Every tensor operation (arithmetic, reshape, permute, slicing, activation functions) does not compute — it **appends a node to an operation chain**. These chains form a directed acyclic graph (DAG) that the Scheduler converts into an executable plan. The workflow has two distinct phases:

### Phase 1: Build (graph construction)

1. Initialize ggml (`ggml_time_init()`, `ggml_log_set()`, `ggml_backend_load_all()`)
2. Create a Backend and Scheduler
3. Build tensors via factory methods (`Tensor::zeros(ctx, shape)`, `Tensor::empty(ctx, shape, type)`) or operations (`a + b`, `.reshape(...)`, `.sin()`)
4. Chain operations — each call returns a new Tensor wrapping a DAG node; no computation happens

### Phase 2: Plan and Execute

5. Call `scheduler.plan(buildable)` — passes ownership of the build logic to the scheduler, which calls through to whatever object provides `build(Context&) → Tensor` (the output tensor). The scheduler walks the DAG from output back to inputs, topologically sorts nodes, assigns them to backends, and returns a Graph.
6. Create `Computation computation(graph)` — wraps the planned graph for execution
7. Load input tensors: `computation.load(tensor, data_ptr)` — copies input data into backend memory
8. Execute: `computation.execute()` — runs all nodes in topological order on the appropriate backends
9. Read output: `auto [shape, data] = computation.read<float>()` — reads result from the target tensor

### Complete Minimal Pattern (from test/utils/tensor-cli/main.cpp)

```cpp
// 1-2: Setup
ggml_time_init();
ggml_log_set(log_callback, nullptr);
ggml_backend_load_all();

Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
Scheduler scheduler({*cpu});

// 3-4: Build (inside a class that provides build(Context&) → Tensor)
Tensor output = tensor_cli.build(ctx);  // builds DAG, returns output tensor

// 5: Plan
auto graph = scheduler.plan(tensor_cli);

// 6-9: Execute
Computation computation(graph);
for (auto& [tensor, data] : inputs_)
    computation.load(tensor, reinterpret_cast<std::byte*>(data.data()));
computation.execute();
auto [shape, data] = computation.read<float>();
```

See [test/utils/tensor-cli/main.cpp](../../test/utils/tensor-cli/main.cpp) for a complete working example covering all tensor operations.

## Key Design Decisions

### Non-owning Tensors

Tensors reference `ggml_tensor` pointers valid only while the parent Context lives. Callers must ensure the Context outlives any Tensor derived from it. No copy overhead — tensors are thin wrappers around graph nodes.

### Explicit Context Propagation

Tensor factory methods take `Context&` as first argument (e.g., `Tensor::zeros(ctx, shape)`). This makes arena ownership explicit and prevents cross-context references.

### Lazy Evaluation

No operation computes until `Computation::execute()`. Operations return new Tensors wrapping DAG nodes; the Scheduler converts the full chain into a topologically sorted execution plan. This enables building complex graphs incrementally before committing to a single scheduled pass.

## GGUF Weight Loading

`GGUFLoaderVisitor` traverses Module trees to load weights from GGUF files into Parameter tensors. For the visitor pattern itself, see [src/nn/README.md](../nn/).
