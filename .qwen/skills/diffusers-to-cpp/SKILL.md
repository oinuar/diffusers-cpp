---
name: diffusers-to-cpp
description: Step-by-step workflow for porting Python Hugging Face diffusers pipeline classes to C++ with GGML backend. Covers Module hierarchy, Tensor operations and factory methods. Use when the user wants to port a Python diffusers class (Transformer, Attention, Embedding, Normalization, etc.) to C++, mentions adding new model components, or asks how to translate PyTorch/diffusers code.
---

# Porting Diffusers Classes to C++

This skill guides you through porting Python `diffusers` library classes to the C++ / GGML project. Follow the workflow steps in order; each step builds on the previous one.

---

## Step 1 — Analyze the Python Class

Open the Python source (e.g., from `diffusers` or `transformers`) and identify:

### 1a. Base class inheritance
```python
class MyModule(nn.Module):  # ← inherits from nn.Module
```
**C++ equivalent**: Inherit from `Module`.

### 1b. Constructor (`__init__`) — register sub-modules and parameters
List every line in `__init__` that creates a layer:

| Python | What it means |
|--------|---------------|
| `self.proj = nn.Linear(in, out)` | Registered sub-module with learnable weights (weight tensor + optional bias tensor) |
| `self.norm = LayerNorm(dim)` | Registered sub-module, may have internal parameters (`weight`, `bias`) |
| `self.act = SiLU()` | Registered sub-module with **no** learnable parameters (activation) |
| `self.scale = nn.Parameter(torch.tensor(...))` | A **Parameter** leaf — a tensor loaded from weights file, not created at init time |

### 1c. The `forward()` method
Identify:
- Input tensors and their shapes/dtypes
- All tensor operations (arithmetic, slicing, reshape, permute, cat, split)
- All submodule calls (`self.proj(x)`)
- Output shape and type

---

## Step 2 — Create the C++ Header

Create `src/models/<category>/<ClassName>.hpp` in the appropriate subdirectory:

```cpp
#pragma once

#include "nn/Module.hpp"       // base class — always required
#include "nn/Linear.hpp"       // any submodule you use
#include "models/embeddings/RotaryEmbedding.hpp"  // other submodules
#include "ggml/Tensor.hpp"               // Tensor type (re-exported via Module.hpp)

class ClassName : public Module {
public:
    // Constructor parameters mirror Python __init__ arguments.
    // Keep the same names and types where practical.
    ClassName(
        int64_t in_channels,
        int64_t out_channels,
        bool bias = true,   // match Python defaults exactly
        float some_param = 1e-5f
    );

    // forward() takes ggml_context* as the first argument — this is the graph context.
    // All computation happens within this context.
    Tensor forward(ggml_context* ctx, Tensor x);

private:
    // Store non-tensor configuration values (bools, ints, floats, strings)
    bool bias_;
    float some_param_;

    // Registered sub-modules — stored as shared_ptr<Module> in modules[] map.
    // Declare them as the concrete type (not Module) so you can static_pointer_cast later.
};
```

### Header conventions:
- **Include guards**: Use `#pragma once` (project standard)
- **Include order**: Project headers first (`modules/...`, `models/...`), then third-party, then stdlib
- **Constructor defaults**: Must match Python defaults exactly
- **No inline implementation** in the header (keep it clean for code review)

---

## Step 3 — Implement the C++ Source

Create `src/models/<category>/<ClassName>.cpp`:

### 3a. Constructor — register sub-modules into `modules[]`
```cpp
ClassName::ClassName(
    int64_t in_channels,
    int64_t out_channels,
    bool bias,
    float some_param)
{
    // Register each sub-module. The key string must match GGUF weight naming.

    // Python: self.proj = nn.Linear(in_channels, inner_dim * 3)
    modules["proj"] = std::make_shared<Linear>(in_channels, inner_dim * 3, bias);

    // Python: self.norm = LayerNorm(inner_dim)
    modules["norm"] = std::make_shared<LayerNorm>(inner_dim);

    // Python: self.out.0 = nn.Linear(inner_dim, out_channels)
    // Note: diffusers uses dot-notation for sequential containers (nn.Sequential / nn.ModuleList)
    modules["out.0"] = std::make_shared<Linear>(inner_dim, out_channels);

    // Python: self.out.1 = some_activation()
    modules["out.1"] = std::make_shared<SiLU>();

    bias_ = bias;
    some_param_ = some_param;
}
```

### 3b. `forward()` — translate tensor operations one-to-one

```cpp
Tensor ClassName::forward(ggml_context* ctx, Tensor x) {
    // 1. Retrieve sub-modules and call forward() on them.
    auto proj = std::static_pointer_cast<Linear>(modules["proj"]);
    auto norm = std::static_pointer_cast<LayerNorm>(modules["norm"]);
    auto out0 = std::static_pointer_cast<Linear>(modules["out.0"]);
    auto out1 = std::static_pointer_cast<SiLU>(modules["out.1"]);

    // 2. Chain operations, translating Python → C++ line by line:

    // Python: hidden_states = self.proj(hidden_states)
    auto hidden_states = proj->forward(ctx, x);

    // Python: hidden_states = self.norm(hidden_states)
    hidden_states = norm->forward(ctx, hidden_states);

    // Python: return self.out.1(self.out.0(hidden_states))
    auto step1 = out0->forward(ctx, hidden_states);
    auto output  = out1->forward(ctx, step1);

    // 3. Handle residual connections:
    // Python: return hidden_states + output
    output = x + output;   // operator overload works directly

    return output;
}
```

### Common forward() patterns (Python → C++):

| Python pattern | C++ equivalent | Notes |
|---------------|----------------|-------|
| `self.linear(x)` | `auto l = std::static_pointer_cast<Linear>(modules["linear"]); l->forward(ctx, x);` | Always cast to concrete type first |
| `self.module_list[i](x)` | Same as above with key `"i"` or use a separate vector member if indexed numerically |
| `torch.cat([a, b], dim=1)` | `Tensor::cat({a, b}, 1)` | Static method on Tensor class |
| `x.shape[1]` | `x.shape()[1]` | `.shape()` returns a Tensor::Shape object (indexable) |
| `x.dtype` (to cast) | `x.dtype()` → pass to `.to()` or use as arg | Returns ggml_type enum value |
| `x.unflatten(-1, (heads, -1))` | `x.unflatten(-1, {heads, -1})` | Tuple → initializer_list in C++ |
| `x.flatten(2, 3)` | `x.flatten(2, 3)` | Direct method call — same signature |
| `x.to(query.dtype)` | `x.to(query.dtype())` | Cast to match another tensor's dtype |
| `a + b * scale` | `b * scale + a` or use intermediate var | C++ operator precedence: `*` before `+`, same as Python/PyTorch |
| ModuleList iteration (Python) | Same loop but call forward on each element via dynamic_cast or store concrete types |

### Slicing patterns (Python → C++):

```cpp
// Python: x[:, 2, :]          →  x[{Tensor::Slice::all(), Tensor::Slice::index(2), Tensor::Slice::all()}]
// Python: x[..., 0:10:2]      →  x[{Tensor::Slice::ellipsis(), Tensor::Slice::range(0, 10, 2)}]
// Python: x[None, ...]        →  x[{Tensor::Slice::none(), Tensor::Slice::ellipsis()}]
// Python: x[:, :seq_len]      →  x[{Tensor::Slice::all(), Tensor::Slice::range(std::nullopt, seq_len)}]
// Python: x[:, seq_len:]      →  x[{Tensor::Slice::all(), Tensor::Slice::range(seq_len, std::nullopt)}]
// Python: x[:, None, :]       →  x[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}]
```

## Step 4 — Use in a Parent Module or Pipeline

When another module needs to use your new class:

### 5a. Register it as a sub-module
```cpp
// In parent's constructor:
modules["attn"] = std::make_shared<ClassName>(in_channels, out_channels, bias);
```

### 5b. Call it from `forward()`
```cpp
auto attn = std::static_pointer_cast<ClassName>(modules["attn"]);
x = attn->forward(ctx, x);
```

### 5c. Factory method pattern (for top-level pipelines)

Top-level models expose a static `from_pretrained()` method:

```cpp
class ClassName : public Module {
public:
    // Factory method — creates the module tree and populates from GGUF.
    static std::shared_ptr<ClassName> from_pretrained(
        ggml_backend_t backend,   // device backend (CPU, CUDA, etc.)
        const std::string& path   // path to GGUF file
    );

private:
    ClassName(...);  // private or protected — use factory instead
};

ClassName ClassName::from_pretrained(Backend& loader_backend, const std::string& path) {

    // NOTE: model might have been constructed with different parameters than source code defaults.
    // Always verify this by checking model config and set parameters here that match the config file.
    ClassName model;

    GGUFLoaderVisitor loader(loader_backend, path);
    model.accept(loader);

    return std::move(model);
}

std::shared_ptr<ClassName> ClassName::from_pretrained(
    ggml_backend_t backend, const std::string& path) {

    auto model = std::make_shared<ClassName>(/* constructor args */);

    // Build the ggml context for weight loading
    auto ctx = gguf_init_from_file(path.c_str(), /* params */);

    // GGUFLoaderVisitor traverses the module tree and loads weights
    GGUFLoaderVisitor loader(ctx, backend, path);
    model->accept(loader);

    return model;
}
```

---

## Step 5 — Verify Correctness

The simplest and most reliable verification: compile a small test program that constructs your C++ module tree, then runs `GGUFLoaderVisitor` on the GGUF file you already produced with `convert_diffusers_safetensors_to_gguf.py`. If loading succeeds, the structure matches. If it fails, the error tells you exactly what's wrong.

```cpp
#include "models/<category>/<TopLevelClassName>.hpp" // <-- your top-level class that includes whole module tree
#include "Backend.hpp"
#include "Scheduler.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>" << std::endl;
        return 1;
    }

    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);

    TopLevelClassName::from_pretrained(cpu, argv[1]);

    return 0;
}
```

**Run it:**
```bash
# Convert the original diffusers checkpoint to GGUF first:
python utils/convert-model/convert_diffusers_safetensors_to_gguf.py \
    --input /path/to/diffusers-checkpoint \
    --output model.gguf \
    --architecture stable-diffusion

# Then verify your C++ port loads it:
./test_model_structure model.gguf
```

**If loading fails**, the error message tells you exactly what to fix:

| Error from GGUFLoaderVisitor | Fix |
|------------------------------|-----|
| `"Tensor not found"` | Missing sub-module registration or wrong key name (e.g., `"proj"` instead of `"to_q"`) |
| `"Parameter dimension mismatch: expected 2, got 3"` | Some tensor operator resulted incorrect dimension. Check the code that manipulates the Tensor. |
| `"Parameter shape mismatch: expected (x,y,z), got (a,b,c)"` | Constructor args are wrong — check `in_channels`, `inner_dim`, etc. against the Python model definition |

---

## Quick Reference: File Locations

| Component | Header location | Source location |
|-----------|----------------|-----------------|
| Base Module | `src/modules/Module.hpp` | `src/modules/Module.cpp` |
| Parameter (leaf) | `src/modules/Parameter.hpp` *(inline)* | — |
| Linear layer | `src/modules/Linear.hpp` | `src/modules/Linear.cpp` |
| Activations (SiLU, etc.) | `src/modules/SiLU.hpp` *(inline)* | — |
| New model class (example) | `src/models/attention/NewClass.hpp` | `src/models/attention/NewClass.cpp` |
| Tensor abstraction | `src/Tensor.hpp` / `Tensor.cpp` | — |

## Quick Reference: Key Headers to Include

```cpp
#include "nn/Module.hpp"       // base class — every module needs this
#include "nn/Parameter.hpp"    // leaf parameter nodes
#include "nn/Linear.hpp"       // dense layers
#include "nn/SiLU.hpp"         // SiLU activation (SwiGLU)
#include "nn/Dropout.hpp"      // dropout layers
#include "ggml/Tensor.hpp"               // Tensor type and operations

// Model-specific headers (use relative paths from src/):
#include "models/normalization/LayerNorm.hpp"
#include "models/normalization/RMSNorm.hpp"
#include "models/embeddings/RotaryEmbedding.hpp"
```
