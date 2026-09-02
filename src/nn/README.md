# Neural Network Primitives (`nn/`)

C++ implementations of neural network modules mirroring PyTorch's `torch.nn`, built on top of the ggml tensor abstraction layer.

## Module System — C++ Implementation Details

### Module Base Class

The `Module` class is the base for all neural network components. It maintains a children map (`unordered_map<string, shared_ptr<Module>>`) and provides a visitor-based `accept()` method for depth-first tree traversal. Unlike PyTorch's attribute-based child access, C++ modules expose children through the map and retrieve them via `static_pointer_cast` at runtime.

### Visitor Pattern

The visitor pattern enables any tree-wide operation without coupling the module hierarchy to a specific implementation. The abstract `Visitor` interface defines visit methods for each module type; concrete visitors implement specific operations such as weight loading, parameter inspection, or serialization. This decoupling means new operations can be added by writing a new visitor — no changes to Module or its children are required.

### Parameter vs. nn.Parameter

`Parameter` is a leaf module that holds a `Tensor::Shape` (declared before weight loading) and assigns tensor data via `set(Tensor)` at load time. Unlike PyTorch's `nn.Parameter` which owns its tensor, the C++ Parameter holds only the declared shape for validation — actual data comes from GGUF weight loading.

## How to Port a Diffusers Pipeline to C++

Follow these steps when porting a Python diffusers module:

1. **Create a class inheriting `Module`** (or a derived primitive like `Linear`).
2. **Add sub-modules via the children map** in the constructor:
   ```cpp
   modules["name"] = std::make_shared<SubModule>(args...);
   ```
3. **Implement `forward(Scope scope, ...)`** — every forward method takes `ctx` as the first parameter and returns a `Tensor`.
4. **Call sub-module forward via `static_pointer_cast`**:
   ```cpp
   auto sub = std::static_pointer_cast<SubModule>(modules["name"]);
   auto result = sub->forward(scope, input);
   ```
5. **Weight loading requires no extra work** — the model's `accept()` is called by the GGUF loader visitor, which traverses all Parameters automatically.

## C++ Syntax Conventions (Notable Differences from torch.nn)

- **`forward()` always takes `ggml_context*` as first parameter** — makes computation context explicit.
- **Sub-modules retrieved via `static_pointer_cast`** — not attribute access (`self.name` in Python).
- **Parameters hold `Tensor::Shape` for validation** — data assigned at load time, not stored in the module.
- **No autograd** — `forward()` returns computed tensors; no gradient tracking or backpropagation.

## Neural Network Primitives

| Class | torch.nn equivalent | C++ note |
|-------|-------------------|----------|
| `Linear` | `nn.Linear` | `forward()` takes ctx; bias optional in constructor |
| `SiLU` | `nn.SiLU` (silu()) | Activation — no parameters, stateless forward() |
| `Dropout` | `nn.Dropout` | p ratio in constructor; stateless during inference (no training mode) |
| `Identity` | `nn.Identity` | Passthrough — forward() returns input unchanged |

These primitives mirror torch.nn behavior. See [src/ggml/README.md](../ggml/) for Tensor operations used within these primitives.
