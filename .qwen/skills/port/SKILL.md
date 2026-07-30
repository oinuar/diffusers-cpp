---
name: port
description: This skill provides steps for translating Python library modules into idiomatic C++ while preserving behavior, structure, and numerical correctness. The Python implementation should be treated as the specification. The goal is to produce a C++ implementation that behaves identically, not one that merely produces similar results.
---

# Porting Diffusers Classes to C++

This skill provides steps for translating Python library modules into idiomatic C++ while preserving behavior, structure, and numerical correctness.

The Python implementation should be treated as the specification. The goal is to produce a C++ implementation that behaves identically, not one that merely produces similar results.

---

## 1. Preserve the Original Structure

Treat the Python implementation as the specification.

When porting:

* Preserve the constructor structure.
* Preserve module hierarchy.
* Preserve execution order.
* Preserve optional code paths.
* Preserve default parameter values.

Do not simplify or optimize during the initial port. Correctness should always come before optimization.

### Build Complex Models Incrementally

Large models are usually compositions of smaller modules.

Port them in dependency order:

1. Primitive layers
2. Reusable building blocks
3. Composite blocks
4. Top-level models

---

## 2. Create the Module Class First

Before porting any implementation, create the C++ module that mirrors the Python class.

Python:

```python
class SomeModule(nn.Module):
    ...
```

becomes:

```cpp
class SomeModule : public Module {
public:
    SomeModule(...);

private:
    // configuration members -- NEVER store module referenes here, those belong to modules registry
};
```

Method signatures should closely resemble the original API.

For example Python:

```python
def forward(self, sample, condition=None):
```

maps naturally to:

```cpp
Tensor forward(
    Runtime& runtime,
    Tensor sample,
    std::optional<Tensor> condition = std::nullopt
)
```

Preserving the public API makes comparing implementations significantly easier.

General guidelines:

* Inherit from the project's `Module` base class.
* Keep the class name identical to the Python implementation whenever possible.
* Use the same constructor arguments and default values.
* Implement methods matching the Python signature as closely as possible.
* Store configuration values as member variables.
* Store child modules using the project's module registration mechanism.

Only after the class skeleton exists should the constructor and other methods to be ported.

---

## 3. Port the Constructor

Translate the constructor before implementing any methods.

For every Python submodule, for example:

```python
self.linear = nn.Linear(in_features, out_features)
self.norm = nn.LayerNorm(out_features)
self.act = nn.SiLU()

blocks = []

for _ in range(num_blocks):
    blocks.append(Block(
        block_param_1,
        block_param_2,
        ...
    ))

self.blocks = ModuleList(blocks)
```

create the equivalent C++ module using the project's module registration pattern:

```cpp
modules["linear"] =
    std::make_shared<Linear>(
        in_features,
        out_features
    );

modules["norm"] =
    std::make_shared<LayerNorm>(
        out_features
    );

modules["act"] =
    std::make_shared<SiLU>();

auto blocks = std::make_shared<ModuleList>(num_blocks);
modules["blocks"] = blocks;

for (auto i = 0; i < blocks->size(); ++i)
    (*blocks)[i] = std::make_shared<Block>(
        block_param_1,
        block_param_2,
        ...
    );
```

The constructor should only:

* create child modules
* store configuration values
* preserve default arguments
* preserve construction order

---

## 4. Port Methods Line-by-Line

Translate each Python statement directly into its C++ equivalent.

For example Python:

```python
sample = self.conv_in(sample)
sample = self.norm(sample)
sample = self.act(sample)
sample = self.conv_out(sample)
```

should become:

```cpp
auto conv_in = std::static_pointer_cast<Conv2d>(modules["conv_in"]);
sample = conv_in->forward(ctx, sample);

auto norm = std::static_pointer_cast<LayerNorm>(modules["norm"]);
sample = norm->forward(ctx, sample);

auto act = std::static_pointer_cast<SiLU>(modules["act"]);
sample = act->forward(ctx, sample);

auto conv_out = std::static_pointer_cast<Conv2d>(modules["conv_out"]);
sample = conv_out->forward(ctx, sample);
```

The goal is that someone reading the C++ implementation can compare it to the original Python implementation line-by-line.

Do not:

* reorder operations
* combine operations
* remove operations that appear redundant
* rewrite the algorithm into a different form
* store module references

Small ordering differences can produce measurable numerical differences.

### Use Existing Abstractions

Always use the project's `Tensor` wrapper and avoid interacting with GGML directly.

The `Tensor` wrapper is designed to provide a PyTorch-like interface, so PyTorch tensor operations should be translated to their `Tensor` wrapper equivalents whenever possible.

For example, Python:

```python
x = x.unsqueeze(-1)
x = x.reshape(...)
x = torch.cat([a, b], dim=-1)
x = x + bias
x = x * scale
```

should become:

```cpp
x = x.unsqueeze(-1);
x = x.reshape(...);
x = Tensor::cat({a, b}, -1);
x = x + bias;
x = x * scale;
```

Avoid calling GGML functions directly inside modules. For example, avoid:

```cpp
ggml_add(...)
ggml_mul(...)
ggml_reshape(...)
ggml_repeat(...)
```

Instead, use the corresponding `Tensor` wrapper methods and operators.

Direct GGML operators should only be used when there is no equivalent `Tensor` wrapper operation available or when a GGML-specific operator has no meaningful high-level abstraction. Examples include specialized operators such as:

```cpp
ggml_rope(...)
ggml_upscale(...)
```

The `Tensor` wrapper is responsible for:

* shape tracking
* broadcasting
* ownership
* interaction with GGML
* logical shapes

Modules should focus on implementing model logic rather than backend details.

### Preserve Logical Shapes

The Tensor wrapper stores both a GGML tensor and its logical shape. Always compute expected shapes manually using `Tensor::Shape` class after direct usage of GGML operators, for example:

```
Tensor(*runtime.context(), ggml_arange(...), Tensor::Shape({size}));
``` 

### Keep Changes Mechanical

During the port, every change should be explainable as a direct translation of the Python source.

Avoid introducing:

* alternative algorithms
* architectural improvements
* performance optimizations
* stylistic rewrites

---

# 5. Write Tests Against the Python Reference

Every ported module should have a Python reference test that verifies the C++ implementation produces the same output.

Use a C++ command-line program (`TestCLI`) that instantiates the C++ module, loads its parameters from the command line, executes its `forward()` method, and prints the resulting tensors in a machine-readable format.

A typical testing workflow is:

1. Instantiate the original Python module.
2. Generate test inputs.
3. Export all learnable parameters from the Python module.
4. Invoke the C++ CLI with the same constructor arguments, inputs, and parameters.
5. Parse the CLI output back into Python tensors.
6. Compare the C++ output against the Python reference.

For example:

```python
model = SomeModule(...)

input = torch.randn(...)

expected = model.forward(input)

actual = self.cli(
    "SomeModule",
    "--input", str(input.tolist()),
    *self.params(model),
)

self.assertTensors(actual, [expected])
```

The CLI should act as a thin wrapper around the C++ implementation. It should only:

* construct the requested module
* load parameters supplied by the test
* execute `forward()`
* print the resulting tensors

Each module should have its own reference test before being integrated into larger models. Testing modules in isolation makes debugging significantly easier because failures can be traced to a single component instead of an entire network.
