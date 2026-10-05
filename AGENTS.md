- Build dir `build/` already exists and is configured.
- Always run tests through `ctest`. 
- If C++ CLI program fails (rc != 0) during `ctest`, the reproduction Python script will be written into build/bin/test-repro. You are free to modify the script while debugging. The script will be overwritten when `ctest` is ran again.
- Do not change source files in submodules (ggml, tokenizers-cpp, json) -- all the changes should be done in main project.
- Note that `GGML_META_DEBUG=1` flag's debug output is misleading: The state of the `src` is always displayed from `src[0]`.
- Real models load from GGUF via `from_pretrained()`; tests never touch GGUF — they use tiny random models from `params()`.
- Tensor shapes are in PyTorch order, but internally in GGML order. For example, Tensor::Shape{1, 2, 3} (PyTorch-style, first index is the slowest / outermost dimension) → ggml's ne[] = {3, 2, 1} (ggml's ne[0] is the fastest / innermost dimension).

## Build & test

```bash
cmake --build build -j8                                   # build dir pre-configured
ctest --test-dir build -j8 --output-on-failure -R <regex> # one ctest entry per python test case
ctest --test-dir build -j8 --output-on-failure            # full suite (~3 min)
```
