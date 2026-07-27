#pragma once

#include "nn/Module.hpp"
#include "ggml/Runtime.hpp"

class SiLU : public Module {
public:
    Tensor forward(Runtime& runtime, Tensor x) {
        return Tensor(*runtime.context(), ggml_silu(*runtime.context(), *x), x.shape());
    }
};
