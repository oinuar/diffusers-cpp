#pragma once

#include "nn/Module.hpp"
#include "ggml/Context.hpp"

class SiLU : public Module {
public:
    Tensor forward(Context& context, Tensor x) {
        return Tensor(*context, ggml_silu(*context, *x), x.shape());
    }
};
