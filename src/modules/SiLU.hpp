#pragma once

#include "modules/Module.hpp"

class SiLU : public Module {
public:
    Tensor forward(ggml_context* ctx, Tensor x) {
        return Tensor(ctx, ggml_silu(ctx, *x));
    }
};
