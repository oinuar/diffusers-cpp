#pragma once

#include "nn/Module.hpp"
#include "ggml/Context.hpp"

class SiLU : public Module {
public:
    Tensor forward(Scope scope, Tensor x) {
        // SiLU(x) = x * sigmoid(x), composed of ops the meta backend supports:
        // ggml_silu (GGML_OP_SILU) has no split-state rule there.
        return x * Tensor(scope.runtime().sigmoid(*x), x.shape());
    }
};
