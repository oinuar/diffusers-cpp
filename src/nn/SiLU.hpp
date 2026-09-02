#pragma once

#include "nn/Module.hpp"
#include "ggml/Context.hpp"

class SiLU : public Module {
public:
    Tensor forward(Scope scope, Tensor x) {
        return Tensor(ggml_silu(*scope.context(), *x), x.shape());
    }
};
