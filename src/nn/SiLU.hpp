#pragma once

#include "nn/Module.hpp"
#include "ggml/Context.hpp"

class SiLU : public Module {
public:
    Tensor forward(Scope scope, Tensor x) {
        return Tensor(scope.engine().silu(*x), x.shape());
    }
};
