#pragma once

#include "nn/Module.hpp"

class Qwen3RMSNorm : public Module {
public:
    Qwen3RMSNorm(int64_t hidden_size, float eps = 1e-6f);

    Tensor forward(Scope scope, Tensor hidden_states);

private:
    float eps;
};
