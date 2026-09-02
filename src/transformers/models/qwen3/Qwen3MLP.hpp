#pragma once

#include "nn/Module.hpp"

struct Qwen3Config;

class Qwen3MLP : public Module {
public:
    Qwen3MLP(const Qwen3Config& config);

    Tensor forward(Scope scope, Tensor x);
};
