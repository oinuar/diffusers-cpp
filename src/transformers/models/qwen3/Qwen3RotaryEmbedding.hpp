#pragma once

#include "nn/Module.hpp"

struct Qwen3Config;

class Qwen3RotaryEmbedding : public Module {
public:
    Qwen3RotaryEmbedding(const Qwen3Config& config);

    Tensor forward(
        Runtime& runtime,
        Tensor x,
        Tensor position_ids
    );

private:
    int64_t head_dim;
    float rope_theta;
};
