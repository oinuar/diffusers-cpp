#pragma once

#include "modules/Module.hpp"

class Qwen3Config;

class Qwen3RotaryEmbedding : public Module {
public:
    Qwen3RotaryEmbedding(const Qwen3Config& config);

    std::pair<Tensor, Tensor> forward(ggml_context* ctx, Tensor x, Tensor position_ids);

private:
    int64_t dim_;
    float base_;
};
