#pragma once

#include "modules/Module.hpp"

class Qwen3RotaryEmbedding : public Module {
public:
    Qwen3RotaryEmbedding(const Qwen3Config& config);

    std::pair<Tensor, Tensor> forward(ggml_context* ctx, Tensor x, std::vector<int64_t> position_ids);
};
