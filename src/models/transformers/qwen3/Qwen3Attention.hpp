#pragma once

#include "modules/Module.hpp"
#include "models/transformers/qwen3/Qwen3Config.hpp"

class Qwen3Attention : public Module {
public:
    Qwen3Attention(const Qwen3Config& config, int layer_idx);

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::tuple<Tensor, Tensor> position_embeddings,
        std::optional<Tensor> attention_mask = std::nullopt
        // past_key_values
    );

private:
    int64_t head_dim_;
    int layer_idx_;
};
