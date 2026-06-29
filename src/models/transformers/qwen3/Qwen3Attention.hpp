#pragma once

#include "modules/Module.hpp"

class Qwen3Config;

class Qwen3Attention : public Module {
public:
    Qwen3Attention(const Qwen3Config& config, int layer_idx);

    std::pair<Tensor, std::optional<Tensor>> forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::pair<Tensor, Tensor> position_embeddings,
        std::optional<Tensor> attention_mask,
        std::optional<Qwen3Cache>& past_key_values,
        bool use_cache);

private:
    int64_t head_dim_, num_attention_heads_, num_key_value_heads_;
    int layer_idx_;
};
