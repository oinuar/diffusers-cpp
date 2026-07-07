#pragma once

#include "nn/Module.hpp"

class Qwen3Config;

class Qwen3DecoderLayer : public Module {
public:
    Qwen3DecoderLayer(const Qwen3Config& config, int layer_idx);

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask,
        std::pair<Tensor, Tensor> position_embeddings,
        std::optional<Qwen3Cache>& past_key_values,
        bool use_cache);

private:
    int layer_idx_;
};
