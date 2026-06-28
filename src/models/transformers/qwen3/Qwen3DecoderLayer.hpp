#pragma once

#include "modules/Module.hpp"
#include "models/transformers/qwen3/Qwen3Config.hpp"

class Qwen3DecoderLayer : public Module {
public:
    Qwen3DecoderLayer(const QwenConfig& config, int layer_idx);

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<Tensor> position_ids = std::nullopt,
        //past_key_values: Cache | None = None,
        bool use_cache = false,
        std::optional<std::tuple<Tensor, Tensor>> position_embeddings = std::nullopt);
};
