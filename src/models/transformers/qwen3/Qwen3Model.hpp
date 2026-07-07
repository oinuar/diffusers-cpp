#pragma once

#include "nn/Module.hpp"

class Qwen3Cache;

struct BaseModelOutputWithPast {
    Tensor last_hidden_state;
    Qwen3Cache past_key_values;
    std::optional<Tensor> hidden_states;
    std::optional<Tensor> attentions;
};

class Qwen3Model : public Module {
public:
    Qwen3Model(const Qwen3Config& config);

    BaseModelOutputWithPast forward(
        ggml_context* ctx,
        std::optional<Tensor> input_ids = std::nullopt,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<Tensor> position_ids = std::nullopt,
        std::optional<Qwen3Cache> past_key_values = std::nullopt,
        std::optional<Tensor> inputs_embeds = std::nullopt,
        bool use_cache = false);

private:
    int64_t num_hidden_layers_;
};
