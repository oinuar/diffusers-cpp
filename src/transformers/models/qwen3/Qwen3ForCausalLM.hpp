#pragma once

#include "nn/Module.hpp"

class Qwen3Cache;

struct CausalLMOutputWithPast {
    std::optional<float> loss;
    Tensor logits;
    Qwen3Cache past_key_values;
    std::optional<Tensor> hidden_states;
    std::optional<Tensor> attentions;
};

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(const Qwen3Config& config);

    CausalLMOutputWithPast forward(
        Runtime& runtime,
        std::optional<Tensor> input_ids = std::nullopt,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<Tensor> position_ids = std::nullopt,
        std::optional<Tensor> inputs_embeds = std::nullopt,
        bool use_cache = false,
        int logits_to_keep = 0);
};
