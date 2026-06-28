#pragma once

#include "modules/Module.hpp"

class Qwen3ForCausalLM : public Module {
public:
    Qwen3Model(const Qwen3Config& config);

    Tensor forward(
        ggml_context* ctx,
        std::optional<std::vector<int64_t>> input_ids = std::nullopt,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::vector<int64_t>> position_ids = std::nullopt,
        // past_key_values: Cache | None = None,
        std::optional<Tensor> inputs_embeds = std::nullopt,
        std::optional<std::vector<int64_t>> labels = std::nullopt,
        bool use_cache = false,
        int logits_to_keep = 0
        //**kwargs: Unpack[TransformersKwargs],
    );
};
