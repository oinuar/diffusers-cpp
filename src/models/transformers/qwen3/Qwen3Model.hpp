#pragma once

#include "modules/Module.hpp"

class Qwen3Model : public Module {
public:
    Qwen3Model(const Qwen3Config& config);

    Tensor forward(
        ggml_context* ctx,
        std::optional<Tensor> input_ids = std::nullopt,
        std::optinal<Tensor> attention_mask = std::nullopt,
        std::optional<Tensor> position_ids = std::nullopt,
        //past_key_values: Cache | None = None,
        std::optional<Tensor> inputs_embeds = std::nullopt,
        bool use_cache = false
        //**kwargs: Unpack[TransformersKwargs],
    );

private:
    int64_t num_hidden_layers_;
};
