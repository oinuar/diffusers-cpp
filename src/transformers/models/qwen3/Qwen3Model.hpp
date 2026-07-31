#pragma once

#include "nn/Module.hpp"

struct Qwen3Config;

class Qwen3Model : public Module {
public:
    Qwen3Model(const Qwen3Config& config);

    Tensor forward(Runtime& runtime, std::optional<Tensor> input_ids = std::nullopt, 
                   std::optional<Tensor> inputs_embeds = std::nullopt, 
                   std::optional<Tensor> attention_mask = std::nullopt,
                   std::optional<Tensor> position_ids = std::nullopt,
                   std::optional<Tensor> past_key_values = std::nullopt,
                   std::optional<bool> use_cache = std::nullopt);

private:
    int vocab_size;
    int num_hidden_layers;
    std::vector<std::string> layer_types;
    bool has_sliding_layers;
};
