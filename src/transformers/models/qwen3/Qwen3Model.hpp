#pragma once

#include "nn/Module.hpp"
#include <unordered_map>

struct Qwen3Config;

class Qwen3Model : public Module {
public:
    Qwen3Model(const Qwen3Config& config);

    Tensor forward(Context& context, std::optional<Tensor> input_ids = std::nullopt, 
                   std::optional<Tensor> inputs_embeds = std::nullopt, 
                   std::optional<Tensor> attention_mask = std::nullopt,
                   std::optional<Tensor> position_ids = std::nullopt,
                   std::optional<Tensor> past_key_values = std::nullopt,
                   std::optional<bool> use_cache = std::nullopt,
                   std::vector<Tensor>* extract_hidden_states = nullptr);

private:
    int vocab_size;
    int num_hidden_layers;
    std::vector<std::string> layer_types;
    bool has_sliding_layers;
    std::optional<int64_t> sliding_window;
};
