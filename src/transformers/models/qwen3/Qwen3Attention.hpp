#pragma once

#include "nn/Module.hpp"

struct Qwen3Config;

template <class AttnOp>
class Qwen3Attention : public Module {
public:
    Qwen3Attention(const Qwen3Config& config, int layer_idx);

    Tensor forward(Runtime& runtime, Tensor hidden_states, Tensor position_ids, 
                   std::optional<Tensor> attention_mask = std::nullopt, 
                   std::optional<Tensor> past_key_values = std::nullopt);

private:
    Qwen3Config config;
    int layer_idx;
    int head_dim;
    int num_key_value_groups;
    bool is_causal;
    std::string layer_type;
    std::optional<int> sliding_window;
};

#include "transformers/models/qwen3/Qwen3Attention.inl"
