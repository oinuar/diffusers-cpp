#pragma once

#include "nn/Module.hpp"

struct Qwen3Config;
class Qwen3RotaryEmbedding;

class Qwen3DecoderLayer : public Module {
public:
    Qwen3DecoderLayer(const Qwen3Config& config, int layer_idx);

    Tensor forward(Scope scope, Qwen3RotaryEmbedding& rotary_emb,
                   Tensor hidden_states, Tensor position_ids, 
                   std::optional<Tensor> attention_mask = std::nullopt,
                   std::optional<Tensor> past_key_values = std::nullopt,
                   std::optional<bool> use_cache = std::nullopt);
};
