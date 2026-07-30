#pragma once

#include <vector>
#include <string>

struct Qwen3Config {
    int64_t vocab_size;
    int64_t hidden_size;
    int64_t intermediate_size;
    int64_t num_hidden_layers;
    int64_t num_attention_heads;
    int64_t num_key_value_heads;
    int64_t head_dim;
    float attention_dropout;
    bool attention_bias;
    float rms_norm_eps;
    std::string hidden_act;
    std::vector<std::string> layer_types;
    int64_t pad_token_id;
    int64_t sliding_window;
    int64_t max_position_embeddings;
    float rope_theta;
};
