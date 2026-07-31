#pragma once

#include <vector>
#include <string>

struct Qwen3Config {
    int64_t vocab_size = 151936;
    int64_t hidden_size = 4096;
    int64_t intermediate_size = 22016;
    int64_t num_hidden_layers = 32;
    int64_t num_attention_heads = 32;
    int64_t num_key_value_heads = 32;
    int64_t head_dim = 128;
    float attention_dropout = 0.0f;
    bool attention_bias = false;
    float rms_norm_eps = 1e-6;
    std::string hidden_act;
    std::vector<std::string> layer_types;
    std::optional<int64_t> pad_token_id;
    int64_t sliding_window = 4096;
    int64_t max_position_embeddings = 32768;
    float rope_theta = 1000.0f;
};
