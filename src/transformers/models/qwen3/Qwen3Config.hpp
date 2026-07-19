#pragma once

#include <string>
#include <vector>

struct Qwen3Config {
    int vocab_size = 151936;
    int hidden_size = 4096;
    int intermediate_size = 22016;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    // Non-optional: defaults to num_attention_heads (no GQA when equal).
    int num_key_value_heads = 32;
    std::optional<int> head_dim = 128; // optional — can be inferred as hidden_size / num_attention_heads.
    int max_position_embeddings = 32768;
    float initializer_range = 0.02;
    float rms_norm_eps = 1e-6;
    bool use_cache = false; //true;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool use_sliding_window = false;
    std::optional<int> sliding_window = 4096;
    int max_window_layers = 28;

    // Explicit per-layer attention type, matching Python's config.layer_types.
    // Contains "full_attention" and/or "sliding_attention" strings.
    std::vector<std::string> layer_types;

    float attention_dropout = 0.0;

    // Optional padding token ID — defaults to no padding (std::nullopt). The Embedding module uses -1 when not set.
    std::optional<int64_t> pad_token_id = std::nullopt;

    struct {
        float rope_theta = 10000.0f;
    } rope_parameters;
};
