#pragma once

#include <vector>
#include <string>
#include <optional>

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
    float rms_norm_eps = 1e-06f;
    float initializer_range = 0.02f;
    
    //std::string hidden_act = "silu";
    //std::string model_type = "qwen3";
    
    int64_t max_position_embeddings = 32768;
    int64_t max_window_layers = 28;
    
    bool use_sliding_window = false;
    std::optional<int64_t> sliding_window = std::nullopt;
    
    bool tie_word_embeddings = false;
    bool use_cache = true;
    
    std::optional<int64_t> bos_token_id = std::nullopt;
    std::optional<int64_t> eos_token_id = std::nullopt;
    std::optional<int64_t> pad_token_id = std::nullopt;
    
    // RoPE parameters
    float rope_theta = 10000.0f;
    //std::string rope_type = "default";
    
    // Default 32 layers initialized to "full_attention"
    std::vector<std::string> layer_types = std::vector<std::string>(32, "full_attention");
};
