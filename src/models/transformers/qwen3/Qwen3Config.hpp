#pragma once

struct Qwen3Config {
    int vocab_size = 151936;
    int hidden_size = 4096;
    int intermediate_size = 22016;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    std::optional<int> num_key_value_heads = 32;
    std::optional<int> head_dim = 128;
    // str hidden_act = "silu";
    int max_position_embeddings = 32768;
    float initializer_range = 0.02;
    float rms_norm_eps = 1e-6;
    bool use_cache = true;
    bool tie_word_embeddings = false;
    // RopeParameters rope_parameters | dict | None = None;
    bool attention_bias = false;
    bool use_sliding_window = false;
    std::optional<int> sliding_window = 4096;
    int max_window_layers = 28;
    // list layer_types[str] | None = None;
    float attention_dropout = 0.0;
    //int pad_token_id | None = None;
    //int bos_token_id | None = None;
    //int eos_token_id | list[int] | None = None;
};
