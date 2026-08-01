#include "transformers/models/qwen3/Qwen3Config.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
void read(const json& j, const char* key, T& value) {
    if (!j.contains(key) || j[key].is_null())
        return;

    value = j[key].get<T>();
}

template <typename T>
void read(const json& j, const char* key, std::optional<T>& value) {
    if (!j.contains(key))
        return;

    if (j[key].is_null())
        value.reset();
    else
        value = j[key].get<T>();
}

Qwen3Config Qwen3Config::from_file(const std::filesystem::path& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + path.string());
    }

    json j;
    file >> j;

    Qwen3Config cfg;

    read(j, "vocab_size", cfg.vocab_size);
    read(j, "hidden_size", cfg.hidden_size);
    read(j, "intermediate_size", cfg.intermediate_size);
    read(j, "num_hidden_layers", cfg.num_hidden_layers);
    read(j, "num_attention_heads", cfg.num_attention_heads);
    read(j, "num_key_value_heads", cfg.num_key_value_heads);
    read(j, "head_dim", cfg.head_dim);

    read(j, "attention_dropout", cfg.attention_dropout);
    read(j, "attention_bias", cfg.attention_bias);
    read(j, "rms_norm_eps", cfg.rms_norm_eps);
    read(j, "initializer_range", cfg.initializer_range);

    read(j, "max_position_embeddings", cfg.max_position_embeddings);
    read(j, "max_window_layers", cfg.max_window_layers);

    read(j, "use_sliding_window", cfg.use_sliding_window);
    read(j, "sliding_window", cfg.sliding_window);

    read(j, "tie_word_embeddings", cfg.tie_word_embeddings);
    read(j, "use_cache", cfg.use_cache);

    read(j, "bos_token_id", cfg.bos_token_id);
    read(j, "eos_token_id", cfg.eos_token_id);
    read(j, "pad_token_id", cfg.pad_token_id);

    read(j, "rope_theta", cfg.rope_theta);
    read(j, "layer_types", cfg.layer_types);

    return cfg;
}
