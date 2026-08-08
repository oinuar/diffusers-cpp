#pragma once

#include "nn/Module.hpp"
#include <filesystem>
#include <unordered_map>

class Backend;
struct Qwen3Config;

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(const Qwen3Config& config);

    static Qwen3ForCausalLM from_pretrained(Backend& loader_backend, Qwen3Config&& config, const std::filesystem::path& path);

    Tensor forward(Runtime& runtime, 
                   std::optional<Tensor> input_ids = std::nullopt, 
                   std::optional<Tensor> attention_mask = std::nullopt,
                   std::optional<Tensor> position_ids = std::nullopt,
                   std::optional<Tensor> past_key_values = std::nullopt,
                   std::optional<Tensor> inputs_embeds = std::nullopt,
                   std::optional<Tensor> labels = std::nullopt,
                   std::optional<bool> use_cache = std::nullopt,
                   int logits_to_keep = 0,
                   std::unordered_map<size_t, Tensor>* extract_hidden_states_by_layer = nullptr);

private:
    int vocab_size;
};
