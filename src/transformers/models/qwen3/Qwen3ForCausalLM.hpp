#pragma once

#include "nn/Module.hpp"
#include <filesystem>
#include <unordered_map>

class Backend;
class Context;
struct Qwen3Config;

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(const Qwen3Config& config);

    static Qwen3ForCausalLM from_pretrained(Context& context, Qwen3Config&& config, const std::filesystem::path& path);

    Tensor forward(Scope scope, 
                   std::optional<Tensor> input_ids = std::nullopt, 
                   std::optional<Tensor> attention_mask = std::nullopt,
                   std::optional<Tensor> position_ids = std::nullopt,
                   std::optional<Tensor> past_key_values = std::nullopt,
                   std::optional<Tensor> inputs_embeds = std::nullopt,
                   std::optional<Tensor> labels = std::nullopt,
                   std::optional<bool> use_cache = std::nullopt,
                   int logits_to_keep = 0,
                   std::vector<Tensor>* extract_hidden_states = nullptr);

private:
    int vocab_size;
};
