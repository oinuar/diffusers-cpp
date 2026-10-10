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

    /** @brief Runs the model.
     *  @return The logits -- unless `extract_hidden_states` is provided:
     *  in that case the caller needs only the hidden states, and the
     *  LM head projection (a vocab_size x seq_len GEMM) is never
     *  computed, and std::nullopt is returned.
     */
    std::optional<Tensor> forward(Scope scope, 
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
