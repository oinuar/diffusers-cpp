#pragma once

#include "nn/Module.hpp"
#include <filesystem>
#include <map>
#include <vector>
#include <optional>

class Backend;
class Context;
struct Qwen3Config;

class Qwen3ForCausalLM : public Module {
public:
    Qwen3ForCausalLM(const Qwen3Config& config, std::optional<std::vector<size_t>> extract_layer_hidden_states = std::nullopt);

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
                   int logits_to_keep = 0);

    const std::optional<std::map<size_t, Tensor>>& hidden_states() const {
        return extract_layer_hidden_states;
    }

private:
    int vocab_size;
    std::optional<std::map<size_t, Tensor>> extract_layer_hidden_states;
};
