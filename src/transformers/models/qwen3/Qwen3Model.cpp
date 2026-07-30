#include "transformers/models/qwen3/Qwen3Model.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3DecoderLayer.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "nn/ModuleList.hpp"
#include "nn/Embedding.hpp"
#include "ggml/Runtime.hpp"

Qwen3Model::Qwen3Model(const Qwen3Config& config) {
    this->vocab_size = config.vocab_size;
    this->pad_token_id = config.pad_token_id;
    this->num_hidden_layers = config.num_hidden_layers;
    this->layer_types = config.layer_types;
    
    this->has_sliding_layers = false;
    for (const auto& lt : layer_types) {
        if (lt == "sliding_attention") {
            this->has_sliding_layers = true;
            break;
        }
    }

    modules["embed_tokens"] = std::make_shared<Embedding>(config.vocab_size, config.hidden_size, config.pad_token_id);
    
    auto layers = std::make_shared<ModuleList>(config.num_hidden_layers);
    modules["layers"] = layers;
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        (*layers)[i] = std::make_shared<Qwen3DecoderLayer>(config, i);
    }

    modules["norm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

Tensor Qwen3Model::forward(
    Runtime& runtime,
    std::optional<Tensor> input_ids, 
    std::optional<Tensor> inputs_embeds, 
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::optional<Tensor> past_key_values,
    std::optional<bool> use_cache
) {    
    if (!input_ids.has_value() && !inputs_embeds.has_value()) {
        throw std::invalid_argument("You must specify exactly one of input_ids or inputs_embeds");
    }

    Tensor hidden_states;
    if (inputs_embeds.has_value()) {
        hidden_states = inputs_embeds.value();
    } else {
        auto embed_tokens = std::static_pointer_cast<Embedding>(modules["embed_tokens"]);
        hidden_states = embed_tokens->forward(runtime, input_ids.value());
    }

    auto seq_len = hidden_states.shape()[1];
    Tensor pos_ids;
    
    if (position_ids.has_value()) {
        pos_ids = position_ids.value();
    } else {
        // Preserve Python logic: position_ids = torch.arange(seq_len) + past_seen_tokens
        // Assuming past_seen_tokens is 0 if past_key_values is not provided
        int past_seen_tokens = 0; // Simplified for porting; would query past_key_values if available
        pos_ids = Tensor::arange(*runtime.context(), 0, seq_len);
        pos_ids = pos_ids + past_seen_tokens;
        pos_ids = pos_ids.unsqueeze(0);
    }

    auto layers = std::static_pointer_cast<ModuleList>(modules["layers"]);
    for (auto i = 0; i < layers->size(); ++i) {
        auto layer = std::static_pointer_cast<Qwen3DecoderLayer>((*layers)[i]);
        
        std::optional<Tensor> layer_mask = std::nullopt;
        if (attention_mask.has_value()) {
            // In Python, causal_mask_mapping selects between full and sliding masks.
            // We pass the pre-computed mask directly to preserve the execution path.
            layer_mask = attention_mask.value();
        }

        hidden_states = layer->forward(runtime, hidden_states, pos_ids, layer_mask, past_key_values, use_cache);
    }

    auto norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["norm"]);
    hidden_states = norm->forward(runtime, hidden_states);

    return hidden_states;
}
