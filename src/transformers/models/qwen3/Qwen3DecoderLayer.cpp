#include "models/transformers/qwen3/Qwen3DecoderLayer.hpp"
#include "models/transformers/qwen3/Qwen3Config.hpp"
#include "models/transformers/qwen3/Qwen3Attention.hpp"
#include "models/transformers/qwen3/Qwen3MLP.hpp"

Qwen3DecoderLayer::Qwen3DecoderLayer(const Qwen3Config& config, int layer_idx)
    : layer_idx_(layer_idx)
{
    modules["self_attn"] = std::make_shared<Qwen3Attention>(config, layer_idx);

    modules["mlp"] = std::make_shared<Qwen3MLP>(config.hidden_size, config.intermediate_size);
    modules["input_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
    modules["post_attention_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

Tensor Qwen3DecoderLayer::forward(
    Runtime& runtime,
    Tensor hidden_states,
    std::optional<Tensor> attention_mask,
    std::pair<Tensor, Tensor> position_embeddings,
    std::optional<Qwen3Cache>& past_key_values,
    bool use_cache) 
{
    auto residual = hidden_states;

    auto input_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["input_layernorm"]);
    hidden_states = input_layernorm->forward(runtime, hidden_states);

    // Self Attention
    auto self_attn = std::static_pointer_cast<Qwen3Attention>(modules["self_attn"]);
    hidden_states = std::get<0>(self_attn->forward(
        ctx,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        position_embeddings
    ));

    hidden_states = residual + hidden_states;

    // Fully Connected
    residual = hidden_states;

    auto post_attention_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["post_attention_layernorm"]);
    hidden_states = post_attention_layernorm->forward(runtime, hidden_states);

    auto mlp = std::static_pointer_cast<Qwen3MLP>(modules["mlp"]);
    hidden_states = residual + mlp->forward(runtime, hidden_states);

    return hidden_states;
}
