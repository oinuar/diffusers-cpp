#include "models/transformers/qwen3/Qwen3DecoderLayer.hpp"
#include "models/transformers/qwen3/Qwen3Attention.hpp"
#include "models/transformers/qwen3/Qwen3MLP.hpp"
#include "models/transformers/qwen3/Qwen3RMSNorm.hpp"

Qwen3DecoderLayer::Qwen3DecoderLayer(const QwenConfig& config, int layer_idx) {
    self.hidden_size = config.hidden_size

    modules["self_attn"] = std::make_shared<Qwen3Attention>(config, layer_idx);

    modules["mlp"] = std::make_shared<Qwen3MLP>(config);
    modules["input_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
    modules["post_attention_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

Tensor Qwen3DecoderLayer::forward(
    ggml_context* ctx,
    Tensor hidden_states,
    std::optional<Tensor> attention_mask = std::nullopt,
    std::optional<Tensor> position_ids = std::nullopt,
    //past_key_values: Cache | None = None,
    bool use_cache = false,
    std::optional<std::tuple<Tensor, Tensor>> position_embeddings = std::nullopt);
{
    auto input_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["input_layernorm"]);

    residual = hidden_states
    hidden_states = input_layernorm->forward(ctx, hidden_states);

    auto self_attn = std::static_pointer_cast<Qwen3Attention>(modules["self_attn"]);

    // Self Attention
    [hidden_states, _] = self_attn->forward(
        ctx,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    auto post_attention_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["post_attention_layernorm"]);
    auto mlp = std::static_pointer_cast<Qwen3MLP>(modules["mlp"]);

    // Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
}
