#include "transformers/models/qwen3/Qwen3DecoderLayer.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "transformers/models/qwen3/Qwen3MLP.hpp"
#include "transformers/models/qwen3/Qwen3Attention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

Qwen3DecoderLayer::Qwen3DecoderLayer(const Qwen3Config& config, int layer_idx) {        
    modules["self_attn"] = std::make_shared<Qwen3Attention<FlashAttentionOp>>(config, layer_idx);
    modules["mlp"] = std::make_shared<Qwen3MLP>(config);
    modules["input_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
    modules["post_attention_layernorm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

Tensor Qwen3DecoderLayer::forward(
    Runtime& runtime,
    Qwen3RotaryEmbedding& rotary_emb,
    Tensor hidden_states,
    Tensor position_ids, 
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> past_key_values,
    std::optional<bool> use_cache
) {
    
    // Self Attention Block
    auto residual = hidden_states;
    
    auto input_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["input_layernorm"]);
    hidden_states = input_layernorm->forward(runtime, hidden_states);

    auto self_attn = std::static_pointer_cast<Qwen3Attention<FlashAttentionOp>>(modules["self_attn"]);
    hidden_states = self_attn->forward(runtime, rotary_emb, hidden_states, position_ids, attention_mask, past_key_values);

    hidden_states = residual + hidden_states;

    // Fully Connected Block
    residual = hidden_states;
    
    auto post_attention_layernorm = std::static_pointer_cast<Qwen3RMSNorm>(modules["post_attention_layernorm"]);
    hidden_states = post_attention_layernorm->forward(runtime, hidden_states);

    auto mlp = std::static_pointer_cast<Qwen3MLP>(modules["mlp"]);
    hidden_states = mlp->forward(runtime, hidden_states);

    hidden_states = residual + hidden_states;

    return hidden_states;
}
