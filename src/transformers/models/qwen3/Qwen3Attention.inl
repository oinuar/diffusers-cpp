#include "transformers/models/qwen3/Qwen3Attention.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "transformers/models/qwen3/Qwen3RotaryEmbedding.hpp"
#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "ggml/Runtime.hpp"
#include <iostream>

template <class AttnOp>
Qwen3Attention<AttnOp>::Qwen3Attention(const Qwen3Config& config, int layer_idx) {
    this->layer_idx = layer_idx;
    this->head_dim = config.head_dim;
    this->num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;
    this->is_causal = true;
    
    this->layer_type = config.layer_types.size() > static_cast<size_t>(layer_idx) 
                        ? config.layer_types[layer_idx] 
                        : "full_attention";
                        
    this->sliding_window = (this->layer_type == "sliding_attention") 
                            ? std::optional<int>(config.sliding_window) 
                            : std::nullopt;

    modules["q_proj"] = std::make_shared<Linear>(config.hidden_size, config.num_attention_heads * this->head_dim, config.attention_bias);
    modules["k_proj"] = std::make_shared<Linear>(config.hidden_size, config.num_key_value_heads * this->head_dim, config.attention_bias);
    modules["v_proj"] = std::make_shared<Linear>(config.hidden_size, config.num_key_value_heads * this->head_dim, config.attention_bias);
    modules["o_proj"] = std::make_shared<Linear>(config.num_attention_heads * this->head_dim, config.hidden_size, config.attention_bias);
    
    // Qwen3 applies RMSNorm specifically on the head_dim
    modules["q_norm"] = std::make_shared<Qwen3RMSNorm>(this->head_dim, config.rms_norm_eps);
    modules["k_norm"] = std::make_shared<Qwen3RMSNorm>(this->head_dim, config.rms_norm_eps);
}

template <class AttnOp>
Tensor Qwen3Attention<AttnOp>::forward(
    Runtime& runtime,
    Qwen3RotaryEmbedding& rotary_emb,
    Tensor hidden_states,
    Tensor position_ids, 
    std::optional<Tensor> attention_mask, 
    std::optional<Tensor> past_key_values
) {
    auto batch_size = hidden_states.shape()[0]; // batch
    auto seq_len = hidden_states.shape()[1]; // seq_len

    auto q_proj = std::static_pointer_cast<Linear>(modules["q_proj"]);
    auto k_proj = std::static_pointer_cast<Linear>(modules["k_proj"]);
    auto v_proj = std::static_pointer_cast<Linear>(modules["v_proj"]);
    auto o_proj = std::static_pointer_cast<Linear>(modules["o_proj"]);
    auto q_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["q_norm"]);
    auto k_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["k_norm"]);

    Tensor::Shape hidden_shape({batch_size, seq_len, -1, head_dim});

    // 1. Projections and reshaping to (batch, seq_len, num_heads, head_dim) -> [B, S, H, D]
    auto query_states = q_proj->forward(runtime, hidden_states).reshape(hidden_shape);
    auto key_states = k_proj->forward(runtime, hidden_states).reshape(hidden_shape);
    auto value_states = v_proj->forward(runtime, hidden_states).reshape(hidden_shape);

    // 2. Apply Q/K normalization
    query_states = q_norm->forward(runtime, query_states);
    key_states = k_norm->forward(runtime, key_states);

    // 3. RoPE calculation
    query_states = rotary_emb.forward(runtime, query_states, position_ids);
    key_states = rotary_emb.forward(runtime, key_states, position_ids);

    // Preserve optional code path for KV cache
    if (past_key_values.has_value()) {
        // TODO: Implement cache update logic matching past_key_values.update(key_states, value_states, layer_idx)
    }

    // 4. Attention calculation using operator
    AttnOp attn_op;
    auto attn_output = attn_op(runtime, query_states, key_states, value_states, attention_mask, pow(head_dim, -0.5));

    // 5. Transpose back, reshape and output projection
    auto attn_output_reshaped = attn_output.reshape({batch_size, seq_len, -1});
    
    return o_proj->forward(runtime, attn_output_reshaped);
}
