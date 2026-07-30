#include "transformers/models/qwen3/Qwen3Attention.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "ggml/Runtime.hpp"

template <class AttnOp>
Qwen3Attention<AttnOp>::Qwen3Attention(const Qwen3Config& config, int layer_idx) {
    this->config = config;
    this->layer_idx = layer_idx;
    this->head_dim = config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads;
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
    Tensor hidden_states,
    Tensor position_ids, 
    std::optional<Tensor> attention_mask, 
    std::optional<Tensor> past_key_values
) {
    
    auto input_shape = Tensor::Shape(hidden_states.shape().rank() - 1); // (batch, seq_len)
    
    // take all dimensions except the last one
    for (auto i = 0; i < input_shape.rank(); ++i)
        input_shape[i] = hidden_states.shape()[i];

    auto batch_size = input_shape[0];
    auto seq_len = input_shape[1];

    auto q_proj = std::static_pointer_cast<Linear>(modules["q_proj"]);
    auto k_proj = std::static_pointer_cast<Linear>(modules["k_proj"]);
    auto v_proj = std::static_pointer_cast<Linear>(modules["v_proj"]);
    auto o_proj = std::static_pointer_cast<Linear>(modules["o_proj"]);
    auto q_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["q_norm"]);
    auto k_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["k_norm"]);

    // 1. Projections and reshaping to (batch, seq_len, num_heads, head_dim) -> [B, S, H, D]
    auto q = q_proj->forward(runtime, hidden_states).reshape({batch_size, seq_len, config.num_attention_heads, head_dim});
    auto k = k_proj->forward(runtime, hidden_states).reshape({batch_size, seq_len, config.num_key_value_heads, head_dim});
    auto v = v_proj->forward(runtime, hidden_states).reshape({batch_size, seq_len, config.num_key_value_heads, head_dim});

    // 2. Apply Q/K normalization
    auto q_normed = q_norm->forward(runtime, q);
    auto k_normed = k_norm->forward(runtime, k);

    // 3. RoPE calculation using ggml_rope_ext (direct position_ids, no sin & cos)
    auto q_rope_ggml = ggml_rope_ext(
        *runtime.context(),
        *q_normed,
        *position_ids,
        nullptr,            // freq_factors
        head_dim,           // n_dims
        0,                  // mode
        0,                  // n_ctx_orig
        config.rope_theta,  // freq_base
        1.0f,               // freq_scale
        0.0f, 0.0f, 0.0f, 0.0f // ext_factor, attn_factor, beta_fast, beta_slow
    );
    auto query_states = Tensor(*runtime.context(), q_rope_ggml, q_normed.shape());

    auto k_rope_ggml = ggml_rope_ext(
        *runtime.context(),
        *k_normed,
        *position_ids,
        nullptr,
        head_dim,
        0,
        0,
        config.rope_theta,
        1.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    );
    auto key_states = Tensor(*runtime.context(), k_rope_ggml, k_normed.shape());

    // Preserve optional code path for KV cache
    if (past_key_values.has_value()) {
        // TODO: Implement cache update logic matching past_key_values.update(key_states, value_states, layer_idx)
    }

    // 4. Attention calculation using operator
    AttnOp attn_op;
    auto attn_output = attn_op(runtime, query_states, key_states, v, attention_mask);

    // 5. Reshape and output projection
    auto attn_output_reshaped = attn_output.reshape({batch_size, seq_len, config.num_attention_heads * head_dim});
    
    return o_proj->forward(runtime, attn_output_reshaped);
}
