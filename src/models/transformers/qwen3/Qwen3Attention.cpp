#include "models/transformers/qwen3/Qwen3Attention.hpp"

Qwen3Attention::Qwen3Attention(Qwen3Config config, int layer_idx)
    : head_dim_(config.head_dim.value_or(config.hidden_size / config.num_attention_heads)), layer_idx_(layer_idx)
{
    auto head_dim = ;

    //self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
    //self.config = config
    //self.layer_idx = layer_idx
    //self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    //self.scaling = self.head_dim**-0.5
    //self.attention_dropout = config.attention_dropout
    //self.is_causal = True

    modules["q_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_attention_heads * head_dim, config.attention_bias
    );
    modules["k_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias
    );
    modules["v_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias
    );
    modules["o_proj"] = std::make_shared<Linear>(
        config.num_attention_heads * head_dim, config.hidden_size, config.attention_bias
    );
    modules["q_norm"] = std::make_shared<Qwen3RMSNorm>(head_dim, config.rms_norm_eps);  // unlike olmo, only on the head dim!
    modules["k_norm"] = std::make_shared<Qwen3RMSNorm>(head_dim, config.rms_norm_eps);  // thus post q_norm does not need reshape
    //self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
}

Tensor Qwen3Attention::forward(
    ggml_context* ctx,
    Tensor hidden_states,
    std::tuple<Tensor, Tensor> position_embeddings,
    std::optional<Tensor> attention_mask
    // past_key_values
)
{
    auto input_shape = Tensor::Shape(hidden_states.shape().rank() - 1);
    auto hidden_shape = Tensor::Shape(input_shape.rank() + 2);

    for (auto i = 0; i < input_shape.rank(); ++i)
        hidden_shape[i] = input_shape[i];

    hidden_shape[input_shape.rank() + 0] = -1;
    hidden_shape[input_shape.rank() + 1] = head_dim_;

    auto q_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["q_norm"]);
    auto q_proj = std::static_pointer_cast<Linear>(modules["q_proj"]);
    auto k_norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["k_norm"]);
    auto k_proj = std::static_pointer_cast<Linear>(modules["k_proj"]);
    auto v_proj = std::static_pointer_cast<Linear>(modules["v_proj"]);

    auto query_states = q_norm->forward(ctx, q_proj->forward(ctx, hidden_states).reshape(hidden_shape)).permute({0, 2, 1, 3});
    auto key_states = k_norm->forward(ctx, k_proj->forward(ctx, hidden_states).reshape(hidden_shape)).permute({0, 2, 1, 3});
    auto value_states = v_proj->forward(ctx, hidden_states).reshape(hidden_shape).permute({0, 2, 1, 3});

    cos, sin = position_embeddings
    [query_states, key_states] = apply_rotary_pos_emb(query_states, key_states, cos, sin);

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, layer_idx_)

    auto [attn_output, attn_weights] = AttnOp(
        ctx,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=scaling_,
        sliding_window=sliding_window_,  // diff with Llama
        **kwargs,
    );

    auto attn_output_shape = Tensor::Shape(input_shape.rank() + 1);

    for (auto i = 0; i < input_shape.rank(); ++i)
        attn_output_shape[i] = input_shape[i];

    attn_output_shape[attn_output_shape.rank()] = -1;

    auto o_proj = std::static_pointer_cast<Linear>(modules["o_proj"]);

    attn_output = attn_output.reshape(attn_output_shape).contiguous()
    attn_output = o_proj->forward(ctx, attn_output);

    return {attn_output, attn_weights};
}
