#include "models/transformers/qwen3/Qwen3Attention.hpp"
#include "models/transformers/qwen3/Qwen3Config.hpp"

static Tensor rotate_half(Tensor x) {
    auto x1 = x[{Tensor::Slice::ellipsis(), Tensor::Slice::range(std::nullopt, x.shape()[-1] / 2)}];
    auto x2 = x[{Tensor::Slice::ellipsis(), Tensor::Slice::range(x.shape()[-1] / 2, std::nullopt)}];
    return Tensor::cat({-x2, x1}, -1);
}

static std::pair<Tensor, Tensor> apply_rotary_pos_emb(Tensor q, Tensor k, Tensor cos, Tensor sin) {
    auto cos_ = cos.unsqueeze(1);
    auto sin_ = sin.unsqueeze(1);
    auto q_embed = (q * cos_) + (rotate_half(q) * sin_);
    auto k_embed = (k * cos_) + (rotate_half(k) * sin_);
    return {q_embed, k_embed};
}

Qwen3Attention::Qwen3Attention(const Qwen3Config& config, int layer_idx)
    : head_dim_(config.head_dim.value_or(config.hidden_size / config.num_attention_heads)),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      layer_idx_(layer_idx)
{
    modules["q_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_attention_heads * head_dim_, config.attention_bias
    );
    modules["k_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_key_value_heads * head_dim_, config.attention_bias
    );
    modules["v_proj"] = std::make_shared<Linear>(
        config.hidden_size, config.num_key_value_heads * head_dim_, config.attention_bias
    );
    modules["o_proj"] = std::make_shared<Linear>(
        config.num_attention_heads * head_dim_, config.hidden_size, config.attention_bias
    );
    modules["q_norm"] = std::make_shared<Qwen3RMSNorm>(head_dim_, config.rms_norm_eps);
    modules["k_norm"] = std::make_shared<Qwen3RMSNorm>(head_dim_, config.rms_norm_eps);

    // TODO:
    // self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None ?
}

std::pair<Tensor, std::optional<Tensor>> Qwen3Attention::forward(
    Runtime& runtime,
    Tensor hidden_states,
    std::pair<Tensor, Tensor> position_embeddings,
    std::optional<Tensor> attention_mask,
    Qwen3Cache& past_key_values,
    bool use_cache)
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

    auto query_states = q_norm->forward(runtime, q_proj->forward(runtime, hidden_states).reshape(hidden_shape)).permute({0, 2, 1, 3});
    auto key_states = k_norm->forward(runtime, k_proj->forward(runtime, hidden_states).reshape(hidden_shape)).permute({0, 2, 1, 3});
    auto value_states = v_proj->forward(runtime, hidden_states).reshape(hidden_shape).permute({0, 2, 1, 3});

    auto& [cos, sin] = position_embeddings;
    auto [rotated_query_states, rotated_key_states] = apply_rotary_pos_emb(query_states, key_states, cos, sin);

    query_states = rotated_query_states;
    key_states = rotated_key_states;

    // KV cache: update and retrieve combined KV states.
    if (past_key_values) {
        auto [k, v] = past_key_values.update(key_states, value_states, layer_idx_);
        key_states = k;
        value_states = v;
    }

    SoftmaxAttnOp attn;

    auto attn_output = attn(
        ctx,
        *query_states,   // q: [N=B, n_head=num_q_heads, L_q=S, d_head=head_dim]
        *key_states,     // k: [N=B, n_head=num_kv_heads, L_k=S', d_head=head_dim] (S' = S + cached)
        *value_states,   // v: [N=B, n_head=num_kv_heads, L_k=S', d_head=head_dim]
        num_attention_heads_,         // n_head
        head_dim_,                    // d_head
        num_key_value_heads_,         // n_kv_head (handles repeat_kv internally)
        hidden_states.shape()[1],     // L_q (query sequence length)
        key_states.shape()[2],        // L_k (key sequence length including cache)
        hidden_states.shape()[0],     // N (batch size)
        *attention_mask.value_or(Tensor())
    );

    auto attn_output_shape = Tensor::Shape(input_shape.rank() + 1);

    for (auto i = 0; i < input_shape.rank(); ++i)
        attn_output_shape[i] = input_shape[i];

    attn_output_shape[attn_output_shape.rank()] = -1;

    auto o_proj = std::static_pointer_cast<Linear>(modules["o_proj"]);

    attn_output = attn_output.reshape(attn_output_shape);
    attn_output = o_proj->forward(runtime, attn_output);

    return {attn_output, std::nullopt};
}
