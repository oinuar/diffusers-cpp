#include "transformers/models/qwen3/Qwen3RotaryEmbedding.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "ggml/Context.hpp"

Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(const Qwen3Config& config)
    : head_dim(config.head_dim), rope_theta(config.rope_theta)
{
}

Tensor Qwen3RotaryEmbedding::forward(
    Scope scope,
    Tensor x,
    Tensor position_ids
) {
    // GGML RoPE expects position IDs to be 32b integers.
    if (position_ids.dtype() != GGML_TYPE_I32)
        position_ids = position_ids.to(GGML_TYPE_I32);

    // GGML RoPE expects position IDs to be a 1D tensor (vector).
    if (!ggml_is_vector(*position_ids))
        position_ids = position_ids.flatten();

    auto rope = ggml_rope_ext(
        *scope.context(),
        *x,
        *position_ids,
        nullptr,             // freq_factors
        head_dim,            // n_dims = head_dim (rotate first 'head_dim' dims per head)
        GGML_ROPE_TYPE_NEOX,
        0,                   // n_ctx_orig
        rope_theta,          // freq_base
        1.0f,                // freq_scale
        0.0f, 1.0f, 0.0f, 0.0f // ext_factor, attn_factor=1.0, beta_fast, beta_slow
    );

    return Tensor(rope, x.shape());
}
