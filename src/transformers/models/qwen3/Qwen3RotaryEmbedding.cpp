#include "transformers/models/qwen3/Qwen3RotaryEmbedding.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "ggml/Runtime.hpp"

Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(const Qwen3Config& config)
    : head_dim(config.head_dim), rope_theta(config.rope_theta)
{
}

Tensor Qwen3RotaryEmbedding::forward(
    Runtime& runtime,
    Tensor x,
    Tensor position_ids
) {
    // GGML RoPE expects position IDs to be 32b integers.
    if (position_ids.dtype() != GGML_TYPE_I32)
        position_ids = position_ids.to(GGML_TYPE_I32);

    // The input x from Python has logical shape [batch, heads, seq_len, head_dim].
    // In GGML (column-major), this is stored as ne = [head_dim, seq_len, heads, batch].
    // However, ggml_rope_ext asserts that ne[2] == position_ids->ne[0] (which is seq_len).
    // To satisfy this, we must swap the 'seq_len' (dim 2) and 'heads' (dim 1) dimensions.
    // Logical permutation {0, 2, 1, 3} changes the shape to [batch, seq_len, heads, head_dim],
    // which in GGML becomes ne = [head_dim, heads, seq_len, batch], correctly putting seq_len at ne[2].
    x = x.permute({0, 2, 1, 3}).contiguous();

    auto rope = ggml_rope_ext(
        *runtime.context(),
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

    // Wrap the ggml_tensor result back into our Tensor abstraction.
    // Its current logical shape is [batch, seq_len, heads, head_dim].
    x = Tensor(*runtime.context(), rope, x.shape());

    // Permute back to the original logical shape [batch, heads, seq_len, head_dim]
    return x.permute({0, 2, 1, 3});
}
