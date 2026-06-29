#include "models/transformers/qwen3/Qwen3RotaryEmbedding.hpp"
#include "modules/Constant.hpp"

Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(const Qwen3Config& config)
    : dim_(config.head_dim.value_or(config.hidden_size / config.num_attention_heads)),
      base_(config.rope_parameters.rope_theta)
{
}

std::pair<Tensor, Tensor> Qwen3RotaryEmbedding::forward(ggml_context* ctx, Tensor x, Tensor position_ids) {
    auto inv_freq = 1.0f / (Tensor::arange(0, dim_, 2) / (float)dim_).pow(base_);

    auto inv_freq_expanded = inv_freq[{Tensor::Slice::none(), Tensor::Slice::all(), Tensor::Slice::none()}].to(GGML_TYPE_F32);
    inv_freq_expanded = inv_freq_expanded.expand({position_ids.shape()[0], -1, 1});

    auto position_ids_expanded = position_ids[{Tensor::Slice::all(), Tensor::Slice::none(), Tensor::Slice::all()}].to(GGML_TYPE_F32);

    auto freqs = ggml_mul_mat(ctx, position_ids_expanded, inv_freq_expanded);
    freqs = ggml_permute(ctx, freqs, 0, 2, 1, 3);

    auto emb = Tensor::cat({freqs, freqs}, -1);

    return {cos(emb).to(x.dtype()), sin(emb).to(x.dtype())};
}
