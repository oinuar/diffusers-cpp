#include "models/embeddings/Qwen3RotaryEmbedding.hpp"

Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(const Qwen3Config& config) {

}

std::pair<Tensor, Tensor> Qwen3RotaryEmbedding::forward(ggml_context* ctx, Tensor x, std::vector<int64_t> position_ids) {
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
}

/*std::pair<Tensor, Tensor> Qwen3RotaryEmbedding::forward(
    ggml_context* ctx, Tensor hidden_states, Tensor position_ids)
{
    const int64_t seq_len = hidden_states.shape()[1]; // [B, S, H] or similar

    // Compute inv_freq lazily (once per embedding instance).
    if (!inv_freq_computed_) {
        // inv_freq = 1.0 / (base ^ (arange(0, dim, 2) / dim))
        auto arange = Tensor::arange(ctx, 0.0f, static_cast<float>(dim_), 2.0f);
        auto div    = arange * (1.0f / static_cast<float>(dim_));
        auto freqs  = div * (1.0f / std::log(rope_theta_)); // Actually: log(base) factor

        // Store inv_freq as a Parameter-like tensor — we'd need to register it properly.
        // For now, just compute it fresh each time (will be optimized later with caching).
        inv_freq_computed_ = true;
    }

    // Compute freqs from position_ids and inv_freq.
    // freqs = inv_freq @ position_ids → [seq_len, dim/2]
    // Then cos/sin with scaling factor.

    // For now, return placeholder — the full implementation requires:
    // 1. Storing inv_freq as a member tensor (needs ggml_context management)
    // 2. Computing freqs from position_ids expansion and matmul
    // 3. Applying cos/sin with optional scaling factor

    // Simplified: compute cos/sin directly from position_ids and dim_.
    auto pos_expanded = position_ids.unsqueeze(1);        // [B, 1, S]
    auto inv_freq     = Tensor::arange(ctx, 0.0f, static_cast<float>(dim_ / 2), 2.0f)
                            * (1.0f / std::log(rope_theta_));

    // freqs: [B, S, dim/2] via broadcasting
    auto freqs = pos_expanded * inv_freq.unsqueeze(0);     // [B, 1, S] × [dim/2] → broadcast

    auto cos = cos(freqs);
    auto sin = sin(freqs);

    return {cos, sin};
}
*/