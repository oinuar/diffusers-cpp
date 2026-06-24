#pragma once

#include "Tensor.hpp"
#include "ggml-backend.h"

/**
 * Apply rotary embeddings to query/key tensor x using precomputed freqs_cis.
 * Only use_real=true mode is implemented (Flux, CogVideoX, Hunyuan-DiT).
 */
static Tensor apply_rotary_emb(
    ggml_context* ctx,
    Tensor x,
    Tensor cos,        // [S, D] or [S/2, D] depending on upstream freqs computation
    Tensor sin,        // same shape as cos
    int sequence_dim = 2)     // 1 → broadcast dim=1 ([B,S,H,D]); 2 → dim=2 ([B,H,S,D], Flux default)
{
    if (sequence_dim == 2) {
        // x is [B, H, S, D] — broadcast cos/sin to [1, 1, S, D]
        auto s = cos.shape();
        Tensor cos_b = cos.reshape({1, 1, s[0], s[1]});
        Tensor sin_b = sin.reshape({1, 1, s[0], s[1]});

        // x_real/imag: reshape [B,H,S,D] → [B,H,S,D//2*2], chunk last dim → each [B,H,S,D//2]
        auto x_shape = x.shape();
        Tensor x_pairs = x.reshape({x_shape[0], x_shape[1], x_shape[2],
                                    (x_shape[3] / 2) * 2});
        auto parts = x_pairs.chunk(2, 3); // dim=3 in ggml order = last logical dim

        // rotated = [-imag, real] → [B,H,S,D//2]
        Tensor x_rotated = Tensor::cat({-parts[1], parts[0]}, 3);

        return x * cos_b + x_rotated * sin_b;
    } else if (sequence_dim == 1) {
        // x is [B, S, H, D] — broadcast cos/sin to [1, S, 1, D]
        auto s = cos.shape();
        Tensor cos_b = cos.reshape({1, s[0], 1, s[1]});
        Tensor sin_b = sin.reshape({1, s[0], 1, s[1]});

        auto x_shape = x.shape();
        Tensor x_pairs = x.reshape({x_shape[0], x_shape[2], x_shape[3] / 2 * 2,
                                    1}); // reorder to [B,D//2*2,H,1] for ggml column-major → [B,H,S,D] logical
        auto parts = x_pairs.chunk(2, 1);

        Tensor x_rotated = Tensor::cat({-parts[1], parts[0]}, 1);

        // reshape back to [B,S,H,D] logical
        return (x * cos_b + x_rotated * sin_b).reshape({x_shape[0], x_shape[2],
                                                        x_shape[3] / 2,
                                                        x_shape[1]});
    }

    throw std::invalid_argument("apply_rotary_emb: sequence_dim must be 1 or 2");
}

/**
 * Precompute 1D rotary positional embedding frequencies (cos & sin).
 *
 * Matches the PyTorch get_1d_rotary_pos_embed with use_real=True and
 * repeat_interleave_real=True (Flux / Hunyuan-Dit / CogVideoX mode).
 *
 * freqs = 1.0 / (theta ^ (arange(0, dim, 2) / dim))       // [dim/2]
 * freqs_cis = outer(pos, freqs)                            // [S, dim/2]
 * cos_out = cos(freqs_cis).repeat_interleave(2, dim=1)    // [S, dim]
 * sin_out = sin(freqs_cis).repeat_interleave(2, dim=1)    // [S, dim]
 *
 * Note: PyTorch uses float64 for intermediate freqs computation then casts
 * to float32. ggml only supports float32, so we compute entirely in f32.
 */
static std::pair<Tensor, Tensor> get_1d_rotary_pos_embed(
    ggml_context* ctx,
    int64_t dim,
    Tensor pos,          // [S], integer positions (used as float in multiplication)
    double theta)
{
    if (dim % 2 != 0) {
        throw std::runtime_error("get_1d_rotary_pos_embed: dim must be even");
    }

    int64_t S = pos.numel();
    int64_t half_dim = dim / 2;

    // ── Step 1: Compute freqs on CPU ──────────────────────────────
    // freqs[i] = 1 / theta^(i / dim) for i in [0, 1, ..., half_dim-1]
    // Equivalent: exp(- (i / dim) * log(theta))
    const double log_theta = std::log(theta);
    std::vector<float> freqs_data(half_dim);
    for (int64_t i = 0; i < half_dim; ++i) {
        double exponent = -(static_cast<double>(i) / static_cast<double>(dim)) * log_theta;
        freqs_data[i] = static_cast<float>(std::exp(exponent));
    }

    // Create a constant tensor from CPU data.
    // TODO: this wont' work: we need to use a custom Module and initialize it with visitor pattern
    Tensor freqs = Tensor::empty(ctx, GGML_TYPE_F32, {half_dim});
    ggml_backend_tensor_set(*freqs, freqs_data.data(), 0, ggml_nbytes(*freqs));

    // ── Step 2: Outer product pos ⊗ freqs → [S, half_dim] ─────────
    // ggml_mul_mat computes C = A^T @ B.
    // With pos reshaped to [1, S] and freqs to [half_dim, 1]:
    //   A^T is [S, 1], B is [half_dim, 1], inner dims (1==1) match.
    //   Result C is [S, half_dim] ✓
    Tensor pos_col = pos.reshape({1, S});                    // [1, S]
    Tensor freqs_row = freqs.reshape({half_dim, 1});         // [half_dim, 1]
    Tensor freqs_cis = Tensor(ctx, ggml_mul_mat(ctx, *pos_col, *freqs_row));  // [S, half_dim]

    // ── Step 3: cos / sin → [S, half_dim] ────────────────────────
    Tensor cos_raw = cos(freqs_cis);   // [S, half_dim]
    Tensor sin_raw = sin(freqs_cis);   // [S, half_dim]

    // ── Step 4: repeat_interleave(2, dim=1) using reshape+concat ──
    // Each element repeated twice consecutively along the last dimension.
    // Strategy: [S, half_dim] → [S*half_dim, 1] → concat w/ self → [S*half_dim, 2]
    //            → reshape to [S, dim]. Row-major layout ensures each element is duplicated.
    Tensor flat = cos_raw.reshape({S * half_dim, 1});          // [S*h, 1]
    Tensor cos_rep = Tensor::cat({flat, flat}, -1);         // [S*h, 2]
    Tensor sin_rep = Tensor::cat({flat, flat}, -1);         // [S*h, 2]

    return {cos_rep.reshape({S, dim}), sin_rep.reshape({S, dim})};  // [S, D], [S, D]
}
