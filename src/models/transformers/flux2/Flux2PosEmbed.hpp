#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include "modules/Module.hpp"
#include "ggml-backend.h"

class Flux2PosEmbed : public Module {
public:
    Flux2PosEmbed(int64_t theta, const std::vector<int64_t>& axes_dim)
        : theta_(theta), axes_dim_(axes_dim)
    {
    }

    std::pair<Tensor, Tensor> forward(ggml_context* ctx, Tensor ids) {
        // ids shape: [S, len(axes_dim)]  (int32 position indices)
        auto pos = ids;  // TODO: need .float() conversion for ggml int32 → float32

        std::vector<Tensor> cos_out, sin_out;

        for (size_t i = 0; i < axes_dim_.size(); ++i) {
            auto [cos, sin] = get_1d_rotary_pos_embed(
                ctx,
                static_cast<int64_t>(axes_dim_[i]),
                pos.narrow(-1, static_cast<int64_t>(i), 1).squeeze(),  // [S]
                static_cast<double>(theta_)
            );

            cos_out.push_back(cos);
            sin_out.push_back(sin);
        }

        auto freqs_cos = Tensor::cat(cos_out, -1);
        auto freqs_sin = Tensor::cat(sin_out, -1);

        return {freqs_cos, freqs_sin};
    }

private:
    int64_t theta_;
    std::vector<int64_t> axes_dim_;

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
        Tensor freqs = Tensor::empty<1>(ctx, GGML_TYPE_F32, {half_dim});
        ggml_backend_tensor_set(*freqs, freqs_data.data(), 0, ggml_nbytes(*freqs));

        // ── Step 2: Outer product pos ⊗ freqs → [S, half_dim] ─────────
        // ggml_mul_mat computes C = A^T @ B.
        // With pos reshaped to [1, S] and freqs to [half_dim, 1]:
        //   A^T is [S, 1], B is [half_dim, 1], inner dims (1==1) match.
        //   Result C is [S, half_dim] ✓
        Tensor pos_col = pos.reshape<2>({1, S});                    // [1, S]
        Tensor freqs_row = freqs.reshape<2>({half_dim, 1});         // [half_dim, 1]
        Tensor freqs_cis = Tensor(ctx, ggml_mul_mat(ctx, *pos_col, *freqs_row));  // [S, half_dim]

        // ── Step 3: cos / sin → [S, half_dim] ────────────────────────
        Tensor cos_raw = cos(freqs_cis);   // [S, half_dim]
        Tensor sin_raw = sin(freqs_cis);   // [S, half_dim]

        // ── Step 4: repeat_interleave(2, dim=1) using reshape+concat ──
        // Each element repeated twice consecutively along the last dimension.
        // Strategy: [S, half_dim] → [S*half_dim, 1] → concat w/ self → [S*half_dim, 2]
        //            → reshape to [S, dim]. Row-major layout ensures each element is duplicated.
        Tensor flat = cos_raw.reshape<2>({S * half_dim, 1});          // [S*h, 1]
        Tensor cos_rep = Tensor::cat({flat, flat}, -1);         // [S*h, 2]
        Tensor sin_rep = Tensor::cat({flat, flat}, -1);         // [S*h, 2]

        return {cos_rep.reshape<2>({S, dim}), sin_rep.reshape<2>({S, dim})};  // [S, D], [S, D]
    }
};
