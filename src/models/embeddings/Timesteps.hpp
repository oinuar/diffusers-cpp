#pragma once

#include <cmath>
#include <vector>

#include "modules/Module.hpp"

class Timesteps : public Module {
public:
    Timesteps(
        int64_t num_channels,
        bool flip_sin_to_cos,
        float downscale_freq_shift,
        float scale = 1.0f) :
        num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale)
    {
    }
    
    Tensor forward(ggml_context* ctx, Tensor timesteps) {
        auto t_emb = get_timestep_embedding(
            ctx,
            timesteps,
            num_channels_,
            flip_sin_to_cos_,
            downscale_freq_shift_,
            scale_
        );

        return t_emb;
    }

private:
    int64_t num_channels_;
    bool flip_sin_to_cos_;
    float downscale_freq_shift_;
    float scale_;

    static Tensor get_timestep_embedding(
        ggml_context* ctx,
        Tensor timesteps,
        int64_t embedding_dim,
        bool flip_sin_to_cos = false,
        float downscale_freq_shift = 1.0f,
        float scale = 1.0f,
        int64_t max_period = 10000) {

        int64_t N = timesteps.shape()[0];

        int64_t half_dim = embedding_dim / 2;

        // Create arange tensor: [0, 1, ..., half_dim-1]
        auto arange = Tensor::arange(ctx, 0, half_dim-1);

        // exponent = -log(max_period) * arange / (half_dim - downscale_freq_shift)
        float exponent_scale = -std::log(static_cast<float>(max_period)) / (static_cast<float>(half_dim) - downscale_freq_shift);
        auto emb = arange * exponent_scale;

        // emb = exp(exponent) → [half_dim]
        emb = exp(emb);

        // t_emb = timesteps[:, None] * emb[None, :] → [N, half_dim]
        auto t_emb_2d = timesteps.reshape({N, 1});   // [N, 1]
        auto emb_2d = emb.unsqueeze(0);             // [1, half_dim]
        t_emb_2d = t_emb_2d * emb_2d;               // broadcasts to [N, half_dim]

        // Scale embeddings
        t_emb_2d = scale * t_emb_2d;

        // Concatenate sin and cos → [N, embedding_dim]
        auto sin_emb = sin(t_emb_2d);
        auto cos_emb = cos(t_emb_2d);
        auto t_concat = Tensor::cat({sin_emb, cos_emb}, -1);

        // Flip sine and cosine if requested → swap first and second halves
        if (flip_sin_to_cos) {
            auto half1 = t_concat.narrow(-1, 0, half_dim);     // [:, :half_dim]
            auto half2 = t_concat.narrow(-1, half_dim, half_dim);  // [:, half_dim:]
            t_concat = Tensor::cat({half2, half1}, -1);
        }

        // Zero pad if embedding_dim is odd → [N, embedding_dim + 1]
        // TODO: check this: Python: torch.nn.functional.pad(emb, (0, 1, 0, 0))
        if (embedding_dim % 2 == 1) {
            auto padded = Tensor::zeros(ctx, {0, 1, 0, 0});
            t_concat = Tensor::cat({t_concat, padded}, -1);
        }

        return t_concat;
    }
};
