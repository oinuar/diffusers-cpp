#pragma once

#include "nn/Module.hpp"
#include "diffusers/models/autoencoders/vae/DiagonalGaussianDistribution.hpp"

class AutoencoderKLFlux2 : public Module {
public:
    AutoencoderKLFlux2(
        int64_t in_channels = 3,
        int64_t out_channels = 3,
        const std::vector<int64_t>& block_out_channels = {128, 256, 512, 512},
        int64_t layers_per_block = 2,
        int64_t latent_channels = 32,
        int64_t norm_num_groups = 32,
        int64_t sample_size = 1024,
        bool force_upcast = true,
        bool use_quant_conv = true,
        bool use_post_quant_conv = true,
        bool mid_block_add_attention = true,
        float batch_norm_eps = 1e-4f,
        float batch_norm_momentum = 0.1f,
        std::tuple<int64_t, int64_t> patch_size = {2, 2}
    );

    DiagonalGaussianDistribution encode(
        Runtime& runtime,
        Tensor x
    );

    Tensor decode(
        Runtime& runtime,
        Tensor z
    );

    Tensor forward(
        Runtime& runtime,
        Tensor sample,
        bool sample_posterior
    );

    Tensor decode_latents(
        Runtime& runtime,
        Tensor latents
    );

private:
    bool use_quant_conv_;
    bool use_post_quant_conv_;
    float scaling_factor_;
    std::optional<float> shift_factor_;
};