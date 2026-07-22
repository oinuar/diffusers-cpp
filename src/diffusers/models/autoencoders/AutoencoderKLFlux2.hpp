#pragma once

#include "nn/Module.hpp"
#include "diffusers/models/autoencoders/vae/DiagonalGaussianDistribution.hpp"

class AutoencoderKLFlux2 : public Module {
public:
    AutoencoderKLFlux2(
        int64_t in_channels = 3,
        int64_t out_channels = 3,
        int64_t latent_channels = 32,
        const std::vector<std::string>& down_block_types = {"DownEncoderBlock2D"},
        const std::vector<std::string>& up_block_types = {"UpDecoderBlock2D"},
        const std::vector<int64_t>& block_out_channels = {128, 256, 512, 512},
        int64_t layers_per_block = 2,
        int64_t norm_num_groups = 32,
        const std::string& act_fn = "silu",
        bool mid_block_add_attention = true,
        bool use_quant_conv = true,
        bool use_post_quant_conv = true,
        float scaling_factor = 1.0f,
        std::optional<float> shift_factor = std::nullopt
    );

    DiagonalGaussianDistribution encode(
        ggml_context* ctx,
        Tensor sample
    );

    Tensor decode(
        ggml_context* ctx,
        Tensor latents
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor sample,
        bool sample_posterior
    );

    Tensor decode_latents(
        ggml_context* ctx,
        Tensor latents
    );

private:
    bool use_quant_conv_;
    bool use_post_quant_conv_;
    float scaling_factor_;
    std::optional<float> shift_factor_;
};