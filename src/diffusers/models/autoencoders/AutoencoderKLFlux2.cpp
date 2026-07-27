#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/autoencoders/vae/Encoder.hpp"
#include "diffusers/models/autoencoders/vae/Decoder.hpp"

AutoencoderKLFlux2::AutoencoderKLFlux2(
    int64_t in_channels,
    int64_t out_channels,
    const std::vector<int64_t>& block_out_channels,
    int64_t layers_per_block,
    int64_t latent_channels,
    int64_t norm_num_groups,
    int64_t sample_size,
    bool force_upcast,
    bool use_quant_conv,
    bool use_post_quant_conv,
    bool mid_block_add_attention,
    float batch_norm_eps,
    float batch_norm_momentum,
    std::tuple<int64_t, int64_t> patch_size
) {
    modules["encoder"] = std::make_shared<Encoder>(
        in_channels,
        latent_channels, // out_channels
        block_out_channels, 
        layers_per_block,
        norm_num_groups,
        true, // double_z
        mid_block_add_attention
    );

    modules["decoder"] = std::make_shared<Decoder>(
        latent_channels, // in_channels
        out_channels,
        block_out_channels,
        layers_per_block,
        norm_num_groups,
        mid_block_add_attention
    );

    if (use_quant_conv_)
        modules["quant_conv"] = std::make_shared<Conv2d>(
            2 * latent_channels, // in_channels
            2 * latent_channels, // out_channels
            1, // kernel_size
            1, // stride
            0 // padding
        );

    if (use_post_quant_conv_)
        modules["post_quant_conv"] = std::make_shared<Conv2d>(
            latent_channels, // in_channels
            latent_channels, // out_channels
            1 // kernel_size
        );
}

Tensor AutoencoderKLFlux2::forward(Runtime& runtime, Tensor sample, bool sample_posterior) {
    auto posterior = encode(runtime, sample);

    Tensor latents;

    if (sample_posterior)
        latents = posterior.sample(runtime);
    else
        latents = posterior.mode();

    return decode(runtime, latents);
}

DiagonalGaussianDistribution AutoencoderKLFlux2::encode(Runtime& runtime, Tensor sample) {
    auto encoder = std::static_pointer_cast<Encoder>(modules["encoder"]);

    sample = encoder->forward(runtime, sample);

    if (use_quant_conv_) {
        auto quant_conv = std::static_pointer_cast<Conv2d>(modules["quant_conv"]);

        sample = quant_conv->forward(runtime, sample);
    }

    return DiagonalGaussianDistribution(runtime, sample);
}

Tensor AutoencoderKLFlux2::decode(
    Runtime& runtime,
    Tensor latents
) {
    if (use_post_quant_conv_) {
        auto post_quant_conv =
            std::static_pointer_cast<Conv2d>(
                modules["post_quant_conv"]);

        latents = post_quant_conv->forward(runtime, latents);
    }

    auto decoder =
        std::static_pointer_cast<Decoder>(
            modules["decoder"]);

    return decoder->forward(runtime, latents);
}

Tensor AutoencoderKLFlux2::decode_latents(
    Runtime& runtime,
    Tensor latents
) {
    /*
        Diffusers:

        latents = latents / scaling_factor
    */

    latents = latents / scaling_factor_;


    /*
        Optional shift factor
    */

    if (shift_factor_) {
        latents = latents + shift_factor_.value();
    }


    auto image =
        decode(
            runtime,
            latents
        );


    /*
        image = image / 2 + 0.5
    */

    image = image * 0.5f;
    image = image + 0.5f;


    /*
        clamp(0, 1)

        Use Tensor clamp implementation here.
    */

    image = image.clamp(
        0.0f,
        1.0f
    );


    return image;
}
