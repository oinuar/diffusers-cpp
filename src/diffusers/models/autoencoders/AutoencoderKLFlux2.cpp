#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/autoencoders/vae/Encoder.hpp"
#include "diffusers/models/autoencoders/vae/Decoder.hpp"

AutoencoderKLFlux2::AutoencoderKLFlux2(
    int64_t in_channels,
    int64_t out_channels,
    int64_t latent_channels,
    const std::vector<std::string>& down_block_types,
    const std::vector<std::string>& up_block_types,
    const std::vector<int64_t>& block_out_channels,
    int64_t layers_per_block,
    int64_t norm_num_groups,
    const std::string& act_fn,
    bool mid_block_add_attention,
    bool use_quant_conv,
    bool use_post_quant_conv,
    float scaling_factor,
    std::optional<float> shift_factor
)
    : use_quant_conv_(use_quant_conv),
      use_post_quant_conv_(use_post_quant_conv),
      scaling_factor_(scaling_factor),
      shift_factor_(shift_factor)
{
    modules["encoder"] = std::make_shared<Encoder>(
        in_channels,
        latent_channels,
        //down_block_types,
        block_out_channels,
        layers_per_block,
        norm_num_groups,
        //act_fn,
        /* double_z = */ true,
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

    if (use_quant_conv_) {
        modules["quant_conv"] = std::make_shared<Conv2d>(
            2 * latent_channels,
            2 * latent_channels,
            1,
            1,
            0
        );
    }

    if (use_post_quant_conv_)
        modules["post_quant_conv"] = std::make_shared<Conv2d>(latent_channels, latent_channels, 1);
}

DiagonalGaussianDistribution AutoencoderKLFlux2::encode(
    ggml_context* ctx,
    Tensor sample
) {
    auto encoder = std::static_pointer_cast<Encoder>(modules["encoder"]);

    sample = encoder->forward(ctx, sample);

    if (use_quant_conv_) {
        auto quant_conv = std::static_pointer_cast<Conv2d>(modules["quant_conv"]);

        sample = quant_conv->forward(ctx, sample);
    }

    return DiagonalGaussianDistribution(ctx, sample);
}

Tensor AutoencoderKLFlux2::decode(
    ggml_context* ctx,
    Tensor latents
) {
    if (use_post_quant_conv_) {
        auto post_quant_conv =
            std::static_pointer_cast<Conv2d>(
                modules["post_quant_conv"]);

        latents = post_quant_conv->forward(ctx, latents);
    }

    auto decoder =
        std::static_pointer_cast<Decoder>(
            modules["decoder"]);

    return decoder->forward(ctx, latents);
}

Tensor AutoencoderKLFlux2::forward(
    ggml_context* ctx,
    Tensor sample,
    bool sample_posterior
) {
    auto posterior = encode(ctx, sample);

    Tensor latents;

    if (sample_posterior) {
        latents = posterior.sample(ctx);
    } else {
        latents = posterior.mode();
    }

    return decode(ctx, latents);
}

Tensor AutoencoderKLFlux2::decode_latents(
    ggml_context* ctx,
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
            ctx,
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
