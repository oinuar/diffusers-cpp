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
) : use_quant_conv_(use_quant_conv),
    use_post_quant_conv_(use_post_quant_conv)
{
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
    auto x = sample;
    auto posterior = encode(runtime, x);

    Tensor z;

    if (sample_posterior)
        z = posterior.sample(runtime);
    else
        z = posterior.mode();
    
    auto dec = decode(runtime, z);

    return dec;
}

DiagonalGaussianDistribution AutoencoderKLFlux2::encode(Runtime& runtime, Tensor x) {
    auto encoder = std::static_pointer_cast<Encoder>(modules["encoder"]);

    auto enc = encoder->forward(runtime, x);

    if (use_quant_conv_) {
        auto quant_conv = std::static_pointer_cast<Conv2d>(modules["quant_conv"]);

        enc = quant_conv->forward(runtime, enc);
    }

    auto posterior = DiagonalGaussianDistribution(runtime, enc);

    return posterior;
}

Tensor AutoencoderKLFlux2::decode(Runtime& runtime, Tensor z) {
    if (use_post_quant_conv_) {
        auto post_quant_conv = std::static_pointer_cast<Conv2d>(modules["post_quant_conv"]);

        z = post_quant_conv->forward(runtime, z);
    }

    auto decoder = std::static_pointer_cast<Decoder>(modules["decoder"]);

    auto dec = decoder->forward(runtime, z);

    return dec;
}
