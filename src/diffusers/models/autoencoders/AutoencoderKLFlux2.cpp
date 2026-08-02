#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/autoencoders/vae/Encoder.hpp"
#include "diffusers/models/autoencoders/vae/Decoder.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

AutoencoderKLFlux2::AutoencoderKLFlux2(const AutoencoderKLFlux2::Config& config) : AutoencoderKLFlux2(
    config.in_channels,
    config.out_channels,
    config.block_out_channels,
    config.layers_per_block,
    config.latent_channels,
    config.norm_num_groups,
    config.sample_size,
    config.force_upcast,
    config.use_quant_conv,
    config.use_post_quant_conv,
    config.mid_block_add_attention,
    config.batch_norm_eps,
    config.batch_norm_momentum,
    config.patch_size
) {

}

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


template <typename T>
void read(const json& j, const char* key, T& value) {
    if (!j.contains(key) || j[key].is_null())
        return;

    value = j[key].get<T>();
}

template <typename T>
void read(const json& j, const char* key, std::optional<T>& value) {
    if (!j.contains(key))
        return;

    if (j[key].is_null())
        value.reset();
    else
        value = j[key].get<T>();
}

AutoencoderKLFlux2::Config AutoencoderKLFlux2::Config::from_file(const std::filesystem::path& path) {
    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Failed to open configuration file: " + path.string());

    json j;
    file >> j;

    Config cfg;

    read(j, "in_channels", cfg.in_channels);
    read(j, "out_channels", cfg.out_channels);

    read(j, "block_out_channels", cfg.block_out_channels);

    read(j, "layers_per_block", cfg.layers_per_block);
    read(j, "latent_channels", cfg.latent_channels);
    read(j, "norm_num_groups", cfg.norm_num_groups);
    read(j, "sample_size", cfg.sample_size);

    read(j, "force_upcast", cfg.force_upcast);
    read(j, "use_quant_conv", cfg.use_quant_conv);
    read(j, "use_post_quant_conv", cfg.use_post_quant_conv);
    read(j, "mid_block_add_attention", cfg.mid_block_add_attention);

    read(j, "batch_norm_eps", cfg.batch_norm_eps);
    read(j, "batch_norm_momentum", cfg.batch_norm_momentum);

    if (j.contains("patch_size") && !j["patch_size"].is_null()) {
        const auto& p = j["patch_size"];
        if (!p.is_array() || p.size() != 2) {
            throw std::runtime_error("patch_size must be an array of length 2");
        }

        cfg.patch_size = std::make_tuple(
            p[0].get<int64_t>(),
            p[1].get<int64_t>()
        );
    }

    return cfg;
}
