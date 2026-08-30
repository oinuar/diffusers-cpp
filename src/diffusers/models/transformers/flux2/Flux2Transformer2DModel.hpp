#pragma once

#include "nn/Module.hpp"
#include <filesystem>

class Backend;

class Flux2Transformer2DModel : public Module {
public:
    struct Config {
        static Config from_file(const std::filesystem::path& path);

        int64_t patch_size = 1;
        int64_t in_channels = 128;
        std::optional<int64_t> out_channels = std::nullopt;

        int64_t num_layers = 8;
        int64_t num_single_layers = 48;

        int64_t attention_head_dim = 128;
        int64_t num_attention_heads = 48;
        int64_t joint_attention_dim = 15360;
        int64_t timestep_guidance_channels = 256;

        float mlp_ratio = 3.0f;
        std::vector<int64_t> axes_dims_rope = {32, 32, 32, 32};

        int64_t rope_theta = 2000;
        float eps = 1e-6f;
        bool guidance_embeds = true;
    };

    explicit Flux2Transformer2DModel(const Config& config);

    Tensor forward(
        Context& context,
        Tensor hidden_states,
        Tensor encoder_hidden_states,
        Tensor timestep = {},
        Tensor img_ids = {},
        Tensor txt_ids = {},
        std::optional<Tensor> guidance = std::nullopt,
        // TODO: support KV cache
        //Tensor joint_attention_kwargs: dict[str, Any] | None = None,
        //Tensor return_dict: bool = True,
        //Tensor kv_cache: "Flux2KVCache | None" = None,
        //Tensor kv_cache_mode: str | None = None,
        int64_t num_ref_tokens = 0,
        float ref_fixed_timestep = 0.0
    );

private:
    Flux2Transformer2DModel(
        int64_t patch_size,
        int64_t in_channels,
        std::optional<int64_t> out_channels,
        int64_t num_layers,
        int64_t num_single_layers,
        int64_t attention_head_dim,
        int64_t num_attention_heads,
        int64_t joint_attention_dim,
        int64_t timestep_guidance_channels,
        float mlp_ratio,
        const std::vector<int64_t>& axes_dims_rope,
        int64_t rope_theta,
        float eps,
        bool guidance_embeds
    );
};
