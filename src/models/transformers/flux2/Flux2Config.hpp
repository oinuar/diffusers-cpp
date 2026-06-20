#pragma once

struct Flux2Config {
    int64_t patch_size            = 1;
    int64_t in_channels           = 128;
    int64_t out_channels          = 128;
    int64_t inner_dim             = 6144;   // num_attention_heads * attention_head_dim
    int64_t num_layers            = 8;      // double-stream DiT blocks
    int64_t num_single_layers     = 48;     // single-stream DiT blocks
    int64_t attention_head_dim    = 128;
    int64_t num_attention_heads   = 48;
    int64_t joint_attention_dim   = 15360;
    int64_t timestep_guidance_channels = 256;
    float mlp_ratio               = 3.0f;
    std::vector<int> axes_dims_rope{32, 32, 32, 32};
    int rope_theta                = 2000;
    float eps                     = 1e-6f;
    bool guidance_embeds          = true;

    static Flux2Config from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
        Flux2Config config;

        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }

            // Count double-stream blocks: "transformer_blocks.N.xxx"
            size_t db = name.find("transformer_blocks.");
            if (db != std::string::npos) {
                std::string rest = name.substr(db + strlen("transformer_blocks."));
                int block_idx    = atoi(rest.c_str());
                if (block_idx + 1 > (int)config.num_layers) {
                    config.num_layers = block_idx + 1;
                }
            }

            // Count single-stream blocks: "single_transformer_blocks.N.xxx"
            size_t sb = name.find("single_transformer_blocks.");
            if (sb != std::string::npos) {
                std::string rest = name.substr(sb + strlen("single_transformer_blocks."));
                int block_idx    = atoi(rest.c_str());
                if (block_idx + 1 > (int)config.num_single_layers) {
                    config.num_single_layers = block_idx + 1;
                }
            }

            // Get dimensions from weight shapes
            if (ends_with(name, ".x_embedder.weight")) {
                config.in_channels = tensor_storage.ne[0];
                config.inner_dim   = tensor_storage.ne[1];
            }
            if (ends_with(name, ".context_embedder.weight")) {
                config.joint_attention_dim = tensor_storage.ne[0];
            }
            if (ends_with(name, ".double_stream_modulation_img.linear.weight")) {
                config.inner_dim = tensor_storage.ne[1] / 6;  // dim * 3 * 2 sets
            }
            if (ends_with(name, ".single_stream_modulation.linear.weight")) {
                config.inner_dim = tensor_storage.ne[1] / 3;   // dim * 3 * 1 set
            }

            // Detect guidance embedding from weight presence
            if (name.find("time_guidance_embed.guidance_embedder") != std::string::npos) {
                config.guidance_embeds = true;
            } else if (name.find("time_guidance_embed.timestep_embedder") != std::string::npos) {
                config.guidance_embeds = false;  // distilled variant
            }
        }

        if (config.inner_dim > 0) {
            config.num_attention_heads = config.inner_dim / config.attention_head_dim;
        }

        LOG_DEBUG("flux2: num_layers=%" PRId64 ", num_single_layers=%" PRId64 ", inner_dim=%" PRId64
                    ", attention_head_dim=%" PRId64 ", num_attention_heads=%" PRId64,
                    config.num_layers, config.num_single_layers, config.inner_dim,
                    config.attention_head_dim, config.num_attention_heads);

        return config;
    }
};