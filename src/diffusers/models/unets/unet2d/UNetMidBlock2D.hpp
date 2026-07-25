#pragma once

#include "nn/Module.hpp"

class UNetMidBlock2D : public Module {
public:
    UNetMidBlock2D(
        int64_t in_channels,
        std::optional<int64_t> temb_channels = std::nullopt,
        float dropout = 0.0f,
        int64_t num_layers = 1,
        float resnet_eps = 1e-6,
        int64_t resnet_groups = 32,
        std::optional<int64_t> attn_groups = std::nullopt,
        bool resnet_pre_norm = true,
        bool add_attention = true,
        int64_t attention_head_dim = 1,
        float output_scale_factor = 1.0f
    );

    Tensor forward(ggml_context* ctx, Tensor sample, std::optional<Tensor> temb = std::nullopt);

private:
    bool add_attention_;
};