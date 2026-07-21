#pragma once

#include "nn/Module.hpp"

class UNetMidBlock2D : public Module {
public:
    UNetMidBlock2D(
        int64_t in_channels,
        float resnet_eps = 1e-6,
        const std::string& resnet_act_fn = "silu",
        float output_scale_factor = 1.0f,
        const std::string& resnet_time_scale_shift = "default",
        int64_t attention_head_dim = 1,
        int64_t resnet_groups = 32,
        std::optional<int64_t> temb_channels = std::nullopt,
        bool add_attention = true
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor sample,
        std::optional<Tensor> temb = std::nullopt
    );

private:
    bool add_attention_;
};