#pragma once

#include <optional>

#include "nn/Module.hpp"

class UpDecoderBlock2D : public Module {
public:
    UpDecoderBlock2D(
        int64_t num_layers,
        int64_t in_channels,
        int64_t out_channels,
        int64_t prev_output_channel,
        bool add_upsample,
        float resnet_eps = 1e-6,
        const std::string& resnet_act_fn = "silu",
        int64_t resnet_groups = 32,
        int64_t attention_head_dim = 1,
        std::optional<int64_t> temb_channels = std::nullopt,
        const std::string& resnet_time_scale_shift = "default"
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> temb = std::nullopt
    );

private:
    bool add_upsample_;
    int64_t num_layers_;
};