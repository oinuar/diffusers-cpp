#pragma once

#include <optional>
#include <string>

#include "nn/Module.hpp"

class DownEncoderBlock2D : public Module {
public:
    DownEncoderBlock2D(
        int64_t num_layers,
        int64_t in_channels,
        int64_t out_channels,
        bool add_downsample,
        float resnet_eps = 1e-6,
        const std::string& resnet_act_fn = "silu",
        int64_t resnet_groups = 32,
        std::optional<int64_t> temb_channels = std::nullopt
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> temb = std::nullopt
    );

private:
    int64_t num_layers_;
    bool add_downsample_;
};