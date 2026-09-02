#pragma once

#include <optional>

#include "nn/Module.hpp"

class UpDecoderBlock2D : public Module {
public:
    UpDecoderBlock2D(
        int64_t in_channels,
        int64_t out_channels,
        std::optional<int64_t> resolution_idx = std::nullopt,
        float dropout = 0.0f,
        int64_t num_layers = 1,
        float resnet_eps = 1e-6,
        int64_t resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0,
        bool add_upsample = true,
        std::optional<int64_t> temb_channels = std::nullopt
    );

    Tensor forward(Scope scope, Tensor hidden_states, std::optional<Tensor> temb = std::nullopt);

private:
    bool add_upsample_;
    int64_t num_layers_;
};