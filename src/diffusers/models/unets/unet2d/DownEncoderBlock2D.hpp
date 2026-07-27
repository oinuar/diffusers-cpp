#pragma once

#include <optional>
#include <string>

#include "nn/Module.hpp"

class DownEncoderBlock2D : public Module {
public:
    DownEncoderBlock2D(
        int64_t in_channels,
        int64_t out_channels,
        float dropout = 0.0f,
        int64_t num_layers = 1,
        float resnet_eps = 1e-6,
        int64_t resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0f,
        bool add_downsample = true,
        int64_t downsample_padding = 1
    );

    Tensor forward(Runtime& runtime, Tensor hidden_states);

private:
    bool add_downsample_;
};
