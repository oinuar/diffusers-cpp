#pragma once

#include "nn/Module.hpp"

class Downsample2D : public Module {
public:
    Downsample2D(
        int64_t channels,
        int64_t out_channels = -1,
        bool use_conv = true,
        int64_t padding = 1
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states
    );

private:
    bool use_conv_;
};