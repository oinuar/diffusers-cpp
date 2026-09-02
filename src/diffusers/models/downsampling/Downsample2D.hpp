#pragma once

#include "nn/Module.hpp"

class Downsample2D : public Module {
public:
    Downsample2D(
        int64_t channels,
        bool use_conv = true,
        std::optional<int64_t> out_channels = std::nullopt,
        int64_t padding = 1
    );

    Tensor forward(Scope scope, Tensor hidden_states);

private:
    bool use_conv_;
    int64_t padding_;
};