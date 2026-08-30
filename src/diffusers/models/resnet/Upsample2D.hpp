#pragma once

#include <optional>

#include "nn/Module.hpp"

class Upsample2D : public Module {
public:
    Upsample2D(
        int64_t channels,
        bool use_conv = false,
        std::optional<int64_t> out_channels = std::nullopt,
        bool use_conv_transpose = false
    );

    Tensor forward(
        Context& context,
        Tensor hidden_states
    );

private:
    bool use_conv_;
    bool use_conv_transpose_;
};