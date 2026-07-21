#pragma once

#include "nn/Module.hpp"

class Upsample2D : public Module {
public:
    Upsample2D(
        int64_t channels,
        bool use_conv = true
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states
    );

private:
    bool use_conv_;
};