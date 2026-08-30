#pragma once

#include "nn/Module.hpp"

class Conv2d : public Module {
public:
    Conv2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        bool bias = true
    );

    Tensor forward(Context& context, Tensor x);

public:
    int64_t stride_;
    int64_t padding_;
    bool bias_;
};
