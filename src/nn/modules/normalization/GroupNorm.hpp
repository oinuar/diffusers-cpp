#pragma once

#include "nn/Module.hpp"

class GroupNorm : public Module {
public:
    GroupNorm(
        int64_t num_groups,
        int64_t num_channels,
        float eps = 1e-5,
        bool affine = true,
        bool bias = true
    );

    Tensor forward(Context& context, Tensor input);

private:
    int64_t num_groups_;
    float eps_;
    bool affine_;
    bool bias_;
};