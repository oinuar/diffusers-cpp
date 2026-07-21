#pragma once

#include "nn/Module.hpp"

class GroupNorm : public Module {
public:
    GroupNorm(
        int64_t num_groups,
        int64_t num_channels,
        float eps = 1e-5,
        bool affine = true
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor x
    );

private:
    int64_t num_groups_;
    int64_t num_channels_;
    float eps_;
    bool affine_;
};