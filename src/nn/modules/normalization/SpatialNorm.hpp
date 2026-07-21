#pragma once

#include "nn/Module.hpp"

class SpatialNorm : public Module {
public:
    SpatialNorm(
        int64_t f_channels,
        int64_t zq_channels,
        float eps = 1e-6
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor x,
        Tensor zq
    );

private:
    float eps_;
};