#pragma once

#include "nn/Module.hpp"

class SpatialNorm : public Module {
public:
    SpatialNorm(
        int64_t f_channels,
        int64_t zq_channels
    );

    Tensor forward(ggml_context* ctx, Tensor f, Tensor zq);
};