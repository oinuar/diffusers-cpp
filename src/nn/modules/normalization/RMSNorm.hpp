#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class RMSNorm : public Module {
public:
    // GGML supports only last dimension reduction norm
    RMSNorm(int64_t dim, float eps = 1e-5f, bool elementwise_affine = true);

    Tensor forward(Runtime& runtime, Tensor x);

private:
    float eps_;
    bool elementwise_affine_;
};
