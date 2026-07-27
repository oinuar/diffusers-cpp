#pragma once

#include "nn/Module.hpp"
#include "ggml/Tensor.hpp"

class LayerNorm : public Module {
public:
    // GGML supports only last dimension reduction norm
    LayerNorm(int64_t dim, float eps = 1e-5f, bool elementwise_affine = true, bool bias = true);

    Tensor forward(Runtime& runtime, Tensor x);

private:
    float eps_;
    bool elementwise_affine_, bias_;
};
