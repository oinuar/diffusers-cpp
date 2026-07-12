#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Tensor.hpp"

class LayerNorm : public Module {
public:
    // GGML supports only last dimension reduction norm
    LayerNorm(int64_t dim, float eps = 1e-5f, bool elementwise_affine = true, bool bias = true)
        : eps_(eps), elementwise_affine_(elementwise_affine), bias_(bias)
    {
        if (elementwise_affine) {
            modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));

            if (bias)
                modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
        }
    }

    Tensor forward(ggml_context* ctx, Tensor x) {
        x = Tensor(ctx, ggml_norm(ctx, *x, eps_), x.shape());

        if (elementwise_affine_) {
            auto weight = std::static_pointer_cast<Parameter>(modules["weight"]);

            x = x * weight->forward();
        }

        if (bias_) {
            auto bias = std::static_pointer_cast<Parameter>(modules["bias"]);

            x = x + bias->forward();
        }

        return x;
    }

private:
    float eps_;
    bool elementwise_affine_, bias_;
};
