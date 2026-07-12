#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class RMSNorm : public Module {
public:
    // GGML supports only last dimension reduction norm
    RMSNorm(int64_t dim, float eps = 1e-5f, bool elementwise_affine = true)
        : eps_(eps), elementwise_affine_(elementwise_affine)
    {
        if (elementwise_affine)
            modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
    }

    Tensor forward(ggml_context* ctx, Tensor x) {
        x = Tensor(ctx, ggml_rms_norm(ctx, *x, eps_), x.shape());

        if (elementwise_affine_) {
            auto weight = std::static_pointer_cast<Parameter>(modules["weight"]);

            x = x * weight->forward();
        }

        return x;
    }

private:
    float eps_;
    bool elementwise_affine_;
};
