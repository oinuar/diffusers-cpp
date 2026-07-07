#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class LayerNorm : public Module {
public:
    LayerNorm(
        int64_t dim,
        float eps = 1e-5f,
        bool elementwise_affine = true,
        bool bias = true
    ) : 
        eps_(eps),
        elementwise_affine_(elementwise_affine),
        bias_(bias)
    {
        if (elementwise_affine) {
            modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));

            if (bias)
                modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
        }
    }

    Tensor forward(ggml_context* ctx, Tensor input) {
        auto weight = elementwise_affine_ ? std::static_pointer_cast<Parameter>(modules["weight"])->forward() : Tensor();
        auto bias = elementwise_affine_ && bias_ ? std::static_pointer_cast<Parameter>(modules["bias"])->forward() : Tensor();

        return Tensor(ctx, ggml_ext_layer_norm(ctx, *input, /*dim_,*/ *weight, *bias, eps_));
    }

private:
    float eps_;
    bool elementwise_affine_, bias_;

    static ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* w,
                                            ggml_tensor* b,
                                            float eps) {
        x = ggml_norm(ctx, x, eps);
        if (w != nullptr) {
            x = ggml_mul_inplace(ctx, x, w);
            if (b != nullptr) {
                x = ggml_add_inplace(ctx, x, b);
            }
        }
        return x;
    }
};
