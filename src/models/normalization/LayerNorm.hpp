#pragma once

#include "modules/Module.hpp"

class LayerNorm : public Module {
public:
    LayerNorm(
        int64_t dim,
        float eps = 1e-5f,
        bool elementwise_affine = true,
        bool bias = true
    ) : 
        eps_(eps)
    {
        if (elementwise_affine) {
            params["weight"] = Parameter([=](ggml_context* ctx) { return Tensor::ones<1>(ctx, {dim}); });

            if (bias)
                params["bias"] = Parameter([=](ggml_context* ctx) { return Tensor::zeros<1>(ctx, {dim}); });
        }
    }

    Tensor forward(ggml_context* ctx, Tensor input) {
        auto weight = params["weight"];
        auto bias = params["bias"];

        return Tensor(ctx, ggml_ext_layer_norm(ctx, *input, /*dim_,*/ **weight, **bias, eps_));
    }

private:
    float eps_;

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
