#pragma once

#include "modules/Module.hpp"

class Linear : public Module {
public:
    Linear(
        const int64_t& in_features,
        const int64_t& out_features,
        const bool& bias = true
    ) : 
        in_features_(in_features),
        out_features_(out_features)
    {
        params["weight"] = Parameter([=](ggml_context* ctx) { return Tensor::empty<2>(ctx, GGML_TYPE_F32, {in_features_, out_features_}); });

        if (bias)
            params["bias"] = Parameter([=](ggml_context* ctx) { return Tensor::empty<1>(ctx, GGML_TYPE_F32, {out_features_}); });
    }
    
    Tensor forward(ggml_context* ctx, const Tensor& x) {
        auto weight = params["weight"];
        auto bias = params["bias"];

        return Tensor(ctx, ggml_ext_linear(ctx, *x, **weight, **bias));
    }

private:
    int64_t in_features_;
    int64_t out_features_;
    bool bias_;

    static bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
        return x->nb[0] == ggml_type_size(x->type) &&
            x->nb[2] == x->nb[1] * x->ne[1] &&
            x->nb[3] == x->nb[2] * x->ne[2];
    }

    static ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                                                ggml_tensor* x,
                                                float factor,
                                                bool inplace = false) {
        if (!ggml_ext_is_padded_1d(x)) {
            x = ggml_cont(ctx, x);
        }
        if (inplace) {
            x = ggml_scale_inplace(ctx, x, factor);
        } else {
            x = ggml_scale(ctx, x, factor);
        }
        return x;
    }

    static ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* w,
                                        ggml_tensor* b,
                                        bool force_prec_f32 = false,
                                        float scale         = 1.f) {
        if (scale != 1.f) {
            x = ggml_ext_scale(ctx, x, scale);
        }
        if (x->ne[2] * x->ne[3] > 1024) {
            // workaround: avoid ggml cuda error
            int64_t ne2 = x->ne[2];
            int64_t ne3 = x->ne[3];
            x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
            x           = ggml_mul_mat(ctx, w, x);
            if (force_prec_f32) {
                ggml_mul_mat_set_prec(x, GGML_PREC_F32);
            }
            x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
        } else {
            x = ggml_mul_mat(ctx, w, x);
            if (force_prec_f32) {
                ggml_mul_mat_set_prec(x, GGML_PREC_F32);
            }
        }
        if (scale != 1.f) {
            x = ggml_ext_scale(ctx, x, 1.f / scale);
        }
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
        return x;
    }
};
