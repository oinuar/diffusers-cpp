#pragma once

#include "modules/Module.hpp"
#include "modules/Parameter.hpp"

class Linear : public Module {
public:
    Linear(
        int64_t in_features,
        int64_t out_features,
        bool bias = true
    ) : bias_(bias)
    {
        modules["weight"] = std::shared_ptr<Module>(new Parameter<2>({in_features, out_features}));

        if (bias_)
            modules["bias"] = std::shared_ptr<Module>(new Parameter<1>({out_features}));
    }
    
    Tensor forward(ggml_context* ctx, Tensor x) {
        auto weight = std::static_pointer_cast<Parameter<2>>(modules["weight"])->forward();
        auto bias = bias_ ? std::static_pointer_cast<Parameter<1>>(modules["bias"])->forward() : Tensor();

        return Tensor(ctx, ggml_ext_linear(ctx, *x, *weight, *bias));
    }

private:
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
