#include "nn/modules/normalization/LayerNorm.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

LayerNorm::LayerNorm(int64_t dim, float eps, bool elementwise_affine, bool bias)
    : eps_(eps), elementwise_affine_(elementwise_affine), bias_(bias)
{
    if (elementwise_affine) {
        modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));

        if (bias)
            modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
    }
}

Tensor LayerNorm::forward(Context& context, Tensor x) {
    x = Tensor(*context, ggml_norm(*context, *x, eps_), x.shape());

    if (elementwise_affine_) {
        auto weight = std::static_pointer_cast<Parameter>(modules["weight"]);

        x = x * weight->forward();

        if (bias_) {
            auto bias = std::static_pointer_cast<Parameter>(modules["bias"]);

            x = x + bias->forward();
        }
    }

    return x;
}
