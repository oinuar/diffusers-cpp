#include "nn/modules/normalization/RMSNorm.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

RMSNorm::RMSNorm(int64_t dim, float eps, bool elementwise_affine)
    : eps_(eps), elementwise_affine_(elementwise_affine)
{
    if (elementwise_affine)
        modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
}

Tensor RMSNorm::forward(Scope scope, Tensor x) {
    x = Tensor(ggml_rms_norm(*scope.context(), *x, eps_), x.shape());

    if (elementwise_affine_) {
        auto weight = std::static_pointer_cast<Parameter>(modules["weight"]);

        x = x * weight->forward();
    }

    return x;
}
