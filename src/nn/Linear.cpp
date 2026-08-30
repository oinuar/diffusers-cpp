#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

Linear::Linear(
    int64_t in_features,
    int64_t out_features,
    bool bias
) : bias_(bias)
{
    modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({out_features, in_features}));

    if (bias_)
        modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({out_features}));
}

Tensor Linear::forward(Context& context, Tensor x) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    // Weight is logically shaped [out_features, in_features] (PyTorch),
    // but stored in GGML's native reversed layout {in_features, out_features}.
    // Since ggml_mul_mat() already performs Aᵀ * B on its first operand,
    // explicitly transposing the weight would transpose it twice.
    auto y = ggml_mul_mat(*context, *weight, *x);

    if (bias_) {
        auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();

        y = ggml_add(*context, y, *bias);
    }

    Tensor::Shape shape = x.shape();
    shape[shape.rank() - 1] = weight.shape()[0];

    return Tensor(*context, y, shape);
}
