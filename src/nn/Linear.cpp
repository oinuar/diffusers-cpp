#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Runtime.hpp"

Linear::Linear(
    int64_t in_features,
    int64_t out_features,
    bool bias
) : bias_(bias)
{
    // The weight (and bias) shard their output features across the devices of a
    // meta device, so the ggml_mul_mat in forward computes a slice of the output
    // rows per device with zero communication.
    auto weight = std::make_shared<Parameter>(Tensor::Shape({out_features, in_features}));
    weight->set_split(0);
    modules["weight"] = weight;

    if (bias_) {
        auto bias = std::make_shared<Parameter>(Tensor::Shape({out_features}));
        bias->set_split(0);
        modules["bias"] = bias;
    }
}

Tensor Linear::forward(Runtime& runtime, Tensor x) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    // Weight is logically shaped [out_features, in_features] (PyTorch),
    // but stored in GGML's native reversed layout {in_features, out_features}.
    // Since ggml_mul_mat() already performs Aᵀ * B on its first operand,
    // explicitly transposing the weight would transpose it twice.
    auto y = ggml_mul_mat(*runtime.context(), *weight, *x);

    if (bias_) {
        auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();

        y = ggml_add(*runtime.context(), y, *bias);
    }

    Tensor::Shape shape = x.shape();
    shape[shape.rank() - 1] = weight.shape()[0];

    return Tensor(*runtime.context(), y, shape);
}
