#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"

Linear::Linear(
    int64_t in_features,
    int64_t out_features,
    bool bias
) : bias_(bias)
{
    modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({in_features, out_features}));

    if (bias_)
        modules["bias"] = std::make_shared<Parameter>(Tensor::Shape({out_features}));
}

Tensor Linear::forward(ggml_context* ctx, Tensor x) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    auto wt = ggml_transpose(ctx, *weight);
    wt = ggml_cont(ctx, wt);

    auto y = ggml_mul_mat(ctx, wt, *x);

    if (bias_) {
        auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();

        y = ggml_add(ctx, y, *bias);
    }

    return Tensor(ctx, y);
}
