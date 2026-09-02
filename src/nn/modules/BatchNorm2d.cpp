#include "nn/modules/BatchNorm2d.hpp"
#include "nn/Parameter.hpp"

BatchNorm2d::BatchNorm2d(int64_t num_features, float eps, float momentum)
    : eps_(eps),
      momentum_(momentum)
{
    // affine=False:
    //   no weight / bias
    //
    // track_running_stats=True:
    //   running_mean, running_var, num_batches_tracked

    modules["running_mean"] = std::make_shared<Parameter>(Tensor::Shape({num_features}));
    modules["running_var"] = std::make_shared<Parameter>(Tensor::Shape({num_features}));
    modules["num_batches_tracked"] = std::make_shared<Parameter>(Tensor::Shape({1}));
}

Tensor BatchNorm2d::forward(Scope scope, Tensor x) {
    auto mean = std::static_pointer_cast<Parameter>(modules["running_mean"])->forward();
    auto var = std::static_pointer_cast<Parameter>(modules["running_var"])->forward();

    mean = mean.reshape({1, mean.shape()[0], 1, 1});
    var = var.reshape({1, var.shape()[0], 1, 1});

    return (x - mean) / sqrt(var + eps_);
}

const Parameter& BatchNorm2d::running_mean() const {
    return *std::static_pointer_cast<const Parameter>(modules.at("running_mean"));
}

const Parameter& BatchNorm2d::running_var() const {
    return *std::static_pointer_cast<const Parameter>(modules.at("running_var"));
}