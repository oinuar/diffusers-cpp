#pragma once

#include "nn/Module.hpp"

class Parameter;

class BatchNorm2d : public Module {
public:
    BatchNorm2d(
        int64_t num_features,
        float eps = 1e-5f,
        float momentum = 0.1f
    );

    Tensor forward(Scope scope, Tensor x);

    std::shared_ptr<Parameter> running_mean() const;

    std::shared_ptr<Parameter> running_var() const;

private:
    float eps_;
    float momentum_;
};
