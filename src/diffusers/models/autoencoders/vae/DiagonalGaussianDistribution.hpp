#pragma once

#include "ggml/Tensor.hpp"

class Context;

class DiagonalGaussianDistribution {
public:
    DiagonalGaussianDistribution(Context& context, Tensor parameters, bool deterministic = false);

    Tensor sample(Context& context);

    Tensor mode() const {
        return mean_;
    }

private:
    Tensor mean_;
    Tensor logvar_;
    Tensor std_;
    Tensor var_;

    bool deterministic_;
};
