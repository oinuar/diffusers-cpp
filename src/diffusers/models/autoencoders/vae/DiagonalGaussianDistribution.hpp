#pragma once

#include "ggml/Tensor.hpp"

class Runtime;

class DiagonalGaussianDistribution {
public:
    DiagonalGaussianDistribution(Runtime& runtime, Tensor parameters, bool deterministic = false);

    Tensor sample(Runtime& runtime);

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
