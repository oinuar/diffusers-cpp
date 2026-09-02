#pragma once

#include "ggml/Tensor.hpp"

class Scope;

class DiagonalGaussianDistribution {
public:
    DiagonalGaussianDistribution(Scope& scope, Tensor parameters, bool deterministic = false);

    Tensor sample(Scope& scope);

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
