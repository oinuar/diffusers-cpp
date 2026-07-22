#pragma once

#include "ggml/Tensor.hpp"

class DiagonalGaussianDistribution {
public:
    DiagonalGaussianDistribution(
        ggml_context* ctx,
        Tensor parameters,
        bool deterministic = false
    );

    Tensor sample(
        ggml_context* ctx
    );

    Tensor mode() const {
        return mean_;
    }

private:
    Tensor mean_;
    Tensor logvar_;
    Tensor std_;

    bool deterministic_;
};