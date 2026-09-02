#include "diffusers/models/autoencoders/vae/DiagonalGaussianDistribution.hpp"
#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include <cmath>

DiagonalGaussianDistribution::DiagonalGaussianDistribution(
    Scope& scope,
    Tensor parameters,
    bool deterministic
)
    : deterministic_(deterministic)
{
    auto chunks = parameters.chunk(2, 1);

    mean_ = chunks[0];
    logvar_ = chunks[1];
    logvar_ = logvar_.clamp(-30.0, 20.0).contiguous();

    if (deterministic) {
        std_ = Tensor::zeros(mean_.shape()).to(mean_.dtype());
        var_ = Tensor::zeros(mean_.shape()).to(mean_.dtype());
    }
    else {
        std_ = exp(0.5f * logvar_);
        var_ = exp(logvar_);
    }
}

Tensor DiagonalGaussianDistribution::sample(Scope& scope) {
    auto numel = mean_.numel();

    auto sample = scope.context().value<float>(mean_.shape(), [=](std::mt19937& rng) {
        std::vector<float> data(numel);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (auto& x : data)
            x = dist(rng);

        return data;
    });

    auto x = mean_ + std_ * sample;
    return x;
}
