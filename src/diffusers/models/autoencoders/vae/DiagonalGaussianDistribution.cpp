#include "diffusers/models/autoencoders/vae/DiagonalGaussianDistribution.hpp"
#include "ggml/Runtime.hpp"
#include <cmath>

DiagonalGaussianDistribution::DiagonalGaussianDistribution(
    Runtime& runtime,
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
        std_ = Tensor::zeros(*runtime.context(), mean_.shape(), mean_.dtype());
        var_ = Tensor::zeros(*runtime.context(), mean_.shape(), mean_.dtype());
    }
    else {
        std_ = exp(0.5f * logvar_);
        var_ = exp(logvar_);
    }
}

Tensor DiagonalGaussianDistribution::sample(Runtime& runtime) {
    auto sample = runtime.create<float>(mean_.shape(), [](Tensor tensor, std::mt19937& rng) {
        std::vector<float> data(tensor.numel());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (auto& x : data)
            x = dist(rng);

        return data;
    });

    auto x = mean_ + std_ * sample;
    return x;
}
