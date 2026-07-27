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
    /*
        Encoder output:

        [B, 2*C, H, W]

        Split into:

        mean   [B,C,H,W]
        logvar [B,C,H,W]
    */

    auto channels = parameters.shape()[1];

    auto chunks =
        parameters.chunk(
            2,
            1
        );

    mean_ = chunks[0];

    logvar_ =
        chunks[1].clamp(
            -30.0f,
            20.0f
        );


    if (!deterministic_) {
        std_ =
            exp(
                logvar_ * 0.5f
            );
    }
    else {
        std_ = Tensor::zeros(*runtime.context(), mean_.shape(), mean_.dtype());
    }
}

Tensor DiagonalGaussianDistribution::sample(
    Runtime& runtime
)
{
    if (deterministic_)
        return mean_;


    //auto noise = Tensor::zeros(ctx, mean_.shape(), mean_.dtype()).rand();
    auto noise = 0.0f; // TODO: this needs to be solved

    return mean_ + std_ * noise;
}
