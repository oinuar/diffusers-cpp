#include "diffusers/models/embeddings/Timesteps.hpp"
#include "ggml/Runtime.hpp"
#include <cmath>

Timesteps::Timesteps(
    int64_t num_channels,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale
)
    : num_channels(num_channels),
        flip_sin_to_cos(flip_sin_to_cos),
        downscale_freq_shift(downscale_freq_shift),
        scale(scale)
{}

Tensor Timesteps::forward(Runtime& runtime, Tensor timesteps) {
    const int64_t half_dim = num_channels / 2;

    auto exponent = Tensor::arange(*runtime.context(), 0, half_dim);

    exponent = exponent * (-std::log(10000.0f) / (half_dim - downscale_freq_shift));

    auto emb = exp(exponent);

    emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0);
    emb = emb * scale;

    auto sin_emb = sin(emb);
    auto cos_emb = cos(emb);

    Tensor out;

    if (flip_sin_to_cos)
        out = Tensor::cat({cos_emb, sin_emb}, -1);
    else
        out = Tensor::cat({sin_emb, cos_emb}, -1);

    if (num_channels % 2 == 1) {
        auto zeros = Tensor::zeros(*runtime.context(), {out.shape()[0], 1}, out.dtype());
        out = Tensor::cat({out, zeros}, -1);
    }

    return out;
}
