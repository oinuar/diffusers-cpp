#pragma once

#include "nn/Module.hpp"

class Timesteps : public Module {
public:
    Timesteps(
        int64_t num_channels,
        bool flip_sin_to_cos,
        float downscale_freq_shift = 1.0f,
        float scale = 1.0f
    );

    Tensor forward(Scope scope, Tensor timesteps);

private:
    int64_t num_channels;
    bool flip_sin_to_cos;
    float downscale_freq_shift;
    float scale;
};
