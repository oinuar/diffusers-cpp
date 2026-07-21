#pragma once

#include <optional>
#include <string>

#include "nn/Module.hpp"


class ResnetBlock2D : public Module {
public:
    ResnetBlock2D(
        int64_t in_channels,
        int64_t out_channels,
        std::optional<int64_t> temb_channels = std::nullopt,
        float eps = 1e-6,
        const std::string& act_fn = "silu",
        float output_scale_factor = 1.0f,
        const std::string& time_scale_shift = "default",
        int64_t groups = 32
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> temb = std::nullopt
    );

private:
    float output_scale_factor_;
    bool use_shortcut_;
    bool use_temb_;
};