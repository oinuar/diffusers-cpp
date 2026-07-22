#pragma once

#include <optional>
#include <string>

#include "nn/Module.hpp"


class ResnetBlock2D : public Module {
public:
    ResnetBlock2D(
        int64_t in_channels,
        std::optional<int64_t> out_channels = std::nullopt,
        std::optional<int64_t> conv_shortcut = std::nullopt,
        float dropout = 0.0f,
        std::optional<int64_t> temb_channels = std::nullopt,
        int64_t groups = 32,
        std::optional<int64_t> groups_out = std::nullopt,
        float eps = 1e-6f,
        const std::string& non_linearity = "swish",
        const std::string& time_embedding_norm = "default",
        int64_t kernel = 3,
        std::optional<int64_t> output_scale_factor = 1,
        bool use_in_shortcut = false,
        bool up = false,
        bool down = false,
        bool conv_shortcut_bias = true,
        int64_t conv_2d_out_channels = 0
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