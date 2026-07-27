#pragma once

#include <optional>
#include <string>

#include "nn/Module.hpp"

template <class NonLinearity>
class ResnetBlock2D : public Module {
public:
    ResnetBlock2D(
        int64_t in_channels,
        std::optional<int64_t> out_channels = std::nullopt,
        std::optional<int64_t> conv_shortcut = std::nullopt,
        float dropout = 0.0f,
        std::optional<int64_t> temb_channels = 512,
        int64_t groups = 32,
        std::optional<int64_t> groups_out = std::nullopt,
        bool pre_norm = true,
        float eps = 1e-6f,
        bool skip_time_act = false,
        int64_t kernel_size = 3,
        float output_scale_factor = 1.0f,
        std::optional<bool> use_in_shortcut = std::nullopt,
        bool up = false,
        bool down = false,
        bool conv_shortcut_bias = true,
        std::optional<int64_t> conv_2d_out_channels = std::nullopt
    );

    Tensor forward(Runtime& runtime, Tensor hidden_states, std::optional<Tensor> temb = std::nullopt);

private:
    float output_scale_factor_;
    bool use_in_shortcut_;
    bool use_temb_;
};

#include "diffusers/models/resnet/ResnetBlock2D.inl"