#include "diffusers/models/downsampling/Downsample2D.hpp"

#include "nn/modules/conv/Conv2d.hpp"

Downsample2D::Downsample2D(
    int64_t channels,
    bool use_conv,
    std::optional<int64_t> out_channels,
    int64_t padding
)
    : use_conv_(use_conv)
{
    const int64_t out_channels_ =
        out_channels.value_or(channels);

    if (use_conv_) {
        modules["conv"] =
            std::make_shared<Conv2d>(
                channels,
                out_channels_,
                3,
                2,
                padding
            );
    }
}

Tensor Downsample2D::forward(
    ggml_context* ctx,
    Tensor hidden_states
)
{
    if (use_conv_) {

        hidden_states =
            std::static_pointer_cast<Conv2d>(
                modules["conv"])
            ->forward(
                ctx,
                hidden_states
            );
    }

    return hidden_states;
}
