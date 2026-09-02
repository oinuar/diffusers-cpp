#include "diffusers/models/resnet/Upsample2D.hpp"
#include "ggml/Context.hpp"
#include "nn/modules/conv/Conv2d.hpp"

Upsample2D::Upsample2D(
    int64_t channels,
    bool use_conv,
    std::optional<int64_t> out_channels,
    bool use_conv_transpose
)
    : use_conv_(use_conv),
      use_conv_transpose_(use_conv_transpose)
{
    const int64_t out_channels_ =
        out_channels.value_or(channels);

    if (use_conv_) {
        modules["conv"] =
            std::make_shared<Conv2d>(
                channels,
                out_channels_,
                3,
                1,
                1
            );
    }

    if (use_conv_transpose_) {
        // TODO: not used in Flux2
    }
}

Tensor Upsample2D::forward(
    Scope scope,
    Tensor hidden_states
)
{
    auto shape = hidden_states.shape();

    const int64_t batch = shape[0];
    const int64_t channels = shape[1];
    const int64_t height = shape[2];
    const int64_t width = shape[3];

    // ne0 = width
    // ne1 = height
    // ne2 = channels
    // ne3 = batch
    // Result: [B,C,H,W] -> [B,C,2H,2W]
    hidden_states = Tensor(
        ggml_upscale(
            *scope.context(),
            *hidden_states,
            2,
            GGML_SCALE_MODE_NEAREST
        ),
        Tensor::Shape({
            batch,
            channels,
            height * 2,
            width * 2
        })
    );

    if (use_conv_) {
        hidden_states =
            std::static_pointer_cast<Conv2d>(
                modules["conv"])
            ->forward(scope, hidden_states);
    }

    return hidden_states;
}
