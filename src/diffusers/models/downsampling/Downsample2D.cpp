#include "diffusers/models/downsampling/Downsample2D.hpp"
#include "ggml/Context.hpp"
#include "nn/modules/conv/Conv2d.hpp"

Downsample2D::Downsample2D(
    int64_t channels,
    bool use_conv,
    std::optional<int64_t> out_channels,
    int64_t padding
)
    : use_conv_(use_conv), padding_(padding)
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

Tensor Downsample2D::forward(Scope scope, Tensor hidden_states) {
    if (use_conv_) {
        if (padding_ == 0) {
            auto shape = hidden_states.shape();

            int64_t n = shape[0];
            int64_t c = shape[1];
            int64_t h = shape[2];
            int64_t w = shape[3];

            // F.pad(hidden_states, (0, 1, 0, 1))
            //
            // First pad width:
            // [N,C,H,W] -> [N,C,H,W+1]
            auto w_zeros = Tensor::zeros(
                {n, c, h, 1}
            ).to(hidden_states.dtype());

            hidden_states = Tensor::cat(
                {hidden_states, w_zeros},
                -1
            );

            // Then pad height:
            // [N,C,H,W+1] -> [N,C,H+1,W+1]
            auto h_zeros = Tensor::zeros(
                {n, c, 1, w + 1}
            ).to(hidden_states.dtype());

            hidden_states = Tensor::cat(
                {hidden_states, h_zeros},
                2
            );
        }

        auto conv =  std::static_pointer_cast<Conv2d>(modules["conv"]);

        hidden_states = conv->forward(scope, hidden_states);
    }
    
    // TODO: implement AvgPool2D
    else {
        auto y = ggml_pool_2d(
            *scope.context(),
            *hidden_states,
            GGML_OP_POOL_AVG,
            2, 2,
            2, 2,
            0.0f, 0.0f
        );

        auto shape = hidden_states.shape();
        shape[2] /= 2;
        shape[3] /= 2;

        hidden_states = Tensor(y, shape);
    }

    return hidden_states;
}
