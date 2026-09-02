#include "nn/modules/conv/Conv2d.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

Conv2d::Conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool bias
)
    : stride_(stride),
      padding_(padding),
      bias_(bias)
{
    modules["weight"] =
        std::make_shared<Parameter>(
            Tensor::Shape({
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            })
        );

    if (bias_) {
        modules["bias"] =
            std::make_shared<Parameter>(
                Tensor::Shape({out_channels})
            );
    }
}

Tensor Conv2d::forward(Scope scope, Tensor x) {
    auto weight =
        std::static_pointer_cast<Parameter>(
            modules["weight"])
        ->forward();


    auto conv = ggml_conv_2d_direct(
        *scope.context(),
        *weight,
        *x,
        stride_,
        stride_,
        padding_,
        padding_,
        1,
        1
    );


    auto shape = x.shape();

    shape[1] = weight.shape()[0];

    shape[2] =
        (shape[2] + 2 * padding_ - weight.shape()[2]) /
        stride_ + 1;

    shape[3] =
        (shape[3] + 2 * padding_ - weight.shape()[3]) /
        stride_ + 1;


    Tensor y(
        conv,
        shape
    );


    if (bias_) {
        auto bias =
            std::static_pointer_cast<Parameter>(
                modules["bias"])
            ->forward();


        /*
         * PyTorch bias:
         *
         * [C]
         *
         * Broadcast as:
         *
         * [N,C,H,W]
         *
         * GGML logical order:
         *
         * [W,H,C,N]
         */
        auto bias_4d = bias.reshape(
            Tensor::Shape({
                1,
                bias.shape()[0],
                1,
                1
            })
        );


        y = y + bias_4d;
    }


    return y;
}
