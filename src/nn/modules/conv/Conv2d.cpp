#include "nn/modules/conv/Conv2d.hpp"
#include "nn/Parameter.hpp"


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


Tensor Conv2d::forward(
    ggml_context* ctx,
    Tensor x
) {
    auto weight =
        std::static_pointer_cast<Parameter>(
            modules["weight"])
        ->forward();

    /*
     * ggml_conv_2d expects:
     *
     * src0: weight
     * src1: input
     *
     * Weight:
     *   [kernel_w, kernel_h, in_channels, out_channels]
     *
     * Input:
     *   [width, height, channels, batch]
     *
     */

    auto y = ggml_conv_2d(
        ctx,
        *weight,
        *x,
        stride_,
        stride_,
        padding_,
        padding_,
        1,
        1
    );


    if (bias_) {
        auto bias =
            std::static_pointer_cast<Parameter>(
                modules["bias"])
            ->forward();

        y = ggml_add(ctx, y, *bias);
    }


    auto shape = x.shape();

    /*
     * PyTorch:
     * NCHW -> N, Cout, Hout, Wout
     *
     * Shape stores PyTorch logical order.
     */
    shape[1] = weight.shape()[0];

    shape[2] =
        (shape[2] + 2 * padding_ - weight.shape()[2]) /
        stride_ + 1;

    shape[3] =
        (shape[3] + 2 * padding_ - weight.shape()[3]) /
        stride_ + 1;


    return Tensor(ctx, y, shape);
}