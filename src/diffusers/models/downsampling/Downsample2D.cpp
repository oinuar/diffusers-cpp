#include "diffusers/models/downsampling/Downsample2D.hpp"

#include "nn/modules/conv/Conv2d.hpp"


Downsample2D::Downsample2D(
    int64_t channels,
    int64_t out_channels,
    bool use_conv,
    int64_t padding
)
    : use_conv_(use_conv)
{
    if (out_channels < 0)
        out_channels = channels;


    if (use_conv_) {

        /*
            Diffusers:

            Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=padding
            )
        */

        modules["conv"] =
            std::make_shared<Conv2d>(
                channels,
                out_channels,
                3,
                2,
                padding
            );
    }
}