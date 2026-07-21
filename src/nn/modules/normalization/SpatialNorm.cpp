#include "nn/modules/normalization/SpatialNorm.hpp"

#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/conv/Conv2d.hpp"


SpatialNorm::SpatialNorm(
    int64_t f_channels,
    int64_t zq_channels,
    float eps
)
    : eps_(eps)
{
    /*
        self.norm_layer = GroupNorm(
            32,
            f_channels,
            eps
        )
    */

    modules["norm_layer"] =
        std::make_shared<GroupNorm>(
            32,
            f_channels,
            eps,
            false
        );


    /*
        scale projection
    */

    modules["conv_y"] =
        std::make_shared<Conv2d>(
            zq_channels,
            f_channels,
            1,
            1,
            0
        );


    /*
        bias projection
    */

    modules["conv_b"] =
        std::make_shared<Conv2d>(
            zq_channels,
            f_channels,
            1,
            1,
            0
        );
}



Tensor SpatialNorm::forward(
    ggml_context* ctx,
    Tensor x,
    Tensor zq
) {
    auto conv_y =
        std::static_pointer_cast<Conv2d>(
            modules["conv_y"]);


    auto conv_b =
        std::static_pointer_cast<Conv2d>(
            modules["conv_b"]);


    /*
        zq -> scale/bias
    */

    auto y =
        conv_y->forward(
            ctx,
            zq
        );


    auto b =
        conv_b->forward(
            ctx,
            zq
        );


    /*
        normalize input
    */

    auto h =
        std::static_pointer_cast<GroupNorm>(
            modules["norm_layer"])
        ->forward(
            ctx,
            x
        );


    /*
        h * (1+y) + b
    */

    auto ones =
        Tensor::ones(
            ctx,
            y.shape(),
            y.dtype()
        );


    y = y + ones;

    h = h * y;
    h = h + b;


    return h;
}