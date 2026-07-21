#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/Parameter.hpp"

#include <cmath>


GroupNorm::GroupNorm(
    int64_t num_groups,
    int64_t num_channels,
    float eps,
    bool affine
)
    : num_groups_(num_groups),
      num_channels_(num_channels),
      eps_(eps),
      affine_(affine)
{
    if (affine_) {
        modules["weight"] =
            std::make_shared<Parameter>(
                Tensor::Shape({num_channels})
            );

        modules["bias"] =
            std::make_shared<Parameter>(
                Tensor::Shape({num_channels})
            );
    }
}


Tensor GroupNorm::forward(
    ggml_context* ctx,
    Tensor x
) {
    /*
     x logical shape:
       [N, C, H, W]

     GGML layout:
       [W, H, C, N]
    */

    const auto shape = x.shape();

    const int64_t batch = shape[0];
    const int64_t channels = shape[1];
    const int64_t height = shape[2];
    const int64_t width = shape[3];


    const int64_t channels_per_group =
        channels / num_groups_;


    /*
       Reshape:

       [N,C,H,W]
           |
       [N,G,C/G,H,W]

       We cannot create a 5D GGML tensor, so flatten
       the normalized dimensions:

       [N,G,C/G*H*W]
    */

    auto grouped = ggml_reshape_3d(
        ctx,
        *x,
        width * height * channels_per_group,
        num_groups_,
        batch
    );


    /*
       Normalize each [C/G*H*W] vector.

       ggml_norm normalizes the first dimension.
    */

    auto normalized = ggml_norm(
        ctx,
        grouped,
        eps_
    );


    auto y = ggml_reshape_4d(
        ctx,
        normalized,
        width,
        height,
        channels,
        batch
    );


    if (affine_) {
        auto weight =
            std::static_pointer_cast<Parameter>(
                modules["weight"])
            ->forward();

        auto bias =
            std::static_pointer_cast<Parameter>(
                modules["bias"])
            ->forward();


        /*
          weight/bias are [C]

          Broadcast across:
            W,H,N
        */

        y = ggml_mul(
            ctx,
            y,
            *weight
        );

        y = ggml_add(
            ctx,
            y,
            *bias
        );
    }


    return Tensor(
        ctx,
        y,
        shape
    );
}