#include "diffusers/models/normalization/SpatialNorm.hpp"
#include "ggml/Context.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/conv/Conv2d.hpp"

SpatialNorm::SpatialNorm(
    int64_t f_channels,
    int64_t zq_channels
) {
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
            1e-6,
            true
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

Tensor SpatialNorm::forward(Scope scope, Tensor f, Tensor zq) {
    /*
        Python:
            f_size = f.shape[-2:]
            zq = F.interpolate(
                zq,
                size=f_size,
                mode="nearest"
            )
    */
    if (zq.shape()[2] != f.shape()[2] ||
        zq.shape()[3] != f.shape()[3]) {

        auto resized = ggml_interpolate(
            *scope.context(),
            *zq,
            f.shape()[3], // W
            f.shape()[2], // H
            zq.shape()[1], // C
            zq.shape()[0], // N
            GGML_SCALE_MODE_NEAREST
        );

        zq = Tensor(
            resized,
            Tensor::Shape({
                zq.shape()[0],
                zq.shape()[1],
                f.shape()[2],
                f.shape()[3]
            })
        );
    }


    /*
        norm_f = self.norm_layer(f)
    */
    auto norm_f =
        std::static_pointer_cast<GroupNorm>(
            modules["norm_layer"])
        ->forward(scope, f);


    /*
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
    */
    auto y =
        std::static_pointer_cast<Conv2d>(
            modules["conv_y"])
        ->forward(scope, zq);

    auto b =
        std::static_pointer_cast<Conv2d>(
            modules["conv_b"])
        ->forward(scope, zq);


    return norm_f * y + b;
}
