#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Context.hpp"

GroupNorm::GroupNorm(int64_t num_groups, int64_t num_channels, float eps, bool affine, bool bias)
    : num_groups_(num_groups),
      eps_(eps),
      affine_(affine),
      bias_(bias)
{
    if (affine_) {
        modules["weight"] =
            std::make_shared<Parameter>(
                Tensor::Shape({num_channels})
            );

        if (bias_)
            modules["bias"] =
                std::make_shared<Parameter>(
                    Tensor::Shape({num_channels})
                );
    }
}

Tensor GroupNorm::forward(
    Context& context,
    Tensor input
) {
    auto shape = input.shape();

    const int64_t batch = shape[0];
    const int64_t channels = shape[1];
    const int64_t height = shape[2];
    const int64_t width = shape[3];


    /*
        PyTorch:

        N,C,H,W
          |
          v
        N,G,C/G*H*W

        Let reshape infer the last dimension.
    */
    input = input.reshape({
        batch,
        num_groups_,
        -1
    });


    input = Tensor(
        *context,
        ggml_norm(
            *context,
            *input,
            eps_
        ),
        input.shape()
    );


    input = input.reshape({
        batch,
        channels,
        height,
        width
    });


    if (affine_) {

        auto weight =
            std::static_pointer_cast<Parameter>(
                modules["weight"])
            ->forward();


        weight = weight.reshape({
            1,
            channels,
            1,
            1
        });


        input = input * weight;


        if (bias_) {
            auto bias =
                std::static_pointer_cast<Parameter>(
                    modules["bias"])
                ->forward();


            bias = bias.reshape({
                1,
                channels,
                1,
                1
            });


            input = input + bias;
        }
    }


    return input;
}